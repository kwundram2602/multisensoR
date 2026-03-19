#' Landsat-8 / Sentinel-2 Patch Dataset
#'
#' A `torch` dataset that reads co-registered, aligned Landsat-8 and Sentinel-2
#' GeoTIFF pairs and yields square patches as normalised float tensors.
#'
#' **Assumptions:**
#' - Both files are on the **same pixel grid** (same CRS, resolution, extent).
#'   Use [preprocess_pairs()] to align L8 (30 m) to the S2 (10 m) grid first.
#' - Pixel values are already scaled to reflectance **[0, 1]** (as produced by
#'   the GEE export pipeline).
#' - Both sensors export **6 bands** in matching spectral order:
#'   Blue, Green, Red, NIR, SWIR1, SWIR2.
#'
#' @param landsat_files  Character vector of paths to (aligned) Landsat-8 GeoTIFFs.
#' @param sentinel_files Character vector of paths to Sentinel-2 GeoTIFFs
#'   (same length and order as `landsat_files`).
#' @param patch_size     Side length of square patches in pixels. Default `256`.
#' @param stride         Step size between patch origins. Default equals
#'   `patch_size` (non-overlapping tiles). Set smaller for more patches.
#' @param augment        If `TRUE`, apply random horizontal and vertical flips.
#'
#' @return A `torch::dataset` object usable with [torch::dataloader()].
#' @export
#'
#' @examples
#' \dontrun{
#' pairs <- find_l8_s2_pairs("export/")
#' pairs <- preprocess_pairs(pairs)        # align L8 → 10 m grid
#'
#' ds <- landsat_sentinel_dataset(
#'   landsat_files  = pairs$l8,
#'   sentinel_files = pairs$s2,
#'   patch_size = 256,
#'   augment    = TRUE
#' )
#' dl <- torch::dataloader(ds, batch_size = 4, shuffle = TRUE, num_workers = 2)
#'
#' batch <- coro::collect(dl, 1)[[1]]
#' dim(batch$landsat)   # [4, 6, 256, 256]  — values in [-1, 1]
#' dim(batch$sentinel)  # [4, 6, 256, 256]  — values in [-1, 1]
#' }
landsat_sentinel_dataset <- torch::dataset(
  name = "LandsatSentinelDataset",

  initialize = function(landsat_files, sentinel_files,
                        patch_size = 256L,
                        stride     = NULL,
                        augment    = FALSE) {
    stopifnot(
      "landsat_files and sentinel_files must have the same length" =
        length(landsat_files) == length(sentinel_files),
      "landsat_files must be non-empty" =
        length(landsat_files) > 0
    )

    self$landsat_files  <- landsat_files
    self$sentinel_files <- sentinel_files
    self$patch_size     <- as.integer(patch_size)
    self$stride         <- as.integer(if (is.null(stride)) patch_size else stride)
    self$augment        <- augment

    # Pre-compute all valid (file_id, row_start, col_start) combinations
    self$index <- .build_patch_index(landsat_files, self$patch_size, self$stride)

    if (nrow(self$index) == 0) {
      stop("No valid patches found. Check patch_size vs. image dimensions.")
    }
    message("Dataset ready: ", nrow(self$index), " patches from ",
            length(landsat_files), " image pair(s).")
  },

  .length = function() {
    nrow(self$index)
  },

  .getitem = function(i) {
    idx <- self$index[i, ]
    p   <- self$patch_size

    ls_r <- terra::rast(self$landsat_files[idx$file_id])
    s2_r <- terra::rast(self$sentinel_files[idx$file_id])

    ls_t <- .crop_to_tensor(ls_r, idx$row, idx$col, p)
    s2_t <- .crop_to_tensor(s2_r, idx$row, idx$col, p)

    # Valid-pixel mask: a pixel is valid only if ALL bands are non-NA in
    # both images.  Shape [1, H, W], dtype float (1 = valid, 0 = NoData).
    ls_valid <- (ls_t != 0)$to(dtype = torch::torch_float())$min(dim = 1L, keepdim = TRUE)[[1]]
    s2_valid <- (s2_t != 0)$to(dtype = torch::torch_float())$min(dim = 1L, keepdim = TRUE)[[1]]
    mask <- ls_valid * s2_valid   # union: valid only where both are valid

    # Normalise [0, 1] → [-1, 1]
    ls_t <- ls_t * 2 - 1
    s2_t <- s2_t * 2 - 1

    # Random augmentation (applied identically to input, target and mask)
    if (self$augment) {
      if (runif(1) > 0.5) {
        ls_t <- torch::torch_flip(ls_t, 3L)
        s2_t <- torch::torch_flip(s2_t, 3L)
        mask <- torch::torch_flip(mask, 3L)
      }
      if (runif(1) > 0.5) {
        ls_t <- torch::torch_flip(ls_t, 2L)
        s2_t <- torch::torch_flip(s2_t, 2L)
        mask <- torch::torch_flip(mask, 2L)
      }
    }

    list(landsat = ls_t, sentinel = s2_t, mask = mask)
  }
)

# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------

#' Build a data.frame of valid patch origins for all image files
#' @noRd
.build_patch_index <- function(landsat_files, patch_size, stride) {
  p <- as.integer(patch_size)
  s <- as.integer(stride)

  entries <- lapply(seq_along(landsat_files), function(fid) {
    r    <- terra::rast(landsat_files[[fid]])
    nr   <- terra::nrow(r)
    nc   <- terra::ncol(r)

    row_starts <- seq(1L, nr - p + 1L, by = s)
    col_starts <- seq(1L, nc - p + 1L, by = s)

    if (length(row_starts) == 0 || length(col_starts) == 0) return(NULL)

    expand.grid(file_id = fid, row = row_starts, col = col_starts)
  })

  entries <- Filter(Negate(is.null), entries)
  if (length(entries) == 0) return(data.frame())
  do.call(rbind, entries)
}

#' Extract a patch from a SpatRaster and return a float tensor [C, H, W]
#' @noRd
.crop_to_tensor <- function(rast_obj, row_start, col_start, patch_size) {
  p  <- patch_size
  xr <- terra::xres(rast_obj)
  yr <- terra::yres(rast_obj)

  x_min <- terra::xFromCol(rast_obj, col_start)           - xr / 2
  x_max <- terra::xFromCol(rast_obj, col_start + p - 1L)  + xr / 2
  y_max <- terra::yFromRow(rast_obj, row_start)            + yr / 2
  y_min <- terra::yFromRow(rast_obj, row_start + p - 1L)  - yr / 2

  e     <- terra::ext(x_min, x_max, y_min, y_max)
  patch <- terra::crop(rast_obj, e)

  mat <- terra::as.array(patch)      # [H, W, C]
  mat[is.na(mat)] <- 0               # mask out NoData as 0 (pre-normalise)
  arr <- aperm(mat, c(3L, 1L, 2L))  # [C, H, W]
  torch::torch_tensor(arr, dtype = torch::torch_float())
}
