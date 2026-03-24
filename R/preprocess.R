#' Find Landsat-8 / Sentinel-2 GeoTIFF pairs
#'
#' Matches Landsat-8 and Sentinel-2 GeoTIFFs from two separate directories by
#' their pair ID (`pair_XX`) in the filename.
#'
#' Expected layout:
#' ```
#' l8_dir/
#'   pair_01_Landsat8_20220616.tif
#'   pair_02_Landsat8_20220618.tif
#'   ...
#' s2_dir/
#'   pair_01_S2_20220617.tif
#'   pair_02_S2_20220617.tif
#'   ...
#' ```
#'
#' @param l8_dir   Character. Directory containing Landsat-8 GeoTIFF files.
#' @param s2_dir   Character. Directory containing Sentinel-2 GeoTIFF files.
#' @param n_pairs  Integer or `NULL`. Maximum number of pairs to return. `NULL`
#'   (default) returns all pairs found.
#'
#' @return A data.frame with columns `pair_id`, `l8`, and `s2` (absolute paths),
#'   one row per valid pair. Raises an error if no pairs are found.
#' @export
find_l8_s2_pairs <- function(l8_dir, s2_dir, n_pairs = NULL) {
  l8_dir <- normalizePath(l8_dir, mustWork = TRUE)
  s2_dir <- normalizePath(s2_dir, mustWork = TRUE)

  l8_files <- list.files(l8_dir, pattern = "\\.tif$", full.names = TRUE)
  s2_files <- list.files(s2_dir, pattern = "\\.tif$", full.names = TRUE)

  if (length(l8_files) == 0) stop("No Landsat8 TIF files found in: ", l8_dir)
  if (length(s2_files) == 0) stop("No S2 TIF files found in: ", s2_dir)

  # Limit each sensor independently before matching
  if (!is.null(n_pairs)) {
    n_pairs  <- as.integer(n_pairs)
    if (n_pairs < 1) stop("n_pairs must be >= 1.")
    l8_files <- head(sort(l8_files), n_pairs)
    s2_files <- head(sort(s2_files), n_pairs)
  }

  # Extract pair ID (e.g. "pair_01") from filename
  get_pair_id <- function(paths) {
    regmatches(basename(paths), regexpr("pair_\\d+", basename(paths), ignore.case = TRUE))
  }

  l8_ids <- get_pair_id(l8_files)
  s2_ids <- get_pair_id(s2_files)

  shared_ids <- intersect(l8_ids, s2_ids)
  if (length(shared_ids) == 0) stop("No matching pair IDs found.")

  rows <- lapply(shared_ids, function(id) {
    data.frame(
      pair_id = id,
      l8      = l8_files[l8_ids == id],
      s2      = s2_files[s2_ids == id],
      stringsAsFactors = FALSE
    )
  })

  result <- do.call(rbind, rows)
  result[order(result$pair_id), ]
}


#' Check and print the value scaling of input GeoTIFF pairs
#'
#' Computes per-band min/max for each file in `pairs` and prints a summary
#' table. Warns if values fall outside the expected `[expected_min, expected_max]`
#' range (default `[0, 1]`.
#'
#' @param pairs        data.frame returned by [find_l8_s2_pairs()].
#' @param expected_min Numeric. Expected minimum value. Default `0`.
#' @param expected_max Numeric. Expected maximum value. Default `1`.
#' @param sample_frac  Numeric in `(0, 1]`. Fraction of pixels to sample per
#'   band (speeds up large images). Default `0.01` (1%).
#'
#' @return A data.frame with columns `pair_id`, `sensor`, `band`, `min`, `max`
#'   (invisibly). Printed as a formatted table.
#' @export
check_scaling <- function(pairs, expected_min = 0, expected_max = 1,
                          sample_frac = 0.01) {
  results <- lapply(seq_len(nrow(pairs)), function(i) {
    rbind(
      .scaling_summary(pairs$l8[[i]], pairs$pair_id[[i]], "L8", sample_frac),
      .scaling_summary(pairs$s2[[i]], pairs$pair_id[[i]], "S2", sample_frac)
    )
  })
  out <- do.call(rbind, results)

  # Print formatted table
  cat(sprintf("\n%-10s %-4s %-6s %8s %8s  %s\n",
              "pair_id", "sen", "band", "min", "max", "status"))
  cat(strrep("-", 52), "\n")
  for (j in seq_len(nrow(out))) {
    row    <- out[j, ]
    in_range <- row$min >= expected_min & row$max <= expected_max
    status <- if (in_range) "OK" else sprintf("WARN  [expected %g, %g]", expected_min, expected_max)
    cat(sprintf("%-10s %-4s %-6s %8.4f %8.4f  %s\n",
                row$pair_id, row$sensor, row$band, row$min, row$max, status))
  }
  cat("\n")

  invisible(out)
}

#' @noRd
.scaling_summary <- function(path, pair_id, sensor, sample_frac) {
  r       <- terra::rast(path)
  n_samp  <- max(1L, as.integer(terra::ncell(r) * sample_frac))
  samples <- terra::spatSample(r, size = n_samp, method = "random",
                               na.rm = TRUE, warn = FALSE)

  band_names <- names(r)
  do.call(rbind, lapply(seq_along(band_names), function(b) {
    vals <- samples[[b]]
    data.frame(
      pair_id = pair_id,
      sensor  = sensor,
      band    = band_names[[b]],
      min     = min(vals, na.rm = TRUE),
      max     = max(vals, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  }))
}


#' Align a Landsat-8 image to the Sentinel-2 reference grid
#' @param l8_path  Character. Path to the Landsat-8 GeoTIFF (reflectance [0, 1]).
#' @param s2_path  Character. Path to the Sentinel-2 GeoTIFF (reflectance [0, 1]).
#' @param out_path Character. Output path for the resampled L8 file. Defaults to
#'   `<l8_basename>_aligned.tif` in the same directory.
#' @param method   Character. Resampling method for [terra::resample()].
#'   `"bilinear"` (default) preserves continuous reflectance values.
#'
#' @return The `out_path` string, invisibly.
#' @export
align_l8_to_s2 <- function(l8_path, s2_path,
                            out_path = sub("\\.tif$", "_aligned.tif",
                                          l8_path, ignore.case = TRUE),
                            method = "bilinear") {
  l8 <- terra::rast(l8_path)
  s2 <- terra::rast(s2_path)

  # Reproject L8 to S2 CRS if they differ
  if (!terra::same.crs(l8, s2)) {
    message("Reprojecting L8 from ", terra::crs(l8, describe = TRUE)$name,
            " to S2 CRS ...")
    l8 <- terra::project(l8, terra::crs(s2), method = method)
  }

  # Crop to shared extent (intersection) 
  shared_ext <- terra::intersect(terra::ext(l8), terra::ext(s2))
  if (is.null(shared_ext)) stop("L8 and S2 images do not overlap: ", l8_path)
  l8 <- terra::crop(l8, shared_ext)
  s2 <- terra::crop(s2, shared_ext)

  # Resample L8 onto the S2 pixel grid (10 m)
  l8_aligned <- terra::resample(l8, s2, method = method)

  terra::writeRaster(l8_aligned, out_path, overwrite = TRUE,
                     datatype = "FLT4S", gdal = c("COMPRESS=LZW"))
  message("Aligned L8 written to: ", out_path)
  invisible(out_path)
}
#' Apply a shared NoData mask to an aligned L8 / S2 pair
#'
#' A pixel is set to `NA` in **both** images if it is `NA` in **either** image
#' (union of NoData masks). 
#'
#' Requires both rasters to already share the same grid (run [align_l8_to_s2()]
#' first).
#' @param l8_path  Character. Path to the aligned Landsat-8 GeoTIFF.
#' @param s2_path  Character. Path to the Sentinel-2 GeoTIFF.
#' @param out_l8   Output path for the masked L8 file. Defaults to
#'   `<basename>_masked.tif` next to the input.
#' @param out_s2   Output path for the masked S2 file. Defaults to
#'   `<basename>_masked.tif` next to the input.
#'
#' @return A named list with elements `l8` and `s2` (output paths), invisibly.
#' @export
mask_nodata_pair <- function(l8_path, s2_path,
                             out_l8 = sub("\\.tif$", "_masked.tif", l8_path, ignore.case = TRUE),
                             out_s2 = sub("\\.tif$", "_masked.tif", s2_path, ignore.case = TRUE),
                             zero_as_nodata = TRUE) {
  l8 <- terra::rast(l8_path)
  s2 <- terra::rast(s2_path)

  # NA in any band → NA for that pixel (sum propagates NA)
  na_l8 <- is.na(terra::app(l8, "sum"))
  na_s2 <- is.na(terra::app(s2, "sum"))

  # treat all-zero L8 pixels as NoData
  if (zero_as_nodata) {
    zero_l8 <- terra::app(l8, "sum") == 0
    na_l8   <- na_l8 | zero_l8
  }

  # Union: mask out wherever either image has NoData or L8 is zero-filled
  combined_na <- na_l8 | na_s2
  n_masked    <- terra::global(combined_na, "sum", na.rm = TRUE)[[1]]
  message(sprintf("Masking %d pixels (%.1f%%) as NoData in both images.",
                  n_masked, 100 * n_masked / terra::ncell(l8)))

  l8_masked <- terra::mask(l8, combined_na, maskvalue = TRUE)
  s2_masked <- terra::mask(s2, combined_na, maskvalue = TRUE)

  terra::writeRaster(l8_masked, out_l8, overwrite = TRUE,
                     datatype = "FLT4S", gdal = c("COMPRESS=LZW"))
  terra::writeRaster(s2_masked, out_s2, overwrite = TRUE,
                     datatype = "FLT4S", gdal = c("COMPRESS=LZW"))

  invisible(list(l8 = out_l8, s2 = out_s2))
}


#' Preprocess all image pairs
#'
#' Pipeline per pair:
#' 1. **Scale** – apply `gain * x + offset` to bring values to \[0, 1\]
#' 2. **Align** – resample L8 onto the S2 pixel grid ([align_l8_to_s2()])
#' 3. **Mask**  – harmonise NoData between sensors ([mask_nodata_pair()])
#'
#' @param pairs       data.frame returned by [find_l8_s2_pairs()].
#' @param out_dir     Output directory. Defaults to `preprocessed_pairs/` next
#'   to the L8 input files.
#' @param scale_l8    Named numeric vector `c(gain, offset)` applied as
#'   `gain * DN + offset` to scale L8 to \[0, 1\].
#'   Default: `c(gain = 0.0000275, offset = -0.2)` (Landsat C2 L2 formula).
#'   Set to `NULL` to skip scaling.
#' @param scale_s2    Named numeric vector `c(gain, offset)` for S2.
#'   Default: `c(gain = 1/10000, offset = 0)` (divide by 10 000).
#'   Set to `NULL` to skip scaling.
#' @param mask_nodata Logical. Apply shared NoData mask? Default `TRUE`.
#' @param overwrite   Logical. Re-process existing files? Default `FALSE`.
#' @param ...         Additional arguments forwarded to [align_l8_to_s2()].
#'
#' @return A data.frame with columns `l8`, `s2` (processed paths), and `l8_orig`.
#' @export
#'
#' @examples
#' \dontrun{
#' pairs <- find_l8_s2_pairs("data/L8/", "data/S2/")
#'
#' # default: L8 C2 L2 formula + S2 / 10000
#' pairs_proc <- preprocess_pairs(pairs)
#'
#' # custom scaling
#' pairs_proc <- preprocess_pairs(pairs,
#'   scale_l8 = c(gain = 0.0000275, offset = -0.2),
#'   scale_s2 = c(gain = 1/10000,   offset = 0))
#'
#' # no scaling (data already in [0,1])
#' pairs_proc <- preprocess_pairs(pairs, scale_l8 = NULL, scale_s2 = NULL)
#' }
preprocess_pairs <- function(pairs,
                             out_dir     = NULL,
                             scale_l8    = c(gain = 0.0000275, offset = -0.2),
                             scale_s2    = c(gain = 1 / 10000, offset = 0),
                             mask_nodata = TRUE,
                             overwrite   = FALSE,
                             ...) {
  if (is.null(out_dir)) {
    out_dir <- file.path(dirname(pairs$l8[[1]]), "preprocessed_pairs")
  }
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  pairs$l8_orig <- pairs$l8

  for (i in seq_len(nrow(pairs))) {
    l8_in <- pairs$l8[[i]]
    s2_in <- pairs$s2[[i]]

    # Step 1: scale L8 to [0, 1] and write to out_dir
    l8_scaled <- file.path(out_dir,
                           sub("\\.tif$", "_scaled.tif", basename(l8_in), ignore.case = TRUE))
    if (!overwrite && file.exists(l8_scaled)) {
      message("Skipping L8 scaling (exists): ", basename(l8_scaled))
    } else {
      .scale_and_write(l8_in, l8_scaled, scale_l8)
    }

    # Step 1b: scale S2 to [0, 1] and write to out_dir
    s2_scaled <- file.path(out_dir,
                           sub("\\.tif$", "_scaled.tif", basename(s2_in), ignore.case = TRUE))
    if (!overwrite && file.exists(s2_scaled)) {
      message("Skipping S2 scaling (exists): ", basename(s2_scaled))
    } else {
      .scale_and_write(s2_in, s2_scaled, scale_s2)
    }

    # Step 2: align L8 to S2 grid
    l8_aligned <- sub("_scaled\\.tif$", "_scaled_aligned.tif", l8_scaled, ignore.case = TRUE)
    if (!overwrite && file.exists(l8_aligned)) {
      message("Skipping alignment (exists): ", basename(l8_aligned))
    } else {
      align_l8_to_s2(l8_scaled, s2_scaled, out_path = l8_aligned, ...)
    }

    # Step 3 : harmonise NoData masks
    if (mask_nodata) {
      l8_out <- sub("_aligned\\.tif$", "_aligned_masked.tif", l8_aligned, ignore.case = TRUE)
      s2_out <- sub("_scaled\\.tif$",  "_scaled_masked.tif",  s2_scaled,  ignore.case = TRUE)

      if (!overwrite && file.exists(l8_out) && file.exists(s2_out)) {
        message("Skipping masking (exists): ", basename(l8_out))
      } else {
        mask_nodata_pair(l8_aligned, s2_scaled, out_l8 = l8_out, out_s2 = s2_out)
      }
      pairs$l8[[i]] <- l8_out
      pairs$s2[[i]] <- s2_out
    } else {
      pairs$l8[[i]] <- l8_aligned
      pairs$s2[[i]] <- s2_scaled
    }
  }

  pairs
}

#' Apply gain/offset scaling to a raster and write to disk
#' @noRd
.scale_and_write <- function(in_path, out_path, scale) {
  r <- terra::rast(in_path)
  if (!is.null(scale)) {
    r <- r * scale[["gain"]] + scale[["offset"]]
    r <- terra::clamp(r, 0, 1)   # clip artefacts outside [0, 1]
    message(sprintf("Scaled %s  (gain=%g, offset=%g) → [0, 1]",
                    basename(in_path), scale[["gain"]], scale[["offset"]]))
  }
  terra::writeRaster(r, out_path, overwrite = TRUE,
                     datatype = "FLT4S", gdal = c("COMPRESS=LZW"))
  invisible(out_path)
}
