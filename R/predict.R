#' Predict Sentinel-2 from a Landsat-8 image using a trained U-Net
#'
#' Loads a saved U-Net checkpoint (state dict `.pt` file), tiles the input
#' Landsat-8 GeoTIFF into non-overlapping patches, runs inference on each
#' patch, and reassembles the predictions into a GeoTIFF with the same
#' spatial reference as the input.
#'
#' @param model_path   Path to a `.pt` checkpoint written by [train_unet()].
#' @param landsat_path Path to a Landsat-8 GeoTIFF whose pixel values are
#'   scaled to reflectance \[0, 1\].
#' @param out_path     Output path for the predicted GeoTIFF.
#' @param in_channels  Number of input channels the model was trained with.
#'   Default `6L`.
#' @param out_channels Number of output channels. Default `6L`.
#' @param patch_size   Patch side length in pixels (must match training).
#'   Default `256L`.
#' @param device       `"cpu"` or `"cuda"`. Defaults to CUDA if available.
#'
#' @return A [terra::SpatRaster] with `out_channels` layers and the same
#'   extent / CRS as the input. Pixel values are in \[0, 1\] (reflectance).
#'   Pixels not covered by any full patch are `NA`.
#' @export
#'
#' @examples
#' \dontrun{
#' pred <- predict_unet(
#'   model_path   = "checkpoints/unet_epoch_005.pt",
#'   landsat_path = "data/L8/scene.tif",
#'   out_path     = "output/pred_s2.tif"
#' )
#' terra::plot(pred)
#' }
predict_unet <- function(model_path,
                         landsat_path,
                         out_path     = NULL,
                         in_channels  = 6L,
                         out_channels = 6L,
                         patch_size   = 256L,
                         device       = NULL) {

  stopifnot(file.exists(model_path), file.exists(landsat_path))

  patch_size <- as.integer(patch_size)

  # ── Device ──────────────────────────────────────────────────────────────────
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  }
  dev <- torch::torch_device(device)
  message("Inference on: ", device)

  # ── Model ───────────────────────────────────────────────────────────────────
  model <- unet_model(in_channels = in_channels, out_channels = out_channels)
  model$load_state_dict(torch::torch_load(model_path))
  model <- model$to(device = dev)
  model$eval()

  # ── Input raster ────────────────────────────────────────────────────────────
  l8_rast <- terra::rast(landsat_path)
  nr <- terra::nrow(l8_rast)
  nc <- terra::ncol(l8_rast)

  row_starts <- seq(1L, nr - patch_size + 1L, by = patch_size)
  col_starts <- seq(1L, nc - patch_size + 1L, by = patch_size)

  n_patches <- length(row_starts) * length(col_starts)
  message("Processing ", n_patches, " patch(es)...")

  xr <- terra::xres(l8_rast)
  yr <- terra::yres(l8_rast)

  patch_list <- vector("list", n_patches)
  idx <- 1L

  # ── Inference loop ───────────────────────────────────────────────────────────
  torch::with_no_grad({
    for (r0 in row_starts) {
      for (c0 in col_starts) {

        x_t <- .crop_to_tensor(l8_rast, r0, c0, patch_size)  # [C, H, W]
        x_t <- (x_t * 2 - 1)$unsqueeze(1L)$to(device = dev)  # [1, C, H, W]

        pred <- model(x_t)$squeeze(1L)                        # [C_out, H, W]
        pred <- ((pred + 1) / 2)$cpu()                        # [0, 1]

        # arr: [C_out, H, W] → matrix [H*W, C_out] in terra's row-major order
        arr <- as.array(pred)                                  # [C_out, H, W]
        mat <- do.call(cbind, lapply(seq_len(out_channels), function(b) {
          as.vector(t(arr[b, , ]))   # t() transposes [H,W] → [W,H]; as.vector then
        }))                          # yields row-major: (r1,c1),(r1,c2),...

        # Build a SpatRaster for this patch with matching extent / CRS
        p_ext <- terra::ext(
          terra::xFromCol(l8_rast, c0)              - xr / 2,
          terra::xFromCol(l8_rast, c0 + patch_size - 1L) + xr / 2,
          terra::yFromRow(l8_rast, r0 + patch_size - 1L) - yr / 2,
          terra::yFromRow(l8_rast, r0)              + yr / 2
        )
        pr <- terra::rast(
          ext   = p_ext,
          nrows = patch_size, ncols = patch_size,
          nlyrs = out_channels,
          crs   = terra::crs(l8_rast)
        )
        terra::values(pr) <- mat

        patch_list[[idx]] <- pr
        idx <- idx + 1L
      }
    }
  })

  # ── Reassemble patches ───────────────────────────────────────────────────────
  pred_rast <- if (length(patch_list) == 1L) {
    patch_list[[1L]]
  } else {
    do.call(terra::merge, patch_list)
  }

  # ── Output ──────────────────────────────────────────────────────────────────
  if (!is.null(out_path)) {
    terra::writeRaster(pred_rast, out_path, overwrite = TRUE)
    message("Prediction saved: ", out_path)
  }

  invisible(pred_rast)
}
