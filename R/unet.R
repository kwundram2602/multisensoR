#' U-Net Encoder Block
#'
#' A single downsampling block: Conv2d -> BatchNorm -> LeakyReLU.
#' The first encoder block skips BatchNorm (standard U-Net convention).
#'
#' @param in_channels Number of input channels.
#' @param out_channels Number of output channels.
#' @param use_batchnorm Whether to apply BatchNorm (default TRUE).
#'
#' @return An `nn_module` instance.
#' @export
unet_encoder_block <- torch::nn_module(
  classname = "UNetEncoderBlock",
  initialize = function(in_channels, out_channels, use_batchnorm = TRUE) {
    self$use_batchnorm <- use_batchnorm
    self$conv <- torch::nn_conv2d(
      in_channels, out_channels,
      kernel_size = 4, stride = 2, padding = 1, bias = !use_batchnorm
    )
    if (use_batchnorm) {
      self$bn <- torch::nn_batch_norm2d(out_channels)
    }
    self$act <- torch::nn_leaky_relu(0.2, inplace = TRUE)
  },
  forward = function(x) {
    x <- self$conv(x)
    if (self$use_batchnorm) x <- self$bn(x)
    self$act(x)
  }
)

#' U-Net Decoder Block
#' @param in_channels Number of input channels (after skip-concat).
#' @param out_channels Number of output channels.
#' @param use_dropout Whether to apply 50% dropout (used in first decoder blocks).
#'
#' @return An `nn_module` instance.
#' @export
unet_decoder_block <- torch::nn_module(
  classname = "UNetDecoderBlock",
  initialize = function(in_channels, out_channels, use_dropout = FALSE) {
    self$use_dropout <- use_dropout
    self$conv <- torch::nn_conv_transpose2d(
      in_channels, out_channels,
      kernel_size = 4, stride = 2, padding = 1, bias = FALSE
    )
    self$bn   <- torch::nn_batch_norm2d(out_channels)
    self$act  <- torch::nn_relu(inplace = TRUE)
    if (use_dropout) {
      self$drop <- torch::nn_dropout(0.5)
    }
  },
  forward = function(x, skip) {
    x <- self$act(self$bn(self$conv(x)))
    if (self$use_dropout) x <- self$drop(x)
    torch::torch_cat(list(x, skip), dim = 2L)
  }
)

#' U-Net Generator
#'
#' A U-Net with 4 encoder blocks, a bottleneck, and 4 decoder blocks.
#' Landsat-8 imagery (default 6 bands) to Sentinel-2 (default 6 bands).
#'
#' Architecture (default filters = 64):
#' @param in_channels  Number of input channels. Default `6` (Landsat-8 bands).
#' @param out_channels Number of output channels. Default `6` (Sentinel-2 bands).
#' @param filters      Base number of filters. Default `64`.
#'
#' @return An `nn_module` instance.
#' @export
unet_model <- torch::nn_module(
  classname = "UNet",
  initialize = function(in_channels = 6L, out_channels = 6L, filters = 64L) {
    f <- filters

    # Encoder  
    self$enc1 <- unet_encoder_block(in_channels, f,     use_batchnorm = FALSE)
    self$enc2 <- unet_encoder_block(f,           f * 2)
    self$enc3 <- unet_encoder_block(f * 2,       f * 4)
    self$enc4 <- unet_encoder_block(f * 4,       f * 8)

    # Bottleneck
    self$bottleneck <- torch::nn_sequential(
      torch::nn_conv2d(f * 8, f * 8, kernel_size = 4, stride = 2, padding = 1, bias = FALSE),
      torch::nn_relu(inplace = TRUE)
    )

    # Decoder
    self$dec4 <- unet_decoder_block(f * 8,  f * 8, use_dropout = TRUE)
    self$dec3 <- unet_decoder_block(f * 16, f * 4, use_dropout = FALSE)
    self$dec2 <- unet_decoder_block(f * 8,  f * 2, use_dropout = FALSE)
    self$dec1 <- unet_decoder_block(f * 4,  f,     use_dropout = FALSE)

    # Output layer
    self$output_conv <- torch::nn_sequential(
      torch::nn_conv_transpose2d(f + f, out_channels,
                                 kernel_size = 4, stride = 2, padding = 1),
      torch::nn_tanh()
    )
  }, # forward layers : enocers, bottleneck, decoders, output layer
  forward = function(x) {
    e1 <- self$enc1(x)
    e2 <- self$enc2(e1)
    e3 <- self$enc3(e2)
    e4 <- self$enc4(e3)

    b  <- self$bottleneck(e4)

    d4 <- self$dec4(b,  e4)
    d3 <- self$dec3(d4, e3)
    d2 <- self$dec2(d3, e2)
    d1 <- self$dec1(d2, e1)

    self$output_conv(d1)
  }
)
