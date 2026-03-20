#' Train the U-Net on Landsat-8 → Sentinel-2 image translation
#'
#' Runs a supervised training loop: Landsat-8 patches are the model input
#' (predictors) and Sentinel-2 patches are the reconstruction target.
#' Loss is L1 (pixel-wise mean absolute error).
#'
#' @param model         An `nn_module` returned by [unet_model()].
#' @param train_dl      A `torch::dataloader` whose items contain `$landsat`
#'   and `$sentinel` tensors (produced by [landsat_sentinel_dataset()]).
#' @param epochs        Integer. Number of training epochs. Default `10`.
#' @param optimizer     An `torch` optimiser instance.
#' @param val_dl        Optional validation `dataloader` .
#' @param device        A `torch_device` or character string (`"cpu"`, `"cuda"`).
#' @param checkpoint_dir Character or `NULL`. If a directory path is given,
#'   the model state is saved as `unet_epoch_<N>.pt` after every epoch.
#' @param verbose       Logical. Print per-epoch progress? Default `TRUE`.
#'
#' @return A `data.frame` with columns `epoch`, `train_loss`, and (if
#'   `val_dl` is provided) `val_loss`. Returned invisibly.
#' @export
train_unet <- function(model,
                       train_dl,
                       epochs         = 10L,
                       optimizer      = NULL,
                       val_dl         = NULL,
                       device         = NULL,
                       checkpoint_dir = NULL,
                       verbose        = TRUE) {

  # Device 
  if (is.null(device)) {
    device <- if (torch::cuda_is_available()) "cuda" else "cpu"
  }
  device <- torch::torch_device(device)
  if (verbose) message("Training on: ", device$type)

  model <- model$to(device = device)

  # optimizer 
  if (is.null(optimizer)) {
    optimizer <- torch::optim_adam(model$parameters, lr = 1e-4)
  }

  # Masked L1: mean absolute error only over valid (non-NoData) pixels.
  # Falls back to standard L1 when no mask is present in the batch.
  masked_l1 <- function(pred, target, mask = NULL) {
    diff <- torch::torch_abs(pred - target)
    if (is.null(mask)) return(diff$mean())
    # mask: [B, 1, H, W] — broadcast across channels
    valid <- (mask > 0)$expand_as(diff)
    if (valid$sum()$item() == 0L) return(torch::torch_tensor(0))
    diff[valid]$mean()
  }

  # Checkpoint directory 
  if (!is.null(checkpoint_dir) && !dir.exists(checkpoint_dir)) {
    dir.create(checkpoint_dir, recursive = TRUE)
  }

  # Training history 
  history <- data.frame(
    epoch      = integer(0),
    train_loss = numeric(0),
    val_loss   = numeric(0)
  )

  # Epoch loop 
  for (ep in seq_len(as.integer(epochs))) {

    # Training phase 
    model$train()
    train_losses <- numeric(0)

    coro::loop(for (batch in train_dl) {
      x    <- batch$landsat$to(device = device)   # [B, C_l8, H, W]
      y    <- batch$sentinel$to(device = device)  # [B, C_s2, H, W]
      mask <- if (!is.null(batch$mask))
                batch$mask$to(device = device)    # [B, 1,    H, W]
              else NULL

      optimizer$zero_grad()
      pred <- model(x)
      loss <- masked_l1(pred, y, mask)
      loss$backward()
      optimizer$step()

      train_losses <- c(train_losses, loss$item())
    })

    mean_train <- mean(train_losses)

    # Validation phase 
    mean_val <- NA_real_
    if (!is.null(val_dl)) {
      model$eval()
      val_losses <- numeric(0)

      torch::with_no_grad({
        coro::loop(for (batch in val_dl) {
          x    <- batch$landsat$to(device = device)
          y    <- batch$sentinel$to(device = device)
          mask <- if (!is.null(batch$mask))
                    batch$mask$to(device = device)
                  else NULL

          pred      <- model(x)
          val_loss  <- masked_l1(pred, y, mask)
          val_losses <- c(val_losses, val_loss$item())
        })
      })

      mean_val <- mean(val_losses)
    }

    # Log , history
    history <- rbind(history, data.frame(
      epoch      = ep,
      train_loss = mean_train,
      val_loss   = mean_val
    ))

    if (verbose) {
      if (is.na(mean_val)) {
        message(sprintf("Epoch %3d / %d  |  train_loss: %.5f",
                        ep, epochs, mean_train))
      } else {
        message(sprintf("Epoch %3d / %d  |  train_loss: %.5f  |  val_loss: %.5f",
                        ep, epochs, mean_train, mean_val))
      }
    }

    # Checkpoint
    if (!is.null(checkpoint_dir)) {
      ckpt_path <- file.path(checkpoint_dir,
                             sprintf("unet_epoch_%03d.pt", ep))
      torch::torch_save(model$state_dict(), ckpt_path)
      if (verbose) message("  Checkpoint saved: ", ckpt_path)
    }
  }
  invisible(history)
}
