.libPaths()
library("remotes")
remotes::install_github("kwundram2602/multisensoR", ref = "dev")
library("multisensoR")
library("terra")
# install torch, session restart might be necessary
if (!torch::torch_is_installed()) torch::install_torch()

data_s2 <- system.file("extdata/subsamples/S2", package = "multisensoR")
data_l8 <- system.file("extdata/subsamples/L8", package = "multisensoR")

# out dir
out <- file.path(getwd(), "ms_out")
# preprocessed tifs
preprocessed <- file.path(out,"preprocessed")

print("Finding pairs...")
pairs      <- find_l8_s2_pairs(data_l8, data_s2, n_pairs = 4)

pairs_proc <- preprocess_pairs(pairs,
                               out_dir  = preprocessed,
                               scale_l8 = c(gain = 0.0000275, offset = -0.2),  
                               scale_s2 = c(gain = 1/10000,   offset = 0))
# dataset 
ds <- landsat_sentinel_dataset(
  landsat_files  = pairs_proc$l8,
  sentinel_files = pairs_proc$s2,
  patch_size = 256, augment = TRUE
)
# data loader
train_dl <- torch::dataloader(ds, batch_size = 1, shuffle = TRUE)

# 3. train model, channels refers to red,green,blue,NIR,
model    <- unet_model(in_channels = 6L, out_channels = 6L)

# 4. train unet model
check_points <- file.path(out,"checkpoints")
history  <- train_unet(model, train_dl, epochs = 5,
                       checkpoint_dir = check_points)

# 5. get last weights and predict
weights <- list.files(check_points)
last = tail(weights, n=1)
last_path = file.path(check_points,last)

pred <- predict_unet(
  model_path   = last_path,
  landsat_path = pairs_proc$l8[[1]],
  out_path     = file.path(out, "prediction_l8_to_s2.tif")
)