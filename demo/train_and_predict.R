.libPaths()
library("remotes")

remotes::install_github("kwundram2602/multisensoR", ref = "v0.1.0",force = TRUE)
library("multisensoR")
library("terra")
# install torch, session restart might be necessary
if (!torch::torch_is_installed()) torch::install_torch()

N_PAIRS = 4 # 1 - 16 # set to lower values for short test
EPOCHS = 5 # set to lower values for short test
BATCH_SIZE = 10

# use sample data or set to your own data directories
# find_l8_s2_pairs (see below) matches the id by "pair_\\d+" so files should be named like "pair_1_l8.tif" and "pair_1_s2.tif"
data_s2 <- system.file("extdata/subsamples/S2", package = "multisensoR")
data_l8 <- system.file("extdata/subsamples/L8", package = "multisensoR")
# out dir
out <- file.path(getwd(), "ms_out")
# preprocessed tifs
print("Output directory :")
print(out)
preprocessed <- file.path(out,"preprocessed")

print("Finding pairs...")
pairs      <- find_l8_s2_pairs(data_l8, data_s2, n_pairs = N_PAIRS)
print("Preprocessing pairs...")
pairs_proc <- preprocess_pairs(pairs,
                               out_dir  = preprocessed,
                               scale_l8 = c(gain = 0.0000275, offset = -0.2),  
                               scale_s2 = c(gain = 1/10000,   offset = 0))
# Plot first input pair → saved to ms_out/input_pair.png
print("Plotting first pair...")
l8_in  <- terra::rast(pairs_proc$l8[[1]])
s2_ref <- terra::rast(pairs_proc$s2[[1]])
input_plot <- file.path(out, "input_pair.png")
png(input_plot, width = 1200, height = 500, res = 120)
par(mfrow = c(1, 2), mar = c(1, 1, 2.5, 1))
terra::plotRGB(l8_in,  r = 3, g = 2, b = 1, stretch = "lin",
               main = "L8 input (preprocessed)")
terra::plotRGB(s2_ref, r = 3, g = 2, b = 1, stretch = "lin",
               main = "S2 target (preprocessed)")
dev.off()
print(paste("Input pair plot saved to:", input_plot))
# dataset 
print("Creating dataset and dataloader...")
ds <- landsat_sentinel_dataset(
  landsat_files  = pairs_proc$l8,
  sentinel_files = pairs_proc$s2,
  patch_size = 256, augment = TRUE
)
# data loader
train_dl <- torch::dataloader(ds, batch_size = BATCH_SIZE, shuffle = TRUE)

# train model, channels refers to red,green,blue,NIR,SWIR1,SWIR2 for the sample data
print("Creating model and training...")
model    <- unet_model(in_channels = 6L, out_channels = 6L)

# train unet model
check_points <- file.path(out,"checkpoints")
history  <- train_unet(model, train_dl, epochs = EPOCHS,
                       checkpoint_dir = check_points)

# get last weights and predict
weights <- list.files(check_points)
last = tail(weights, n=1)
last_path = file.path(check_points,last)

print("Predicting with trained model...")
# Depending on the number of pairs and epochs, the model quality varies.
# Prediction quality also depends on those parameters.
pred <- predict_unet(
  model_path   = last_path,
  landsat_path = pairs_proc$l8[[1]],
  out_path     = file.path(out, "prediction_l8_to_s2.tif")
)

# Plot result → saved to ms_out/prediction_result.png
pred_r <- terra::rast(file.path(out, "prediction_l8_to_s2.tif"))
result_plot <- file.path(out, "prediction_result.png")
png(result_plot, width = 1800, height = 500, res = 120)
par(mfrow = c(1, 3), mar = c(1, 1, 2.5, 1))
terra::plotRGB(l8_in,  r = 3, g = 2, b = 1, stretch = "lin", main = "L8 input")
terra::plotRGB(pred_r, r = 3, g = 2, b = 1, stretch = "lin", main = "Model output")
terra::plotRGB(s2_ref, r = 3, g = 2, b = 1, stretch = "lin", main = "S2 target")
dev.off()
print(paste("Result plot saved to:", result_plot))
