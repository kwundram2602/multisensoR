library("remotes")
path_to_package <- "D:\\EAGLE\\multisensoR"
remotes::install_local(path_to_package)
library("multisensoR")
library("terra")

data_s2 = "D:\\EAGLE\\multisensoR\\python\\data\\subsamples\\S2"
data_l8 = "D:\\EAGLE\\multisensoR\\python\\data\\subsamples\\L8"
print("Finding pairs...")
pairs      <- find_l8_s2_pairs(data_l8, data_s2, n_pairs = 4)
print("Scaling check:")
#check_scaling(pairs,sample_frac =0.005)
pairs_proc <- preprocess_pairs(pairs,
  out_dir  = "D:\\EAGLE\\multisensoR\\python\\data\\subsamples\\preprocessed_pairs",
  scale_l8 = c(gain = 0.0000275, offset = -0.2),  # L8 C2 L2 Formel
  scale_s2 = c(gain = 1/10000,   offset = 0))      # S2 / 10000

#check_scaling(pairs_proc,sample_frac =0.005)   # L8 → 10m S2-Grid

ds <- landsat_sentinel_dataset(
  landsat_files  = pairs_proc$l8,
  sentinel_files = pairs_proc$s2,
  patch_size = 256, augment = TRUE
)

train_dl <- torch::dataloader(ds, batch_size = 1, shuffle = TRUE)

# 3. Modell erstellen (out_channels = Anzahl S2-Bänder in deinen TIFs!)
model    <- unet_model(in_channels = 6L, out_channels = 6L)

# 4. Trainieren
checkppoint_dir <- "D:\\EAGLE\\multisensoR\\python\\data\\checkpoints"
history  <- train_unet(model, train_dl, epochs = 5,
                       checkpoint_dir = checkppoint_dir)

# 5. Vorhersage
pred <- predict_unet(
  model_path   = file.path(checkppoint_dir, "unet_epoch_030.pt"),
  landsat_path = pairs_proc$l8[[1]],
  out_path     = file.path(checkppoint_dir, "prediction_l8_to_s2.tif")
)