import yaml
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import joblib
import os
import logging
from sys import stdout


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Read data
COLUMNS = {
    "hw_device": str, 
    "manufactureur": str, 
    "hw_model": str, "type": str, 
    "lon": np.float32,
    "lat": np.float32,
    "rssnr": np.float32, 
    "rsrp": np.float32,
    "cqi": np.float32,
    "type": str,
    "geo_class": str,
    "band": str, 
    "datetime": object,
    "location_source": str,
    "gps_speed": np.float32,
    "qual": np.float32,
    "model": str, 
    "rsrq": np.float32,
    "rssi": np.float32
}
df = pd.read_csv(CONFIG["data_file"], dtype=COLUMNS)
df.drop(df.columns[0], axis=1, inplace=True)

# Remove entries, where features or labels are NaN
for feature in CONFIG["features"]:
    df = df[~df[feature].isnull()]
df = df[~df[CONFIG["label"]].isnull()]

# Trim the df for easier computation
df = df.sample(10_000)

# Get inputs and outputs for GPR
X = df[CONFIG["features"]].to_numpy()
y = df[CONFIG["label"]].to_numpy()

if CONFIG["normalize_data"]:
    X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
    X_std, y_std = np.std(X, axis=0), np.std(y)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_std

# Init GPR
kernel = RBF()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# Load an existing model or train a new one
if os.path.exists(CONFIG["model_file"]) and CONFIG["load_model"]:
    logging.info(f"Loading existing model under {CONFIG['model_file']}")
    gpr = joblib.load(CONFIG["model_file"])
else:
    # Iteratively train the model with mini batches of the data as
    # the data does not fit into memory entirely
    N_BATCHES = 1
    X_batches = np.array_split(X, N_BATCHES)
    y_batches = np.array_split(y, N_BATCHES)
    for i in range(N_BATCHES):
        logging.info(f"Fitting GPR to batch {i + 1} out of {N_BATCHES}")
        gpr.fit(X_batches[i], y_batches[i])

    # Save the model to prevent having to fit it every time
    joblib.dump(gpr, CONFIG["model_file"])

# Create grid to run GPR on
TOP, LEFT, BOTTOM, RIGHT = CONFIG["leipzig_bbox"].values()
lon = np.linspace(BOTTOM, TOP, 100)
lat = np.linspace(LEFT, RIGHT, 100)
if CONFIG["normalize_data"]:
    lon = (lon - np.mean(lon)) / np.std(lon)
    lat = (lat - np.mean(lat)) / np.std(lat)
grid = np.array(np.meshgrid(lon, lat)).reshape((2, -1)).T

# Run GPR
predicted_labels, predicted_stds = gpr.predict(grid, return_std=True)
logging.info("predicted_labels: ", predicted_labels)