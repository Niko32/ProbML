import yaml
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic
import numpy as np
import joblib
import os
import logging
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats.distributions import norm
matplotlib.use("TkAgg")


def plot_results(lat, lon, preds, variances, orig_X, orig_labels):
    lon, lat = np.meshgrid(lon, lat)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_wireframe(lon, lat, np.reshape(preds, lat.shape), rstride=10, cstride=10, color="orange")
    ax.scatter3D(orig_X.T[0], orig_X.T[1], orig_labels)
    ax.plot_surface(lon, lat, np.reshape(preds + variances, lat.shape), alpha=0.2, color="orange")
    ax.plot_surface(lon, lat, np.reshape(preds - variances, lat.shape), alpha=0.2, color="orange")
    ax.set_zlim(orig_labels.min(), orig_labels.max())
    plt.show()


def filter_dataframe_by_bounding_box(df, top, left, bottom, right):
    filtered_df = df[(df['lat'] >= bottom) & (df['lat'] <= top) &
                     (df['lon'] >= left) & (df['lon'] <= right)]
    return filtered_df


if __name__ == "__main__":
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
        df = df[~df[feature].isna()]
    df = df[~df[CONFIG["label"]].isna()]

    # filter out points not within Leipzig

    TOP, LEFT, BOTTOM, RIGHT = CONFIG["leipzig_bbox"].values()
    df = filter_dataframe_by_bounding_box(df, TOP, LEFT, BOTTOM, RIGHT)

    # Trim the df for easier computation
    if len(df) > 10_000:
        logging.info("Sampling down to 10k training points for reduced complexity")
        df = df.sample(4000)

    # Get inputs and outputs for GPR
    X = df[CONFIG["features"]].to_numpy()
    y = df[CONFIG["label"]].to_numpy()

    if CONFIG["normalize_data"]:
        X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
        X_std, y_std = np.std(X, axis=0), np.std(y)
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std

    # Init GPR
    kernel = RBF(length_scale=1, length_scale_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel)

    # Load an existing model or train a new one
    if os.path.exists(CONFIG["model_file"]) and CONFIG["load_model"]:
        logging.info(f"Loading existing model under {CONFIG['model_file']}")
        gpr = joblib.load(CONFIG["model_file"])

    else:
        # Iteratively train the model with mini batches of the data as
        # the data does not fit into memory entirely
        # TODO: Is this a valid approach for GPR?
        N_BATCHES = 1
        X_batches = np.array_split(X, N_BATCHES)
        y_batches = np.array_split(y, N_BATCHES)
        for i in range(N_BATCHES):
            logging.info(f"Fitting GPR to batch {i + 1} out of {N_BATCHES}")
            gpr.fit(X_batches[i], y_batches[i])
        logging.info("GPR fitting completed. Saving...")
        # Save the model to prevent having to fit it every time
        joblib.dump(gpr, CONFIG["model_file"])

    # Create grid to run GPR on
    TOP, LEFT, BOTTOM, RIGHT = CONFIG["leipzig_bbox"].values()
    lat = np.linspace(BOTTOM, TOP, 100)
    lon = np.linspace(LEFT, RIGHT, 100)
    # Normalize based on training data mean and variance
    if CONFIG["normalize_data"]:
        lon = (lon - X_mean[0]) / X_std[0]
        lat = (lat - X_mean[1]) / X_std[1]
    grid = np.array(np.meshgrid(lon, lat)).reshape((2, -1)).T

    # Run GPR
    logging.info(f"predicting at {grid.shape[0]} points...")
    predicted_labels, predicted_stds = gpr.predict(grid, return_std=True)
    logging.info(f"predicted_labels: {predicted_labels}")
    plot_results(lat, lon, predicted_labels, predicted_stds, X, y)
