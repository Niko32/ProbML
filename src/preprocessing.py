import logging
import yaml
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

def filter_dataframe_by_bounding_box(df, top, left, bottom, right):
    filtered_df = df[(df['lat'] >= bottom) & (df['lat'] <= top) &
                     (df['lon'] >= left) & (df['lon'] <= right)]
    return filtered_df

def prepare_data():
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
        df = df.sample(100)

    # Get inputs and outputs for GPR
    X = df[CONFIG["features"]].to_numpy()
    y = df[CONFIG["label"]].to_numpy()

    if CONFIG["normalize_data"]:
        X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
        X_std, y_std = np.std(X, axis=0), np.std(y)
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std

    # Create grid to run GPR on
    TOP, LEFT, BOTTOM, RIGHT = CONFIG["leipzig_bbox"].values()
    lat = np.linspace(BOTTOM, TOP, 100)
    lon = np.linspace(LEFT, RIGHT, 100)
    # Normalize based on training data mean and variance
    if CONFIG["normalize_data"]:
        lon = (lon - X_mean[0]) / X_std[0]
        lat = (lat - X_mean[1]) / X_std[1]
    X_test = np.array(np.meshgrid(lon, lat)).reshape((2, -1)).T

    return X, y, X_test