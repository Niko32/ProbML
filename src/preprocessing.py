import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


def filter_dataframe_by_bounding_box(df, top, left, bottom, right):
    filtered_df = df[(df['lat'] >= bottom) & (df['lat'] <= top) &
                     (df['lon'] >= left) & (df['lon'] <= right)]
    return filtered_df


def split_data(df: pd.DataFrame, train_fraction=0.7, eval_fraction=0.15, test_fraction=0.15):
    assert train_fraction + eval_fraction + test_fraction == 1.

    # Shuffle the df
    df = df.sample(frac=1)

    # Compute the number of rows in each set
    n_total = len(df)
    n_train = int(train_fraction * n_total)
    n_eval = int(eval_fraction * n_total)

    # Get the rows
    train_df = df.iloc[:n_train]
    eval_df = df.iloc[n_train:n_train + n_eval]
    test_df = df.iloc[n_train + n_eval:]

    return train_df, eval_df, test_df


def split_array(dataX, dataY, train_ratio=0.7, test_ratio=0.15, validation_ratio=0.15):
    # train is now 70% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

    # test is now 15% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))
    return x_train, y_train, x_test, y_test, x_val, y_val


def prepare_data(split=False):
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
    if len(df) > CONFIG["trim_size"]:
        logging.info(f"Sampling down to {CONFIG['trim_size']} training points for reduced complexity")
        df = df.sample(CONFIG["trim_size"])


    # Get inputs and outputs for GPR
    X = df[CONFIG["features"]].to_numpy()
    y = df[CONFIG["label"]].to_numpy()

    if CONFIG["normalize_data"]:
        X_mean, y_mean = np.mean(X, axis=0), np.mean(y)
        X_std, y_std = np.std(X, axis=0), np.std(y)
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std


    # Create grid to run GPR on
    lat = np.linspace(BOTTOM, TOP, 100)
    lon = np.linspace(LEFT, RIGHT, 100)
    # Normalize based on training data mean and variance
    if CONFIG["normalize_data"]:
        lon = (lon - X_mean[0]) / X_std[0]
        lat = (lat - X_mean[1]) / X_std[1]
    X_grid = np.array(np.meshgrid(lon, lat)).reshape((2, -1)).T
    X_train, y_train, X_test, y_test, X_val, y_val = split_array(X, y)
    return X_train, y_train, X_test, y_test, X_val, y_val, X_grid
