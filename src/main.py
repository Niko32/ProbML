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

from visualisation import plot_results
from preprocessing import prepare_data

matplotlib.use("TkAgg")

def read_tristans_file(filepath):
    """
    Reads a file with coordinates of a hexagonized leipzig.
    :param filepath: path to a file with content [(lat, long), ...., (lat, long)] in one line.
    :return: np array with unique centers of the hexagonized leipzig map.
    """
    with open(filepath, "r") as file:
        # awkward format makes eval the easiest way of opening. This is a security hazard.
        full_arr = np.array(eval(file.readline()), dtype=float)
    return np.unique(full_arr, axis=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read config
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)

    # Get data
    X_train, y_train, X_test, y_test, X_val, y_val, X_grid = prepare_data()

    # Init GPR
    kernel = RationalQuadratic(length_scale=0.6, length_scale_bounds="fixed", alpha=30, alpha_bounds="fixed")
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.8)

    # Load an existing model or train a new one
    if os.path.exists(CONFIG["model_file"]) and CONFIG["load_model"]:
        logging.info(f"Loading existing model under {CONFIG['model_file']}")
        gpr = joblib.load(CONFIG["model_file"])

    else:
        # Iteratively train the model with mini batches of the data as
        # the data does not fit into memory entirely
        # TODO: Is this a valid approach for GPR?
        N_BATCHES = 1
        X_batches = np.array_split(X_train, N_BATCHES)
        y_batches = np.array_split(y_train, N_BATCHES)
        for i in range(N_BATCHES):
            logging.info(f"Fitting GPR to batch {i + 1} out of {N_BATCHES}")
            gpr.fit(X_batches[i], y_batches[i])
        logging.info("GPR fitting completed. Saving...")
        # Save the model to prevent having to fit it every time
        joblib.dump(gpr, CONFIG["model_file"])

    # Run GPR
    logging.info(f"predicting at {X_grid.shape[0]} points...")
    predicted_labels, predicted_stds = gpr.predict(X_grid, return_std=True)
    logging.info(f"predicted_labels: {predicted_labels}")
    plot_results(X_grid, predicted_labels, predicted_stds, X_train, y_train)
