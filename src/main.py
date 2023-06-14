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



def plot_results(X_test, preds, variances, orig_X, orig_labels):
    lon, lat = X_test.T.reshape(2, 100, 100)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_wireframe(lon, lat, np.reshape(preds, lat.shape), rstride=10, cstride=10, color="orange")
    ax.scatter3D(orig_X.T[0], orig_X.T[1], orig_labels)
    ax.plot_surface(lon, lat, np.reshape(preds + variances, lat.shape), alpha=0.2, color="orange")
    ax.plot_surface(lon, lat, np.reshape(preds - variances, lat.shape), alpha=0.2, color="orange")
    ax.set_zlim(orig_labels.min(), orig_labels.max())
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read config
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)

    # Get data
    X, y, X_test = prepare_data()

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

    # Run GPR
    logging.info(f"predicting at {X_test.shape[0]} points...")
    predicted_labels, predicted_stds = gpr.predict(X_test, return_std=True)
    logging.info(f"predicted_labels: {predicted_labels}")
    plot_results(X_test, predicted_labels, predicted_stds, X, y)
