import numpy as np


def calculate_r_square(actual, predicted):
    """
    Calculates the R-square metric given the actual and predicted values.

    R-square is a statistical measure that represents the proportion of the variance in the dependent variable (actual)
    that is predictable from the independent variable (predicted).

    Args:
        actual (array-like): Array or list of actual values.
        predicted (array-like): Array or list of predicted values.

    Returns:
        float: The R-square metric, ranging from 0 to 1. A value of 1 indicates a perfect fit, while lower values
        indicate poorer fits.

    Notes:
        - The input arrays/lists are converted to NumPy arrays for calculation.
        - The R-square value is calculated as 1 - (residual sum of squares / total sum of squares).
        - The total sum of squares (TSS) represents the variability of the actual values around their mean.
        - The residual sum of squares (RSS) represents the variability of the predicted values around the actual values.
    """
    # Convert actual and predicted to NumPy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate the mean of the actual values
    mean_actual = np.mean(actual)

    # Calculate the total sum of squares (TSS)
    tss = np.sum((actual - mean_actual) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = np.sum((actual - predicted) ** 2)

    # Calculate the R-square value
    r_square = 1 - (rss / tss)

    return r_square


def calculate_mean_likelihood(predictor_means, predictor_stddevs, actual_values):
    """
    Calculates the mean likelihood of observing the actual values given a predictor that assumes a Gaussian distribution
    at each point and returns separate arrays of means and standard deviations.

    Args:
        predictor_means (ndarray): NumPy array of mean values returned by the predictor.
        predictor_stddevs (ndarray): NumPy array of standard deviation values returned by the predictor.
        actual_values (ndarray): NumPy array of actual measurement values at each point.

    Returns:
        float: The likelihood of observing the actual values.

    Raises:
        ValueError: If the lengths of predictor_means, predictor_stddevs, and actual_values do not match.

    """
    if len(predictor_means) != len(predictor_stddevs) or len(predictor_means) != len(actual_values):
        raise ValueError("Lengths of predictor_means, predictor_stddevs, and actual_values must be the same.")

    likelihood = np.mean(1.0 / (np.sqrt(2 * np.pi) * predictor_stddevs) * np.exp(
        -0.5 * ((actual_values - predictor_means) / predictor_stddevs) ** 2))
    return likelihood


def calculate_mse(actual, predicted):
    """
    Calculates the mean squared error (MSE) between the actual and predicted values.

    Mean squared error is a common metric used to measure the average squared difference between the predicted and
    actual values. It provides a measure of how well the predicted values align with the actual values.

    Args:
        actual (array-like): Array or list of actual values.
        predicted (array-like): Array or list of predicted values.

    Returns:
        float: The mean squared error (MSE) between the actual and predicted values.

    Notes:
        - The input arrays/lists are converted to NumPy arrays for calculation.
        - The MSE is calculated as the average of the squared differences between the actual and predicted values.
        - It provides a non-negative value, with a lower MSE indicating a better fit between the two sets of values.
    """
    # Convert actual and predicted to NumPy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate the squared differences between actual and predicted
    squared_diff = (actual - predicted) ** 2

    # Calculate the mean squared error (MSE)
    mse = np.mean(squared_diff)

    return mse
