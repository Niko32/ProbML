from matplotlib import pyplot as plt
from numpy import ndarray, reshape
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch import no_grad, Tensor

def plot_results(X_test: ndarray, preds: ndarray, variances: ndarray, orig_X: ndarray, orig_labels: ndarray):
    lon, lat = X_test.T.reshape(2, 100, 100)
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_wireframe(lon, lat, reshape(preds, lat.shape), rstride=10, cstride=10, color="orange")
    ax.scatter3D(orig_X.T[0], orig_X.T[1], orig_labels)
    # ax.plot_surface(lon, lat, reshape(preds + variances, lat.shape), alpha=0.2, color="orange")
    # ax.plot_surface(lon, lat, reshape(preds - variances, lat.shape), alpha=0.2, color="orange")
    ax.set_zlim(orig_labels.min(), orig_labels.max())
    plt.show()

def plot_gp(observed_pred: MultivariateNormal, train_x: Tensor, train_y: Tensor, test_x: Tensor):
    """ Plot the GP results as in https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html """
    with no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.scatter(test_x.numpy(), observed_pred.mean.numpy())
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

    plt.show()