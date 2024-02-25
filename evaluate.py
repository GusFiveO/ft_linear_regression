import numpy as np


def r2_score(predictions: np.array, target_values: np.array):
    mean_target = np.mean(target_values)
    rss = np.sum((target_values - predictions) ** 2)
    tss = np.sum((target_values - mean_target) ** 2)
    if tss == 0:
        return 0.0
    return 1 - rss / tss


def mean_absolute_error(predictions: np.array, target_values: np.array):
    sum_diff = np.sum(np.abs(target_values - predictions))
    return sum_diff / len(target_values)


def root_mean_square_error(predictions: np.array, target_values: np.array):
    n = len(target_values)
    square_error = (
        np.dot((target_values - predictions).T, target_values - predictions) / n
    )
    return np.sqrt(square_error)


def plot_regression_evaluation(fig, predictions: np.array, target_values: np.array):
    model_r2_score = r2_score(predictions, target_values) * 100
    model_mean_absolute_error = mean_absolute_error(predictions, target_values)
    model_root_mean_square_error = root_mean_square_error(predictions, target_values)[
        0
    ][0]

    ax = fig.add_subplot(222, xticks=[], yticks=[])
    ax.text(2, 8, f"R2 SCORE: {model_r2_score:.2f}", fontsize=10)
    ax.text(2, 5, f"MAE: {model_mean_absolute_error:.2f}", fontsize=10)
    ax.text(2, 2, f"RMSE: {model_root_mean_square_error:.2f}", fontsize=10)
    ax.set(xlim=(0, 10), ylim=(0, 10))
