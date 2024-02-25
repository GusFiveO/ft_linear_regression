import numpy as np
import json
import pandas as pd


def load_csv(pathname: str):
    """load a csv file"""
    try:
        return pd.read_csv(pathname)
    except Exception:
        return None


def reverse_normalize_coefficients(
    normalized_coefficients, feature_values, target_values
):
    min_feature = np.min(feature_values)
    max_feature = np.max(feature_values)
    min_target = np.min(target_values)
    max_target = np.max(target_values)

    unnormalized_theta1 = (
        normalized_coefficients[0] * (max_target - min_target) + min_target
    )

    unnormalized_theta2 = (
        normalized_coefficients[1]
        * (max_target - min_target)
        / (max_feature - min_feature)
    )

    return np.array([unnormalized_theta1, unnormalized_theta2])


def z_score_unnormalize(normalized_array, std, mean):
    unnormalized_array = (normalized_array * std) + mean
    return unnormalized_array


def z_score_normalize(array):
    normalized_array = (array - np.mean(array)) / np.std(array)
    return normalized_array


def compute_mean_cost_derivative(
    thetas: np.array, target_values: np.array, feature_values: np.array
):
    """compute the derivative of the mean squared error"""
    n = len(target_values)
    error = np.dot(feature_values, thetas) - target_values
    gradient = (-1 / n) * np.dot(feature_values.T, error)
    return gradient


def save_thetas(theta1: np.array, theta2: np.array):
    thetas = {"theta1": theta1.tolist(), "theta2": theta2.tolist()}

    with open("thetas.json", "w") as jsonfile:
        json.dump(thetas, jsonfile)
