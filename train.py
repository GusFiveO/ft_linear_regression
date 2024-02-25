#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from train_anim import plot_gradient_descent, plot_predict_function

from train_utils import (
    compute_mean_cost_derivative,
    load_csv,
    save_thetas,
    z_score_normalize,
)

from evaluate import plot_regression_evaluation


def train():
    """train linear regression model"""
    csv = load_csv("data.csv")
    if csv is None:
        print("No such file 'data.csv'")
        return
    km = np.array(csv["km"])
    price = np.array([csv["price"]]).reshape(-1, 1)

    feature_values = z_score_normalize(km)
    feature_values = np.column_stack((np.ones(len(feature_values)), feature_values))

    target_values = price

    theta1, theta2 = [0], [0]
    thetas = np.array([theta1, theta2])

    thetas_history = gradient_descent(
        thetas, target_values, feature_values, 2000, 0.005
    )

    theta1, theta2 = thetas_history[-1]

    save_thetas(theta1, theta2)

    fig = plt.figure()
    ani_descent = plot_gradient_descent(
        fig, feature_values, target_values, thetas_history
    )
    ani_predict = plot_predict_function(
        fig, feature_values, target_values, thetas_history, km
    )

    predictions = np.dot(feature_values, thetas_history[-1])

    plot_regression_evaluation(fig, predictions, target_values)

    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()


def gradient_descent(thetas, target_values, feature_values, epochs, learning_rate):
    thetas_history = np.array([thetas])

    for _ in range(epochs):
        mean_cost_derivative = compute_mean_cost_derivative(
            thetas, target_values, feature_values
        )
        thetas = thetas + learning_rate * mean_cost_derivative
        thetas_history = np.vstack([thetas_history, [thetas]])
    return thetas_history


if __name__ == "__main__":
    train()
