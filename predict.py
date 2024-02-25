#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from train_utils import z_score_normalize, z_score_unnormalize


def load_json(pathname: str):
    try:
        f = open(pathname)
        data = json.load(f)
        f.close()
        return data
    except Exception:
        return None


def load_csv(pathname: str):
    """load a csv file"""
    try:
        return pd.read_csv(pathname)
    except Exception:
        return None


def plot_prediction(features, targets, thetas, to_predict, km):
    theta1, theta2 = thetas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    predictions = np.dot(features, thetas)
    ax.plot(km, predictions, color="red")

    ax.scatter(km, targets)

    prediction = theta1[0] + to_predict * theta2[0]

    to_predict_unormalized = (to_predict * np.std(km)) + np.mean(km)

    ax.plot(to_predict_unormalized, prediction, c="black", marker="x")
    ax.annotate(
        f"{prediction:.2f}",
        xy=(to_predict_unormalized, prediction),
        xytext=(to_predict_unormalized + 0.1, prediction),
    )

    plt.show()


def predict():
    thetas_json = load_json("thetas.json")
    if thetas_json is None:
        print("No such file 'thetas.json'")
        return
    theta1, theta2 = thetas_json.values()
    thetas = np.array((theta1, theta2))
    csv = load_csv("data.csv")
    if csv is None:
        print("No such file 'data.csv'")
        return
    km = np.array(csv["km"])
    normalized_km = z_score_normalize(km)
    feature_values = np.column_stack((np.ones(len(normalized_km)), normalized_km))

    price = np.array([csv["price"]]).reshape(-1, 1)

    try:
        to_predict = int(input("Choose a mileage: "))
        normalized_input = (to_predict - np.mean(km)) / np.std(km)
        plot_prediction(feature_values, price, thetas, normalized_input, km)
    except Exception:
        print("Use a valid mileage")
    return


if __name__ == "__main__":
    predict()
