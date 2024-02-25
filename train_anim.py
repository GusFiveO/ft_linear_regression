import matplotlib
import matplotlib.animation as animation
import numpy as np


def plot_gradient_descent(
    fig: matplotlib.figure.Figure,
    feature_values: np.array,
    target_values: np.array,
    thetas_history: np.array,
):
    thetas = thetas_history[-1]
    theta1_values = np.linspace(thetas[0] - 5000, thetas[0] + 5000, 100)
    theta2_values = np.linspace(thetas[1] - 5000, thetas[1] + 5000, 100)
    Theta1, Theta2 = np.meshgrid(theta1_values, theta2_values)

    theta = np.vstack((Theta1.ravel(), Theta2.ravel()))
    mse_values = np.mean((target_values - np.dot(feature_values, theta)) ** 2, axis=0)
    mse_values = mse_values.reshape(Theta1.shape)
    mse_max = np.max(mse_values)
    mse_min = np.min(mse_values)
    mse_values_normalized = (mse_values - mse_min) / (mse_max - mse_min)

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )
    surface = ax1.plot_surface(Theta1, Theta2, mse_values_normalized, cmap="viridis")

    ax1.set_xlabel("theta1")
    ax1.set_ylabel("theta2")
    ax1.set_zlabel("MSE")
    ax1.set_title("MSE en fonction de theta1 et theta2")

    (scatt,) = ax1.plot([], [], [], c="black", marker="x", markersize=5)

    def init():
        return surface, scatt

    def animate(i):
        theta1, theta2 = thetas_history[i]
        mse_value = np.mean(
            (target_values - np.dot(feature_values, thetas_history[i])) ** 2
        )
        scatt.set_data(theta1, theta2)
        scatt.set_3d_properties((mse_value - mse_min) / (mse_max - mse_min))
        return surface, scatt

    frames = len(thetas_history)
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        interval=5000 / frames,
        blit=True,
        repeat=False,
    )
    return ani


def plot_predict_function(
    fig: matplotlib.figure.Figure,
    feature_values: np.array,
    target_values: np.array,
    thetas_history: np.array,
    km,
):
    ax2 = fig.add_subplot(212)
    scatt = ax2.scatter(
        km,
        target_values,
        label="Données d'entraînement",
    )
    (line,) = ax2.plot([], [], color="red", label="Fonction affine")

    ax2.set_xlabel("km")
    ax2.set_ylabel("price")

    def init():
        line.set_data([], [])
        return line, scatt

    def animate(i):
        thetas = thetas_history[i]
        predictions = np.dot(feature_values, thetas)
        line.set_data(km, predictions)
        return line, scatt

    frames = len(thetas_history)
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        interval=5000 / frames,
        blit=True,
        repeat=False,
    )
    return ani
