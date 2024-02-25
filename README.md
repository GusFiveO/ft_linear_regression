# Linear Regression Model Training and Prediction

This project provides a simple implementation of a linear regression model for predicting car prices based on their mileage. It consists of two main components: training the model and making predictions.

## Training

The `train.py` script trains the linear regression model using gradient descent optimization. It loads data from a CSV file containing car mileage and prices, normalizes the features, and iteratively updates the model parameters to minimize the cost function.

To train the model, run:

```bash
python3 train.py
```

This will save the trained model parameters (`thetas.json`) and generate visualizations to demonstrate the training process.

### Gradient Descent Animation
<!-- Add gif of gradient descent animation here -->

![Gradient Descent Animation](path/to/gradient_descent_animation.gif)

## Prediction

	The `predict.py` script allows users to input a mileage value and predicts the corresponding car price using the trained model parameters. It loads the trained parameters and the data, performs normalization, and plots the predicted price along with the actual prices.

	To make a prediction, run:

	```bash
	python3 predict.py
	```

### Prediction Plot
	<!-- Add image of prediction plot here -->

	![Prediction Plot](path/to/prediction_plot.png)

## Requirements

	- Python 3.x
	- NumPy
	- Matplotlib
	- pandas

## Dataset

	The dataset used for training and prediction is expected to be in CSV format with columns `km` for mileage and `price` for car prices.

## Notes

	- This project is for educational purposes and may not be suitable for production use without further enhancements and validations.
	- Adjustments to hyperparameters and model architecture may be necessary for better performance.
	- Additional features and more complex models could be explored for improving prediction accuracy.

	Feel free to explore and modify the code to suit your needs!
