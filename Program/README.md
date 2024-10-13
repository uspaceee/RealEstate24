#Explanation of the program:
###Data generation: We create artificial data, where the area (independent variable) varies, and the price (dependent variable) is calculated according to the formula 
y=4+3X+noise, where noise is random deviation.
###Creating a DataFrame: We convert the data into a pandas DataFrame for ease of processing.
###Data splitting: We split the data into training and test sets (80% for training and 20% for testing).
###Creating and training the model: We initialize the linear regression model and train it on the training set.
###Forecasting: We use the trained model to forecast prices on the test set.
###Model evaluation: We determine the mean squared error (MSE) and the coefficient of determination (R²) to evaluate the quality of the model.
###Visualization: We create a graph to visualize real and forecasted prices, as well as a regression line.
