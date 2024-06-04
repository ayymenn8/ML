import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
dataset = pd.read_csv('iris.csv')

# Impute missing values with the mean for numeric columns
numeric_imputer = SimpleImputer(strategy='mean')
dataset[['Length', 'Width', 'Petal.Length', 'Petal.Width']] = numeric_imputer.fit_transform(dataset[['Length', 'Width', 'Petal.Length', 'Petal.Width']])

# Drop non-numeric columns
dataset_numeric = dataset.drop(columns=['Species'])

# Split the dataset into features (x) and target variable (y)
x = dataset_numeric.drop(columns=['Petal.Width'])  # Features
y = dataset_numeric['Petal.Width']  # Target variable

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the testing set
predictions = model.predict(x_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate R-Squared
r_squared = r2_score(y_test, predictions)
print("R-Squared:", r_squared)

# Calculate Adjusted R-Squared (Optional)
n = len(y_test)
p = x_train.shape[1]  # Number of features
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print("Adjusted R-Squared:", adjusted_r_squared)
