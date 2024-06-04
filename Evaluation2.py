import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
dataset = pd.read_csv('cereal.csv')

# Preprocess the data
# Handle missing values
dataset.dropna(inplace=True)

# Convert categorical variables into numerical format using one-hot encoding
dataset_encoded = pd.get_dummies(dataset, columns=['mfr', 'type'])

# Split the dataset into features (x) and target variable (y)
x = dataset_encoded.drop(columns=['name', 'rating'])
y = dataset_encoded['rating']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the regression model pipeline with variations
# You can include different regression models and variations here
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # Apply feature scaling
    ('regressor', LinearRegression())  # Regression model
])

# Train the regression model
pipeline.fit(x_train, y_train)

# Make predictions on the testing set
predictions = pipeline.predict(x_test)

# Evaluate the model
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
