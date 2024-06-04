import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('most_watched_600_youtube_videos_2024.csv')
x = dataset['view_count'].values.reshape(-1, 1)
y = dataset['like_count'].values
missing_values = dataset.isnull().sum()
column_data = ['view_count','like_count']
label_encoder = LabelEncoder()
dataset[column_data] = label_encoder.fit_transform(dataset['column_data'])
X_train, X_test,Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model_linear = LinearRegression()
model_linear.fit(X_train,Y_train)
y_pred = model_linear.predict(X_test)
print(y_pred)