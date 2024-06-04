import pandas as pd 
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset=pd.read_csv('Groceries_dataset.csv')
num = dataset.select_dtypes(include=['float', 'int']).columns
dataset[num] = dataset[num].fillna(dataset[num].mean(axis=0))
str =dataset.select_dtypes(include=[object]).columns
dataset[str] = dataset[str].fillna(dataset[str].mode().iloc[0])
input_num_feature = ['Member_number','price']
normal = StandardScaler()
dataset[input_num_feature] = normal.fit_transform(dataset[input_num_feature])
encode = ['itemDescription']
dataset=pd.get_dummies(dataset, columns=encode)
dataset.drop_duplicates(inplace=True)
dataset.to_csv('modifiedgrocerystoredataset.csv',index=False)
First=dataset.iloc[:,:-1]
Second=dataset.iloc[:,-1:]
First_train, first_test, second_train, second_test = train_test_split(First,Second , test_size=0.1, random_state=42)
model = LinearRegression()
model.fit(First_train, second_train)
predictions = model.predict(first_test)
mae = mean_absolute_error(second_test, predictions)
print("Mean Absolute Error (MAE):", mae)
mse = mean_squared_error(second_test, predictions)
print("Mean Squared Error (MSE):", mse)
r_squared = r2_score(second_test, predictions)
print("R-Squared:", r_squared)
n = len(second_test)
p = First_train.shape[1] 
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print("Adjusted R-Squared:", adjusted_r_squared)
