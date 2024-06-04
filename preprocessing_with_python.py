import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('cereal.csv')  
df_encoded = pd.get_dummies(df, columns=['name','mfr','type','calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups','rating'])
print(df_encoded)
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].astype('category').cat.codes
print(df)
mean_values = df.mean()
median_values = df.median()
mode_values = df.mode().iloc[0]  
df_mean = df.fillna(mean_values)
df_median = df.fillna(median_values)
df_mode = df.fillna(mode_values)
print("Imputed with mean:")
print(df_mean)
print("\nImputed with median:")
print(df_median)
print("\nImputed with mode:")
print(df_mode)