import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

dataset = pd.read_csv('ecom_data.csv')
label_encoder = LabelEncoder()
dataset['product'] = label_encoder.fit_transform(dataset['product'])
dataset['customer_country'] = label_encoder.fit_transform(dataset['customer_country'])
X = dataset[['product', 'price', 'quantity', 'customer_country']]
y = dataset['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Kernel Function: Linear")
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
precision_score(y_test, y_pred, average='weighted', zero_division=1)
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F-measure:", f1_score(y_test, y_pred, average='weighted'))