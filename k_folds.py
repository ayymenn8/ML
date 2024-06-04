import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

iris = load_iris()
wine = load_wine()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['class'] = iris.target
wine_data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_data['class'] = wine.target
X_iris = iris_data.drop("class", axis=1)
y_iris = iris_data["class"]
X_wine = wine_data.drop("class", axis=1)
y_wine = wine_data["class"]
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVC": SVC(random_state=42)
}
def perform_k_fold_cv(X, y, classifiers, k_values):
    results = {}
    for name, clf in classifiers.items():
        for k in k_values:
            kfold = KFold(n_splits=k, random_state=42, shuffle=True)
            cv_results = cross_val_score(clf, X, y, cv=kfold)
            results[(name, k)] = np.mean(cv_results)
    return results
k_values = [5,7,10]
iris_results = perform_k_fold_cv(X_iris, y_iris, classifiers, k_values)
print("Iris Dataset Results:")
for (name, k), score in iris_results.items():
    print(f"{name} with k={k}: {score:.4f}")
wine_results = perform_k_fold_cv(X_wine, y_wine, classifiers, k_values)
print("\nWine Dataset Results:")
for (name, k), score in wine_results.items():
    print(f"{name} with k={k}: {score:.4f}")