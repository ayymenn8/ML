from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def train_decision_tree(max_depth):
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier
def train_random_forest(n_estimators, max_depth, random_state):
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier
dt_classifier = train_decision_tree(max_depth=5)
rf_classifier = train_random_forest(n_estimators=100, max_depth=5, random_state=42)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
classification_report_dt = classification_report(y_test, y_pred_dt)
print("Decision Tree Classifier Evaluation:")
print("Accuracy:", accuracy_dt)
print("Classification Report:\n", classification_report_dt)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)
print("\nRandom Forest Classifier Evaluation:")
print("Accuracy:", accuracy_rf)
print("Classification Report:\n", classification_report_rf)

from sklearn.model_selection import GridSearchCV
param_grid_dt = {'max_depth': [3, 5, 7, 9, None]}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)
print("Best parameters for Decision Tree Classifier:", grid_search_dt.best_params_)
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 9, None], 'random_state': [42]}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
print("Best parameters for Random Forest Classifier:", grid_search_rf.best_params_)