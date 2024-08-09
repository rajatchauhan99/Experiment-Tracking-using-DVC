import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import Live


# Load the Iris dataset
data = pd.read_csv("data/iris_data.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model with some parameters
n_estimators = 100
max_depth = 3
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# Log model parameters and evaluation metrics

with Live(save_dvc_exp = True) as Live:
    Live.log_param("n_estimators", n_estimators)
    Live.log_param("max_depth", max_depth)
    Live.log_param("random_state", 42)
    
    Live.log_metric("accuracy", accuracy)
    Live.log_metric("precision", precision)
    Live.log_metric("recall", recall)
    Live.log_metric("f1_score", f1)






