import numpy as np
from sklearn .datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNN

iris = load_iris()
x = iris.data # x contains features
y = iris.target # y contains labels

# Split into training and testing sets
# 80% of the data is used for training, 20% for testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
knn = KNN(k=3)
#train the model (store the training data)
knn.fit(x_train, y_train)
#Predict labels for the test set
y_pred = knn.predict(x_test)
#evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")