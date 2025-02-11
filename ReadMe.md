# K-Nearest Neighbors (KNN) Classifier using Python

## 📌 Introduction
This project implements the **K-Nearest Neighbors (KNN) algorithm** from scratch in Python using the **Iris dataset**.  

It includes:
- A **custom KNN classifier** built from scratch.
- A **training script** to evaluate the model.
- A **Jupyter Notebook** for data visualization.

---

## 📊 Dataset: Iris
The Iris dataset consists of:

- 150 samples
- 3 classes (Setosa, Versicolor, Virginica)
- 4 features (sepal length, sepal width, petal length, petal width)
It is available by default in sklearn.datasets.

## 🚀 Example Output
The project provides:

- Model accuracy using the test set.
- Visualizations of decision boundaries.
- Predictions on new sample data.

## 🖥️ Technologies Used
- Python 🐍
- NumPy – Array operations
- Pandas – Data handling
- Scikit-learn – Machine learning tools
- Matplotlib & Seaborn – Data visualization
- Jupyter Notebook – Interactive coding

## How the KNN Algorithm Works
1. Load the dataset and preprocess data.
2. Calculate distances between test and training points.
3. Find K nearest neighbors based on distance.
4. Determine the majority class among neighbors.
5. Predict the class of the test sample.
