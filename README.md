# AI Model for Classification: Traditional ML vs Neural Networks

## Project Description

This project compares the performance of traditional machine learning (ML) models and neural networks for solving a classification problem. The workflow includes data preprocessing, feature selection, dimensionality reduction, model training, and evaluation. The models are trained on a processed dataset, and the results are compared to assess accuracy and other metrics.

### Features:
- Data preprocessing (handling missing values, feature scaling, and encoding categorical variables).
- Feature selection (correlation analysis, feature importance, and PCA).
- Model implementation using both traditional ML and neural networks.
- Performance evaluation with confusion matrix, accuracy, and classification report.
- Visualization of training history (for neural networks), confusion matrix, and feature importance.

## Installation Instructions

Follow these steps to set up the project on your local machine:

### 1. Clone the repository

```bash
git clone https://github.com/beltschatsar-01/final_project_AI.git
cd final_project_AI.git


pip list 

Package                      Version
---------------------------- -----------
TensorFlow                   2.18
NumPy                        2.0.2
Pandas                       2.2.3
Scikit-learn                 1.6.0
Matplotlib                   3.10.0
seaborn                      0.13.2

You can easily fint the data on main.py

data = """
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
...
...
...


"""