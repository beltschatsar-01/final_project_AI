import os
import pandas as pd
from io import StringIO
from models.traditional_ml import train_traditional_ml
from models.neural_networks import train_neural_network
from utils.data_preprocessing import preprocess_data
from utils.feature_selection import correlation_analysis, feature_importance_analysis, pca_dimensionality_reduction
from utils.evaluation import evaluate_model, evaluate_neural_network
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = "./data/raw_data.csv"
PROCESSED_DATA_PATH = "./data/processed_data.csv"

data = """
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
6.5,3.0,5.5,1.8,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica
"""  


def main():
    print("Loading data...")
    df = pd.read_csv(StringIO(data), header=None)

    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    print("Encoding target variable...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Features (X):")
    print(X.head())
    print("\nTarget (y):")
    print(y.head())
    print("\nEncoded Target (y_encoded):")
    print(y_encoded[:5])

    print("Starting data preprocessing...")
    processed_data = preprocess_data(df, PROCESSED_DATA_PATH)
    if processed_data is None:
        print("Error in preprocessing. Exiting...")
        return

    print("Performing feature selection...")
    X_reduced, dropped_features = correlation_analysis(X) 
    print(f"Shape of X_reduced after correlation analysis: {X_reduced.shape}")
    print("Dropped features due to high correlation:", dropped_features)
    print("First few rows of X_reduced:")
    print(X_reduced.head())


    try:
        top_features = feature_importance_analysis(X_reduced, y_encoded)  
        print(f"Top features: {top_features}")
    except Exception as e:
        print(f"Error in feature importance analysis: {e}")
        return

    X_pca, pca_model = pca_dimensionality_reduction(X_reduced)  

    """# Step 4: Train and Evaluate Traditional ML Model
    print("Training and evaluating traditional ML model...")
    ml_results = train_traditional_ml(X_pca, y_encoded)

    # Step 5: Train and Evaluate Neural Network Model
    print("Training and evaluating neural network model...")
    nn_results = train_neural_network(X_pca, y_encoded)

    # Step 6: Compare Results
    print("\n--- Model Comparison ---")
    print(f"Traditional ML Accuracy: {ml_results['accuracy']:.2f}")
    print(f"Neural Network Accuracy: {nn_results['accuracy']:.2f}")"""
    print("Training and evaluating traditional ML model...")
    ml_results = train_traditional_ml(X_pca, y_encoded)

    print("Training and evaluating neural network model...")
    nn_results = train_neural_network(X_pca, y_encoded)

    evaluate_neural_network(nn_results['history'], nn_results['y_test'], nn_results['y_pred'], model_type="Neural Network")


    print("\n--- Model Comparison ---")
    print(f"Traditional ML Accuracy: {ml_results['accuracy']:.2f}")
    print(f"Neural Network Accuracy: {nn_results['accuracy']:.2f}")


    if 'history' in nn_results:
        evaluate_neural_network(nn_results['history'], nn_results['y_test'], nn_results['y_pred'], model_type="Neural Network")


if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./utils"):
        os.makedirs("./utils")
    
    print("Starting the AI workflow...")
    main()
