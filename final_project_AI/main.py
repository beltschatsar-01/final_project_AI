# Import required libraries
import os
import pandas as pd
from models.traditional_ml import train_traditional_ml
from models.neural_networks import train_neural_network
from utils.data_preprocessing import preprocess_data
from utils.feature_selection import correlation_analysis, feature_importance_analysis, pca_dimensionality_reduction
from utils.evaluation import evaluate_model, evaluate_neural_network

# Paths to input/output files
RAW_DATA_PATH = "./data/raw_data.csv"
PROCESSED_DATA_PATH = "./data/processed_data.csv"

# Main function to orchestrate the workflow
def main():
    # Step 1: Data Preprocessing
    print("Starting data preprocessing...")
    processed_data = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    if processed_data is None:
        print("Error in preprocessing. Exiting...")
        return

    # Step 2: Split data into features (X) and target (y)
    print("Splitting data into features and target...")
    target_column = 'Outcome'  # Change 'Outcome' to your dataset's target column
    if target_column not in processed_data.columns:
        print(f"Target column '{target_column}' not found in the dataset!")
        return

    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]

    # Step 3: Feature Selection/Dimensionality Reduction
    print("Performing feature selection...")
    X_reduced = correlation_analysis(X)  # Drop correlated features
    top_features = feature_importance_analysis(X_reduced, y)  # Analyze top important features
    X_pca, pca_model = pca_dimensionality_reduction(X_reduced)  # Perform PCA

    # Step 4: Train and Evaluate Traditional ML Model
    print("Training and evaluating traditional ML model...")
    ml_results = train_and_evaluate_ml_model(X_pca, y)

    # Step 5: Train and Evaluate Neural Network Model
    print("Training and evaluating neural network model...")
    nn_results = train_and_evaluate_nn(X_pca, y)

    # Step 6: Compare Results
    print("\n--- Model Comparison ---")
    print(f"Traditional ML Accuracy: {ml_results['accuracy']:.2f}")
    print(f"Neural Network Accuracy: {nn_results['accuracy']:.2f}")

    # Visualize evaluation for neural network (if history available)
    if 'history' in nn_results:
        evaluate_neural_network(nn_results['history'], y, nn_results['y_pred'], model_type="Neural Network")

if __name__ == "__main__":
    # Ensure the required directories exist
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./utils"):
        os.makedirs("./utils")
    
    # Run the main workflow
    print("Starting the AI workflow...")
    main()
