import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_analysis(data, threshold=0.8):
    """
    Perform correlation analysis to identify and visualize highly correlated features.

    Args:
        data (pd.DataFrame): The dataset (features only, no target column).
        threshold (float): Correlation coefficient threshold for dropping features.

    Returns:
        pd.DataFrame: Dataset with highly correlated features removed.
        list: List of dropped features.
    """
    corr_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column].abs() > threshold)]

    print(f"Features to drop (correlation > {threshold}): {to_drop}")

    reduced_data = data.drop(columns=to_drop)
    return reduced_data, to_drop

def feature_importance_analysis(X, y, top_n=10):
    """
    Analyze feature importance using a Random Forest classifier.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
        top_n (int): Number of top important features to visualize.

    Returns:
        list: List of top important features.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(importances)[::-1]

    top_n = min(top_n, len(feature_names))

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(top_n), importances[sorted_indices][:top_n], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in sorted_indices[:top_n]], rotation=45)
    plt.tight_layout()
    plt.show()

    top_features = [feature_names[i] for i in sorted_indices[:top_n]]
    print(f"Top {top_n} important features: {top_features}")

    return top_features

def pca_dimensionality_reduction(X, n_components=0.95):
    """
    Perform Principal Component Analysis (PCA) for dimensionality reduction.

    Args:
        X (pd.DataFrame): Features dataset.
        n_components (float or int): Number of components to keep. If < 1, it represents the variance ratio.

    Returns:
        np.ndarray: Transformed dataset with reduced dimensions.
        PCA: The fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid()
    plt.show()

    print(f"Number of components selected: {pca.n_components_}")
    return X_pca, pca

if __name__ == "__main__":
    raw_data_path = "../data/processed_data.csv"
    data = pd.read_csv(raw_data_path)
    
    X = data.drop(columns=['Species'])  
    y = data['Species']
    
    reduced_data, dropped_features = correlation_analysis(X)
    
    top_features = feature_importance_analysis(reduced_data, y)
    
    X_pca, pca_model = pca_dimensionality_reduction(reduced_data)
