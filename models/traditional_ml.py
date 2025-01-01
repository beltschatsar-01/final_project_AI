import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_traditional_ml(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    return {"model": model, "accuracy": accuracy, "classification_report": report}

if __name__ == "__main__":
    
    dataset_path = "../data/processed_data.csv" 
    
    df = pd.read_csv(dataset_path)
    
    target_column = 'species'  
    X = df.drop(columns=[target_column])
    y = df[target_column]

    results = train_traditional_ml(X, y)
    print(f"Trained Model: {results['model']}")
    print(f"Accuracy: {results['accuracy']:.2f}")
