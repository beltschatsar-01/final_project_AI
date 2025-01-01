import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

def train_neural_network(X, y):
  
    y_encoded = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network Accuracy: {accuracy:.2f}")

    y_pred = model.predict(X_test)
    
    return {"model": model, "history": history, "y_pred": y_pred, "y_test": y_test, "accuracy": accuracy}


if __name__ == "__main__":
    dataset_path = "../data/processed_data.csv"  

    df = pd.read_csv(dataset_path)

    target_column = "species"  
    X = df.drop(columns=[target_column])
    y = df[target_column]

    results = train_neural_network(X, y)

    print(f"Final Neural Network Accuracy: {results['accuracy']:.2f}")
