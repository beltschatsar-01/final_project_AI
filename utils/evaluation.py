from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, model_type="Traditional ML"):
    """
    Evaluate the performance of a classification model.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_type (str): Type of model being evaluated. Default is "Traditional ML".

    Returns:
        dict: A dictionary containing evaluation metrics and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Model Type: {model_type}")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(cm, model_type=model_type)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, model_type="Model"):
    """
    Plot the confusion matrix.

    Args:
        cm (array-like): Confusion matrix.
        model_type (str): Type of model for title.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.title(f"Confusion Matrix: {model_type}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.savefig(f'{model_type}_confusion_matrix.png')  
    plt.show()

def evaluate_neural_network(history, y_true, y_pred, model_type="Neural Network"):
    """
    Evaluate a neural network model and visualize training history.

    Args:
        history: Training history object returned by the model's fit() method.
        y_true (array-like): True labels (class indices or one-hot encoded).
        y_pred (array-like): Predicted probabilities or class indices.
        model_type (str): Type of model being evaluated.

    Returns:
        dict: A dictionary containing evaluation metrics and confusion matrix.
    """
    import numpy as np
    
    if len(y_true.shape) > 1:
        y_true_indices = np.argmax(y_true, axis=1)  
    else:
        y_true_indices = y_true  

    if len(y_pred.shape) > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred

    metrics = evaluate_model(y_true_indices, y_pred_indices, model_type=model_type)

    plot_training_history(history, model_type=model_type)

    return metrics


def plot_training_history(history, model_type="Neural Network"):
    """
    Plot training and validation accuracy/loss from a neural network's history.

    Args:
        history: Training history object returned by the model's fit() method.
        model_type (str): Type of model for title.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{model_type} Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{model_type} Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
