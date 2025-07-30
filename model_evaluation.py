# model_evaluation.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loading import load_solar_panel_data # To load raw data for evaluation
from data_preprocessing import perform_eda_and_preprocess # To get preprocessed data if needed

def evaluate_model(model, val_ds, class_names, plot_save_dir='eda_plots'):
    """
    Evaluates the trained classification model and displays/saves metrics.
    Saves confusion matrix plot to the specified directory.

    Args:
        model (tensorflow.keras.Model): The trained model.
        val_ds (tf.data.Dataset): The validation dataset.
        class_names (list): List of class names.
        plot_save_dir (str): Directory to save plots.
    """
    print("\n--- Evaluating Model Performance ---")

    # Basic Evaluation
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Generate predictions for confusion matrix and classification report
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure plot save directory exists
    os.makedirs(plot_save_dir, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, 'confusion_matrix.png')) # Save the plot
    plt.close() # Close the plot to free memory

    # Classification Report (Precision, Recall, F1-Score)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\n--- Model Evaluation Complete. ---")

if __name__ == "__main__":
    # Configuration
    DATA_DIR = r'C:\Users\Admin\OneDrive\Desktop\SolarGuard_Project\Faulty_solar_panel'
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 32
    MODEL_SAVE_PATH = 'solar_panel_classifier_model.h5'
    PLOT_SAVE_DIR = 'eda_plots'

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at '{MODEL_SAVE_PATH}'. Please run model_training.py first.")
    else:
        # Load only the validation data (raw, then preprocess it)
        # We call load_solar_panel_data directly here as we only need val_ds_raw
        _, val_ds_raw, class_names = load_solar_panel_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, validation_split=0.2)

        # Manually preprocess validation data for this specific script's execution
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        val_ds_processed = val_ds_raw.map(lambda x, y: (normalization_layer(x), y))
        AUTOTUNE = tf.data.AUTOTUNE
        val_ds_processed = val_ds_processed.cache().prefetch(buffer_size=AUTOTUNE)

        # Load the trained model
        loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)

        # Evaluate the model
        evaluate_model(loaded_model, val_ds_processed, class_names, plot_save_dir=PLOT_SAVE_DIR)