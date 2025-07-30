# model_training.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
from data_loading import load_solar_panel_data
from data_preprocessing import perform_eda_and_preprocess # Now imports the combined function

def build_and_train_classifier(train_ds, val_ds, class_names, img_height, img_width, epochs=15, model_save_path='solar_panel_classifier_model.h5', plot_save_dir='eda_plots'):
    """
    Builds, compiles, and trains a MobileNetV2-based classification model.
    Saves training history plots to the specified directory.

    Args:
        train_ds (tf.data.Dataset): Preprocessed training dataset.
        val_ds (tf.data.Dataset): Preprocessed validation dataset.
        class_names (list): List of class names for the output layer.
        img_height (int): Input image height for the model.
        img_width (int): Input image width for the model.
        epochs (int): Number of training epochs.
        model_save_path (str): Path to save the trained model.
        plot_save_dir (str): Directory to save plots.
    Returns:
        tensorflow.keras.Model: The trained classification model.
    """
    num_classes = len(class_names)
    print(f"\n--- Building MobileNetV2 Transfer Learning Model for {num_classes} classes ---")

    # Load pre-trained MobileNetV2, excluding its top (classification) layer
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                             include_top=False,
                             weights='imagenet')

    # Freeze the convolutional base
    base_model.trainable = False

    # Create a new model on top of the pre-trained base
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax') # Output layer for your classes
    ])

    # Compile the Model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Train the Model
    print(f"\n--- Starting Model Training for {epochs} epochs ---")
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=val_ds)

    print("\n--- Model Training Complete. ---")

    # Ensure plot save directory exists
    os.makedirs(plot_save_dir, exist_ok=True)

    # Plot Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, 'training_history.png')) # Save the plot
    plt.close() # Close the plot to free memory

    # Save the Trained Model
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    return model

if __name__ == "__main__":
    # Configuration
    DATA_DIR = r'C:\Users\Admin\OneDrive\Desktop\SolarGuard_Project\Faulty_solar_panel'
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 32
    EPOCHS = 15
    MODEL_SAVE_PATH = 'solar_panel_classifier_model.h5'
    PLOT_SAVE_DIR = 'eda_plots'

    # 1. Perform EDA and Preprocess Data (combines data_loading and data_preprocessing steps)
    train_ds_processed, val_ds_processed, class_names = perform_eda_and_preprocess(
        DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, PLOT_SAVE_DIR
    )

    # 2. Build and Train Model
    trained_model = build_and_train_classifier(
        train_ds_processed,
        val_ds_processed,
        class_names,
        IMG_HEIGHT,
        IMG_WIDTH,
        epochs=EPOCHS,
        model_save_path=MODEL_SAVE_PATH,
        plot_save_dir=PLOT_SAVE_DIR
    )
    print("Model training execution complete.")
    