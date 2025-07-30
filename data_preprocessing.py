# data_preprocessing.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # For opening image files
import os

def perform_eda_and_preprocess(data_dir, img_height, img_width, batch_size, plot_save_dir='eda_plots', validation_split=0.2, seed=123):
    """
    Performs Exploratory Data Analysis (EDA) and applies preprocessing steps
    (normalization, augmentation) to the datasets. Saves EDA plots.

    Args:
        data_dir (str): Path to the root directory containing class subfolders.
        img_height (int): Desired height for the images.
        img_width (int): Desired width for the images.
        batch_size (int): Number of images per batch.
        plot_save_dir (str): Directory to save EDA plots.
        validation_split (float): Fraction of data to reserve for validation.
        seed (int): Seed for reproducible data splitting.

    Returns:
        tuple: (processed_train_ds, processed_val_ds, class_names)
               processed_train_ds (tf.data.Dataset): Training dataset after preprocessing.
               processed_val_ds (tf.data.Dataset): Validation dataset after preprocessing.
               class_names (list): List of class names.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Error: Data directory not found at '{data_dir}'")

    os.makedirs(plot_save_dir, exist_ok=True) # Ensure plot save directory exists

    # --- Initial EDA (Image Counts and Sample Display) ---
    print(f"\n--- Starting Exploratory Data Analysis (EDA) for: {os.path.basename(data_dir)} ---")

    # Get class names by listing subdirectories (assuming they are class names)
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        print("Warning: No class subfolders found in data_dir. Please check your data structure.")
        # Fallback if no subfolders found, but will likely fail later
        class_names = []

    # Count Images per Class
    print("\nImage counts per class:")
    image_counts = {}
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
            image_counts[class_name] = num_images
            print(f"- {class_name}: {num_images} images")
        else:
            print(f"Warning: Class folder '{class_name}' not found at '{class_path}'")
            image_counts[class_name] = 0

    # Display Sample Images and save to file
    print("\nDisplaying sample images from each class (first image found) and saving to eda_plots:")
    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        sample_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        if sample_images:
            img_path = os.path.join(class_path, sample_images[0])
            try:
                img = Image.open(img_path)
                plt.subplot(2, 3, i + 1)
                plt.imshow(img)
                plt.title(class_name)
                plt.axis('off')
            except Exception as e:
                print(f"Could not open image {img_path}: {e}")
                plt.subplot(2, 3, i + 1)
                plt.text(0.5, 0.5, f"Error loading\n{class_name}", horizontalalignment='center', verticalalignment='center')
                plt.title(class_name)
                plt.axis('off')
        else:
            print(f"No images found for class: {class_name} at {class_path}")
            plt.subplot(2, 3, i + 1)
            plt.text(0.5, 0.5, f"No images\n{class_name}", horizontalalignment='center', verticalalignment='center')
            plt.title(class_name)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, 'sample_images_per_class.png'))
    plt.close() # Close the plot to free memory

    print("\n--- EDA Complete. ---")

    # --- Load Dataset using TensorFlow's image_dataset_from_directory ---
    print(f"\n--- Loading dataset from: {data_dir} for preprocessing ---")
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical'
    )
    # Ensure class_names match what TF loaded
    class_names = train_ds.class_names
    print(f"Class Names (from TF dataset): {class_names}")
    print(f"Found {len(train_ds) * batch_size} training images (approx).")
    print(f"Found {len(val_ds) * batch_size} validation images (approx).")


    # --- Data Preprocessing (Normalization & Augmentation) ---
    print("\n--- Applying Data Preprocessing (Normalization & Augmentation) ---")

    # Rescale pixel values from [0, 255] to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Data augmentation for the training set
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Apply augmentation only to the training set
    processed_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    processed_train_ds = processed_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    processed_val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("--- Data Preprocessing Complete. Datasets are ready for model training. ---")
    return processed_train_ds, processed_val_ds, class_names

if __name__ == "__main__":
    # Example usage when running this script directly
    DATA_DIR = r'C:\Users\Admin\OneDrive\Desktop\SolarGuard_Project\Faulty_solar_panel'
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 32
    PLOT_SAVE_DIR = 'eda_plots'

    try:
        # This will perform EDA and then preprocess the data
        train_ds_processed, val_ds_processed, class_names = perform_eda_and_preprocess(
            DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, PLOT_SAVE_DIR
        )
        print("\nEDA and Preprocessing successful for example data.")
    except FileNotFoundError as e:
        print(e)