# data_loading.py
import tensorflow as tf
import os

def load_solar_panel_data(data_dir, img_height, img_width, batch_size, validation_split=0.2, seed=123):
    """
    Loads solar panel images from the specified directory into TensorFlow Dataset objects.

    Args:
        data_dir (str): Path to the root directory containing class subfolders.
        img_height (int): Desired height for the images.
        img_width (int): Desired width for the images.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of data to reserve for validation (0.0 to 1.0).
        seed (int): Seed for reproducible data splitting.

    Returns:
        tuple: (train_ds, val_ds, class_names)
               train_ds (tf.data.Dataset): Dataset for training.
               val_ds (tf.data.Dataset): Dataset for validation.
               class_names (list): List of class names inferred from subfolder names.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Error: Data directory not found at '{data_dir}'")

    print(f"\n--- Loading dataset from: {data_dir} ---")
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical' # Use 'categorical' for one-hot encoded labels
    )

    class_names = train_ds.class_names
    print(f"Class Names: {class_names}")
    print(f"Found {len(train_ds) * batch_size} training images (approx).")
    print(f"Found {len(val_ds) * batch_size} validation images (approx).")

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    # Example usage when running this script directly for verification
    DATA_DIR = r'C:\Users\Admin\OneDrive\Desktop\SolarGuard_Project\Faulty_solar_panel'
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 32

    try:
        train_ds, val_ds, class_names = load_solar_panel_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
        print("\nData loading successful.")
    except FileNotFoundError as e:
        print(e)