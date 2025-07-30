# prepare_yolov8_dataset.py
import os
import random
import shutil

def prepare_yolov8_dataset(source_data_dir, dest_dataset_dir, images_per_defect_class=20, train_ratio=0.8):
    """
    Randomly selects images from specified defect classes, copies them to
    the YOLOv8 dataset structure (images/train, images/val), and
    creates corresponding empty .txt label files (labels/train, labels/val).

    Args:
        source_data_dir (str): Path to the original full dataset (e.g., 'Faulty_solar_panel').
        dest_dataset_dir (str): Path to the target YOLOv8 dataset root (e.g., 'custom_od_dataset').
        images_per_defect_class (int): Number of images to randomly select per defect class.
        train_ratio (float): Ratio of images to be used for training (e.g., 0.8 for 80% train, 20% val).
    """
    print(f"\n--- Preparing YOLOv8 Dataset Structure for Annotation ---")
    print(f"Source Raw Data: {source_data_dir}")
    print(f"Destination YOLOv8 Dataset Root: {dest_dataset_dir}")
    print(f"Images to select per DEFECT class: {images_per_defect_class}")
    print(f"Train/Validation Split Ratio: {train_ratio*100}% Train, {(1-train_ratio)*100}% Validation")

    if not os.path.exists(source_data_dir):
        print(f"Error: Source data directory not found at '{source_data_dir}'")
        print("Please ensure your 'Faulty_solar_panel' folder exists at this path.")
        return

    # Define the subdirectories for images and labels
    img_train_dir = os.path.join(dest_dataset_dir, 'images', 'train')
    img_val_dir = os.path.join(dest_dataset_dir, 'images', 'val')
    label_train_dir = os.path.join(dest_dataset_dir, 'labels', 'train')
    label_val_dir = os.path.join(dest_dataset_dir, 'labels', 'val')

    # Create all necessary directories, clearing previous content if exists
    for dir_path in [img_train_dir, img_val_dir, label_train_dir, label_val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path) # Danger: Clears existing content!
        os.makedirs(dir_path)
        print(f"Created/Cleaned directory: {dir_path}")

    # Define defect classes for object detection
    defect_classes = ['Dusty', 'Bird-Drop', 'Electrical-Damage', 'Physical-Damage']

    print(f"\nProcessing only defect classes: {defect_classes}")
    print("\nCopying and preparing structure...")

    total_images_copied = 0
    for class_name in defect_classes:
        source_class_path = os.path.join(source_data_dir, class_name)
        if not os.path.exists(source_class_path):
            print(f"Warning: Source class directory '{source_class_path}' not found. Skipping.")
            continue

        all_images_in_class = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not all_images_in_class:
            print(f"No images found in '{source_class_path}'. Skipping.")
            continue

        # Ensure we don't try to sample more images than available
        num_to_sample = min(images_per_defect_class, len(all_images_in_class))
        sampled_images = random.sample(all_images_in_class, num_to_sample)
        print(f"  - From '{class_name}': Randomly selected {num_to_sample} images out of {len(all_images_in_class)}.")

        # Split sampled images into train and validation sets
        random.shuffle(sampled_images)
        train_count = int(num_to_sample * train_ratio)
        train_images = sampled_images[:train_count]
        val_images = sampled_images[train_count:]

        # Copy images and create empty label files
        for img_name in train_images:
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(img_train_dir, img_name)
            shutil.copy(src_img_path, dest_img_path)

            label_filename = os.path.splitext(img_name)[0] + '.txt'
            open(os.path.join(label_train_dir, label_filename), 'a').close() # Create empty .txt file
            total_images_copied += 1

        for img_name in val_images:
            src_img_path = os.path.join(source_class_path, img_name)
            dest_img_path = os.path.join(img_val_dir, img_name)
            shutil.copy(src_img_path, dest_img_path)

            label_filename = os.path.splitext(img_name)[0] + '.txt'
            open(os.path.join(label_val_dir, label_filename), 'a').close() # Create empty .txt file
            total_images_copied += 1

    print(f"\n--- Dataset preparation complete. Total images copied: {total_images_copied} ---")
    print(f"Images are in: {img_train_dir} and {img_val_dir}")
    print(f"EMPTY label files created in: {label_train_dir} and {label_val_dir}")
    print("\n**NEXT CRITICAL STEP:** You MUST now use a tool like LabelImg to manually annotate (draw boxes and add labels) on these images. The `.txt` files are currently empty and need to be filled with your bounding box data!")


if __name__ == "__main__":
    # --- ACTION REQUIRED: Verify/Adjust these paths and parameters ---
    SOURCE_DATASET_DIR = r'C:\Users\Admin\OneDrive\Desktop\SolarGuard_Project\Faulty_solar_panel'
    DEST_YOLOV8_DATASET_DIR = r'C:\Users\Admin\OneDrive\Desktop\SolarGuard_Project\custom_od_dataset'
    IMAGES_TO_COPY_PER_DEFECT_CLASS = 50 # How many images to pick from EACH defect class
    TRAIN_VALIDATION_SPLIT_RATIO = 0.8 # 80% for training, 20% for validation
    # --- END ACTION REQUIRED ---

    prepare_yolov8_dataset(
        source_data_dir=SOURCE_DATASET_DIR,
        dest_dataset_dir=DEST_YOLOV8_DATASET_DIR,
        images_per_defect_class=IMAGES_TO_COPY_PER_DEFECT_CLASS,
        train_ratio=TRAIN_VALIDATION_SPLIT_RATIO
    )