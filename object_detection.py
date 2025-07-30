# Combined Ultralytics YOLOv8 Training & Evaluation Script for Google Colab

# --- 1. Install necessary libraries ---
# Install Roboflow for dataset download
!pip install -q roboflow

# Install Ultralytics for YOLOv8 model training and evaluation
!pip install -q ultralytics

# --- 2. Import Libraries ---
import os
from roboflow import Roboflow
from ultralytics import YOLO

# --- 3. Roboflow Dataset Download ---
# IMPORTANT: Replace "YOUR_ROBOFLOW_API_KEY" with your actual API key from Roboflow.
# You can find it on your Roboflow account settings.
# This code snippet comes directly from Roboflow's "Download Dataset" -> "Colab" option.
print("--- Downloading Dataset from Roboflow ---")
rf = Roboflow(api_key="xWtJBmhueEk0zP0pN5BS") # Replace with your actual API key
project = rf.workspace("solarguardproject").project("solarpaneldetection-9t9cw")
version = project.version(2) # Using version 2 as per your previous output
dataset = version.download("yolov8") # This downloads and extracts the dataset

# --- 4. Identify Downloaded Dataset Folder and data.yaml Path ---
# Roboflow usually downloads to a folder like 'your_project_name-your_version'
# We'll automatically find the folder name created by Roboflow.
downloaded_dataset_folder_name = None
for item in os.listdir('/content/'):
    item_path = os.path.join('/content/', item)
    if os.path.isdir(item_path) and 'data.yaml' in os.listdir(item_path):
        downloaded_dataset_folder_name = item
        break

if downloaded_dataset_folder_name is None:
    print("\nERROR: Roboflow dataset folder not found. Please check Roboflow API key and project details.")
    # Exit script if dataset not found
    exit()

data_yaml_path = os.path.join('/content', downloaded_dataset_folder_name, 'data.yaml')

# --- 5. Verify Dataset Structure (Optional, but good for confirmation) ---
print(f"\n--- Verifying Dataset Structure for: {data_yaml_path} ---")
print(f"Identified Roboflow dataset folder: /content/{downloaded_dataset_folder_name}")
print(f"Contents of /content/{downloaded_dataset_folder_name}/ :")
!ls -F /content/{downloaded_dataset_folder_name}/

print(f"\nContents of /content/{downloaded_dataset_folder_name}/train/images/ (first 5):")
!ls -F /content/{downloaded_dataset_folder_name}/train/images/ | head -n 5

print(f"\nContents of /content/{downloaded_dataset_folder_name}/train/labels/ (first 5):")
!ls -F /content/{downloaded_dataset_folder_name}/train/labels/ | head -n 5

print(f"\nContent of data.yaml for {downloaded_dataset_folder_name}:")
!cat {data_yaml_path}

# --- 6. Load and Train YOLOv8 Model ---
print("\n--- Starting YOLOv8 Model Training ---")

# Load a pre-trained YOLOv8n model (nano version - smallest and fastest)
# ultralytics will automatically download 'yolov8n.pt' if it's not found in cache.
model = YOLO('yolov8n.pt')

# Training parameters
EPOCHS = 50       # Number of training epochs. Adjust higher for better performance (e.g., 100-300)
IMG_SIZE = 640    # Input image size (e.g., 640x640)
BATCH_SIZE = 16   # Batch size. Adjust based on GPU memory. Colab T4 can often handle 16 or 32.
PROJECT_NAME = 'solar_panel_od_run' # Name for the training run results folder

# Start training
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name=PROJECT_NAME # This will be the folder name inside 'runs/detect/'
)

print("\n--- Model Training Complete! ---")
print(f"Training results saved to: {results.save_dir}")

# --- 7. Evaluate the Trained Model ---
print("\n--- Starting YOLOv8 Model Evaluation ---")

# Evaluate the model on the validation set
# Ultralytics' val() method automatically calculates mAP, precision, recall etc.
# It also saves plots like confusion matrix, F1-curve to the run directory.
metrics = model.val(data=data_yaml_path)

print("\n--- Evaluation Complete! ---")
print(f"mAP50-95 (Average Precision across IoU thresholds 0.5 to 0.95): {metrics.box.map:.4f}")
print(f"mAP50 (Average Precision at IoU=0.5): {metrics.box.map50:.4f}")
print(f"mAP75 (Average Precision at IoU=0.75): {metrics.box.map75:.4f}")

print(f"\nEvaluation results and plots saved to: {metrics.save_dir}")

print("\nAll Object Detection tasks (Download, Train, Evaluate) completed successfully!")
