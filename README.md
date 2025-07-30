# SolarGuard: Intelligent Defect Detection on Solar Panels using DeepLearning

## Project Overview
This project leverages Deep Learning and Computer Vision to automate the inspection and maintenance of solar panels. The primary aim is to enhance the efficiency and reduce operational costs of solar energy systems by intelligently detecting and localizing defects.

## Problem Statement
Solar panel efficiency is often reduced by the accumulation of dust, snow, bird droppings, and physical/electrical damage. Manual monitoring is time-consuming and expensive. This project develops an automated system to accurately identify and localize these issues, leading to optimized maintenance and improved energy production.

## Project Objectives
1.  **Classification Task:** Develop a deep learning model to categorize solar panel images into six conditions: Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, and Snow-Covered.
2.  **Object Detection Task:** Develop an object detection model to identify and precisely localize dust, bird droppings, and physical/electrical damage on solar panels.

## Tools and Technologies
* **Programming Language:** Python
* **Classification Framework:** TensorFlow (with MobileNetV2)
* **Object Detection Framework:** Ultralytics YOLOv8
* **Data Annotation:** Roboflow
* **Web Application:** Streamlit
* **Environment:** Google Colab (for GPU acceleration)
* **Libraries:** NumPy, Pandas, Pillow, OpenCV-Python, Scikit-learn, Matplotlib, Seaborn, PyYAML


## Setup and Running the Project

### 1. Local Setup (for Classification / Development)
1.  **Clone the repository:** `git clone https://github.com/Roshan-25-cbe/SolarGuard-Intelligent-Defect-Detection-on-Solar-Panels-using-DeepLearning.git`
2.  **Navigate to project directory:** `cd SolarGuard_Project`
3.  **Create a virtual environment:** `python -m venv .venv`
4.  **Activate environment:**
    * Windows: `.venv\Scripts\activate`
    * macOS/Linux: `source .venv/bin/activate`
5.  **Install Classification dependencies:** `pip install -r requirements_classification.txt`
6.  **Run Streamlit App:** `streamlit run app.py`

### 2. Object Detection Training & Evaluation (Recommended via Google Colab)

Due to GPU requirements and environment complexities, the object detection task is best run in Google Colab using a dedicated script.

1.  **Open a FRESH, NEW Google Colab Notebook.**
2.  **Set Runtime to GPU:** `Runtime > Change runtime type > GPU`.
3.  **Paste the content of the `object_detection.py` script into the first cell.**
4.  **Run the cell.** This script will:
    * Install necessary libraries (`roboflow`, `ultralytics`).
    * Download your specific dataset from Roboflow using your API key.
    * Verify the dataset structure.
    * Load a pre-trained YOLOv8n model.
    * Train the model on your data.
    * Evaluate the trained model.
    * Save results and the trained model (`best.pt`) in `runs/detect/solar_panel_od_run/weights/` and `runs/detect/solar_panel_od_run/`.

## Results Highlights

### Classification Task
* **Model:** MobileNetV2 (TensorFlow)
* **Validation Accuracy:** 0.6723 (67.23%)
* **Validation Loss:** 0.9759
*(See `eda_plots/confusion_matrix.png` and `eda_plots/training_history.png` for more details.)*

### Object Detection Task
* **Model:** YOLOv8n (Ultralytics)
* **Overall mAP50-95:** 0.1099
* **Overall mAP50:** 0.344
* **Overall mAP75:** 0.110
* **Performance by Class (mAP50):**
    * Bird-drop: 0.0113
    * Dusty: 0.233
    * Electrical-damage: 0.721
    * Physical-damage: 0.412
*(See `runs/detect/solar_panel_od_run/` for detailed metrics and plots.)*

## Future Work
* **Dataset Expansion:** Collect more diverse and larger datasets.
* **Model Optimization:** Explore larger YOLOv8 models (YOLOv8s, YOLOv8m) and fine-tune hyperparameters.
* **Full Streamlit Integration:** Integrate object detection inference into the Streamlit application.
* **Real-time Processing:** Explore deployment on edge devices or cloud platforms.
* **Defect Severity Analysis:** Develop methods to quantify defect severity.

## Presentation

The project has been presented, and you can find the slides [here](https://1drv.ms/p/c/ff0f2d98614978aa/EVSvWfpUSopEt6my-GBbESUBx9omYGJ5B4ELesZeNmIrhQ?e=p5kwmu)

## Project Author

-   Roshan
-   GitHub: [Roshan-25-cbe](https://github.com/Roshan-25-cbe)
-   LinkedIn: [www.linkedin.com/in/roshan-angamuthu-195ba230a] 

## Contact

For any inquiries or collaboration opportunities, feel free to contact me:
* Email: [roshana36822@gmail.com] 



