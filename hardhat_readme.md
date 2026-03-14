# Hardhat Detection System (YOLOv8)

## Project Overview

This project implements a real-time hardhat/helmet detection system using the YOLOv8 object detection framework. The primary goal is to monitor video feeds (e.g., webcam, safety camera) to identify individuals wearing hardhats ("helmet") and those who are not ("no helmet").

The system features robust inference logic (`inference.py`) that includes:
* **Temporal Smoothing:** Uses history to stabilize predictions and reduce rapid flickering/false positives.
* **Alert System:** Triggers a visible **ALERT** on screen if an unprotected head is detected for a configurable duration.

---

## Repository Contents

This repository contains all the necessary components—the inference script, the training notebook, and the two stages of trained model weights.

| File | Description |
| :--- | :--- |
| **`inference.py`** | **Real-Time Detection Script.** The Python file containing the `HelmetDetectionSystem` class and the logic to load a trained model and run live inference on a webcam feed. |
| **`hardhat.ipynb`** | **Training and Analysis Notebook.** The Jupyter Notebook detailing the entire model lifecycle: dataset preparation, configuration, **two-stage training** (frozen and fine-tuned), and performance analysis. |
| **`stage2_full_finetune_yolov8s_best.pt`** | **Final Model Weights (Recommended).** The best checkpoint achieved after the final **full fine-tuning stage**. Use this model for production inference. |
| **`stage1_frozen_best.pt`** | **Intermediate Model Weights.** The best checkpoint from the initial **frozen-backbone training stage**. Included for reproducibility and comparative analysis. |

---

## Installation

This project requires Python 3.8+ and several machine learning libraries. It is highly recommended to use a virtual environment (e.g., Conda or Venv).

### Environment Setup

**Using Conda:**

```bash
conda create -n helmet_detector python=3.9
conda activate helmet_detector
```

**Or using venv:**

```bash
python -m venv helmet_detector
source helmet_detector/bin/activate  # On Windows: helmet_detector\Scripts\activate
```

### Install Dependencies

Install the necessary libraries, particularly ultralytics (for YOLOv8) and opencv-python (for video stream handling).

```bash
pip install ultralytics opencv-python pandas
```

---

## Running Inference

Once your environment is set up, you can start the real-time detection system.

1. Ensure your environment (`helmet_detector`) is active.
2. Run the main script from the terminal. The script is configured by default to use the final, optimized model (`stage2_full_finetune_yolov8s_best.pt`) and connect to your default webcam (ID 0).

```bash
python inference.py
```

**Note:** The webcam feed window will open. Press the **`q`** key to quit the stream.

### Customization

You can easily adjust core settings at the bottom of the `inference.py` file:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `model_path` | `'stage2_full_finetune_yolov8s_best.pt'` | Change this to switch to `stage1_frozen_best.pt` if desired. |
| `alert_delay` | `3.0` seconds | The duration a "NO HELMET" detection must persist before the on-screen ALERT is triggered. |
| `cam_id` | `0` | The ID of the camera to use (0 is usually the built-in webcam). |

---

## Training and Performance Analysis

The full details of the model development are contained within the Jupyter Notebook.

### Two-Stage Training

The training was performed in two sequential stages to optimize performance and prevent catastrophic forgetting:

1. **Stage 1 (Frozen):** Initial training on custom data with the large YOLOv8 backbone frozen.
2. **Stage 2 (Fine-Tuned):** Unfreezing all layers and continuing training using the Stage 1 weights as a starting point.

### Inspecting Results

To inspect the quantitative results (mAP, loss curves, confusion matrices) for both training stages, open the `hardhat.ipynb` notebook and run the code chunks under the following header:

**`## Step 6: Analyze Results (Evaluation Metrics and Confusion Matrices)`**

This section of the notebook contains the code used to generate the final performance metrics and visualizations for the project evaluation.

---

## Example Output

When running the system, you'll see:
* Live video feed with bounding boxes around detected heads
* Green boxes for "helmet" detections
* Red boxes for "no helmet" detections
* ALERT banner appears when someone without a helmet is detected for more than 3 seconds