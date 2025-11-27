# Pneumonia Chest X-Ray Classification (CNN + Transfer Learning)

A deep learning project that classifies **Normal vs Pneumonia** from chest X-ray images using **ResNet18**, **transfer learning**, and a **single-image prediction interface**.

This project was created as part of my **Machine Learning Capstone** and demonstrates complete end-to-end ML workflow:

1. Data loading

2. Preprocessing & augmentation

3. Transfer learning (ResNet18)

4. Class-imbalance handling

5. Model training with checkpointing

6. Evaluation (Accuracy, Precision, Recall, ROC-AUC)

7. Single-image prediction helper

---

## Project Structure

```
pneumonia-xray-classification
│
├── notebooks/
│   └── pneumonia_cnn_training.ipynb
│
├── models/
│   ├── best_model.pth         # saved best checkpoint
│
├── README.md
└── requirements.txt / Pipenv
```

---

## Project Overview

### **Objective**

Build a deep learning model that accurately detects **pneumonia** from chest X-ray images, supporting medical decision systems.

### **Dataset**

* **Hugging Face Dataset:**
  `keremberke/chest-xray-classification`
* Two classes:

  * **0 → NORMAL**
  * **1 → PNEUMONIA**

### **Input Format**

* X-ray image (`224 × 224`)
* Grayscale images automatically converted to 3-channel for CNNs

---

## Model Architecture

This project uses:

### **1. Transfer Learning**

* Base model: **ResNet18 (ImageNet pre-trained)**
* Final FC layer replaced with:

  ```python
  nn.Linear(num_ftrs, 2)
  ```

### **2. Loss Handling**

* Dataset is imbalanced → pneumonia cases ≈ 3× normal
* Used **weighted CrossEntropyLoss**:

  ```python
  weight = 1 / class_counts
  ```

### **3. Training Enhancements**

* Data augmentation (flip, rotation)
* Learning rate scheduler (`ReduceLROnPlateau`)
* Checkpointing (save best model):

  ```python
  torch.save(model.state_dict(), "best_model.pth")
  ```

---

## Final Results (Test Set)

| Metric                   | Score     |
| ------------------------ | --------- |
| **Accuracy**             | **96%**   |
| **NORMAL Recall**        | **0.99**  |
| **PNEUMONIA Recall**     | **0.95**  |
| **F1-score (Pneumonia)** | **0.97**  |
| **ROC-AUC**              | **0.995** |

### Confusion Matrix

```
[[169   2]
 [ 20 391]]
```

### Interpretation:

* The model **rarely misses pneumonia**
* Low false positives + extremely high sensitivity
* ROC-AUC 0.99 → near-perfect discrimination

## Single Image Prediction

You can run inference on any image:

```python
from PIL import Image

predict_image("path_to_xray.png")
```

Sample output:

```
Prediction: PNEUMONIA
Confidence: 0.9821
NORMAL: 0.0173
PNEUMONIA: 0.9821
```

## Installation

### **Using Pipenv**

```
pipenv install torch torchvision datasets scikit-learn matplotlib tqdm pillow gradio
```

### **Or using pip**

```
pip install torch torchvision datasets scikit-learn matplotlib tqdm pillow gradio
```

---

## How to Run

### **1. Train Model**

Open the notebook:

```
notebooks/pneumonia_cnn_training.ipynb
```

Run cells to:

* Load dataset
* Train model
* Save best checkpoint

### **2. Run Inference**

```python
predict_image("sample_xray.png")
```

---

## Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **Hugging Face Datasets**
* **Matplotlib**

---

## Key Learnings

* How to handle medical image classification
* Dealing with imbalanced datasets
* Applying transfer learning effectively
* Creating checkpoints for the best validation model

