# Advanced Rice Leaf Disease Diagnosis using EfficientNet & Grad-CAM

---

### **Project Overview**

Rice is a primary staple for more than half of the world's population, but its production is frequently threatened by various diseases. This project implements a state-of-the-art computer vision pipeline to automate the detection and classification of rice leaf diseases, including **Bacterial Blight**, **Blast**, **Brown Spot**, and **Tungro**.

Leveraging **Keras 3** with a **PyTorch** backend, the system achieves rapid GPU-accelerated training and high-precision inference. Beyond simple classification, the platform integrates **Explainable AI (XAI)** via **Grad-CAM**, providing visual heatmaps that highlight the specific focal points the model used for its diagnosis, ensuring transparency and trust for agricultural experts.

---

### **Key Features**

* **High-Performance Backend:** Native GPU acceleration using Keras 3 and PyTorch on Windows.
* **EfficientNet Architecture:** Utilizes `EfficientNetB0` transfer learning for superior feature extraction with minimal parameters.
* **Explainable AI (Grad-CAM):** Dynamic focal visualization to map model attention to leaf lesions.
* **Automated Data Pipeline:** End-to-end ingestion, augmentation, and preprocessing for robust training.
* **Interactive Diagnostic App:** A production-ready Streamlit dashboard for real-time field diagnostics.
* **Cross-Framework Compatibility:** Robust handling of TensorFlow, PyTorch, and NumPy tensor types.

---

### **Dataset**

* **Classes:** Bacterialblight, Blast, Brownspot, Tungro.
* **Input Resolution:** 224x224 RGB images.
* **Source:** Expert-labeled rice disease image repositories.

#### **Target Classes**

| Class Name | Description |
| ---------- | ----------- |
| `Bacterialblight` | Systematic leaf wilting caused by bacteria. |
| `Blast` | Fungal infection appearing as spindle-shaped lesions. |
| `Brownspot` | Small, circular brown lesions with yellow halos. |
| `Tungro` | Viral disease causing yellowing and stunting. |

---

### **Project Structure**

```bash
Rice-Leaf-Diagnosis/
│
├── data/                     # Train, validation, and test datasets
├── notebooks/                # Jupyter notebooks for model research
├── models/                   # Saved production-grade Keras models
├── visualizations/           # Grad-CAM focal maps and metric plots
├── src/
│   ├── data_loader.py        # Dataset ingestion & augmentation
│   ├── model_builder.py      # EfficientNet transfer learning architecture
│   ├── interpretability.py   # Grad-CAM focal point logic (Torch Hooks)
│   ├── evaluation.py         # Confusion matrices & performance plots
│   └── streamlit_app.py      # Production Streamlit dashboard
│
├── requirements.txt          # Python dependencies (Keras 3 + Torch)
├── .gitignore                # Optimized repository exclusion list
└── README.md
```

---

### **How It Works**

### **1. Data Augmentation & Ingestion**
The system uses a randomized augmentation layer to improve generalization:
* Horizontal & Vertical Flips
* Random Rotations
* Intensity adjustments

### **2. EfficientNet Transfer Learning**
The model leverages pre-trained ImageNet weights, freezing the base layers to retain low-level feature detection while fine-tuning the dense head for agricultural specifics.

| Layer Type | Purpose |
| ---------- | ------- |
| `EfficientNetB0` | Base feature extractor (frozen) |
| `GlobalAveragePooling` | Dimensionality reduction |
| `Dropout (0.3)` | Overfitting prevention |
| `Dense (128)` | Specialized feature learning |
| `Softmax` | Final probability distribution |

---

### **3. Explainability (Grad-CAM)**

To avoid "black-box" decisions, the system uses PyTorch forward hooks to capture activations from the `top_conv` layer of the EfficientNet backbone.

---

### **How to Run**

### **1. Environment Setup**
Install the optimized dependencies:
```bash
pip install -r requirements.txt
```

### **2. Launch Analysis Notebook**
Explore the full training and XAI pipeline:
```bash
jupyter notebook notebooks/AI system for rice leaf.ipynb
```

### **3. Launch Production App**
Start the real-time diagnostic dashboard:
```bash
python3.11 -m streamlit run src/streamlit_app.py
```

---
**Developed with Keras 3 & PyTorch Backend**
