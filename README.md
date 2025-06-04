Deep learning project in medical field 


<h1 align="center">🧩📸 <strong>Autism Image Classification with Deep Learning</strong></h1>

<p align="center"><em>The puzzle piece is widely recognized in autism advocacy.</em></p>

<p align="center">
  <img src="https://s.wsj.net/public/resources/images/BN-VK947_201710_GR_20171004174436.gif" width="450" />
</p>

<p align="center">Detecting Autism Spectrum Disorder from facial images using VGG16 and Convolutional Neural Networks.


> A personal initiative to explore how deep learning application  can assist early autism detection, blending my interests in AI and healthcare.

This project demonstrates a practical application of **transfer learning and CNNs** to solve a real-world medical problem: **binary image classification of autistic vs. non-autistic individuals**. I built this to:
- Understand how visual cues can be interpreted by machines in healthcare
- Apply deep learning methods like VGG16 to real datasets
- Learn about model generalization, performance tuning, and ethical considerations in medical AI

</p>

---


## Why Autism Detection ❓

Autism Spectrum Disorder (ASD) often goes undiagnosed until later stages, especially in underserved regions.  
Many studies suggest subtle visual patterns may exist — detectable via AI — to assist early-stage screening.

This model aims to **augment clinical diagnosis** (not replace it), offering a **low-cost, scalable solution** to help identify potential ASD cases using image data.


---

## 🛠️ My Role & Key Learnings

| Area                      | Contribution                                           |
|---------------------------|--------------------------------------------------------|
| Dataset Handling          | Structured images into train/val/test with augmentations |
| Model Architecture        | Built CNN using pre-trained VGG16 + custom classifier  |
| Training Strategy         | Implemented early stopping, learning rate reduction    |
| Evaluation                | Used ROC, AUC, Confusion Matrix, Accuracy              |
| Deployment Preparation    | Configured everything to run locally (no cloud reliance) |

> 🚀 Learned how to optimize deep models on limited hardware, tune training loops, and extract model insights with visual diagnostics.

---



## 🧠 Project Workflow

1. Data Loading and Preparation

2. Preprocessing
   
3. Model Building
   
4. Training
   
5. Evaluation

6. Classification report
   

## 📈 Evaluation Metrics

| Metric             | Description                                   |
|--------------------|-----------------------------------------------|
| ✅ Accuracy         | Overall correctness of predictions             |
| 🧾 Classification   | Precision, Recall, F1-score per class          |
| 📊 Confusion Matrix | Visualizes TP, FP, TN, FN                      |
| 📈 ROC & AUC        | Performance measured across all thresholds     |


---

## 📁 Dataset Structure

The dataset is stored in Dataset Folder and structured as:

```
AutismDataset/
├── train/
│   ├── Autistic/
│   └── Non_Autistic/
├── valid/
│   ├── Autistic/
│   └── Non_Autistic/
└── test/
    ├── Autistic/
    └── Non_Autistic/
```

Each subdirectory contains images corresponding to its label.


## 🛠️ Tech Stack

| Category         | Libraries / Tools                                   |
|------------------|-----------------------------------------------------|
| Core Libraries   | `pandas`, `numpy`, `os`, `gc`                       |
| Visualization    | `matplotlib`, `seaborn`                             |
| Image Processing | `opencv-python`, `ImageDataGenerator`               |
| Deep Learning    | `TensorFlow`, `Keras`, `VGG16`                      |
| Metrics & Eval   | `scikit-learn`                                      |
| Platform         | Google Colab + Google Drive                         |



## 🧪 Model Architecture

```python
Base Model: VGG16 (include_top=False)
→ GlobalAveragePooling2D
→ Dense(128, activation='relu')
→ Dropout(0.5)
→ Dense(1, activation='sigmoid')
```

🔧 Only the final classification layers are trained.  
💾 Total Parameters: ~14M | Trainable: ~128K

---

## 🚀 How to Run 

### ⚙️ Setup Instructions

1. Clone the project
2. Place the dataset in `./AutismDataset/`
3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the script or notebook:

```bash

     jupyter notebook autism_classifier.ipynb
```

---

 💡 Use Cases

- Research in **neurodevelopmental disorders**
- Benchmark project for **transfer learning**
- Prototype for **AI-based medical image screening**
  
---

🧑‍💻 Author - PriiiAiVerse
 
🧾 License

This repository is for educational/research purposes. Please verify dataset licensing before usage or publication.

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-blue?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab" />
  <img src="https://img.shields.io/badge/Model-VGG16-red?logo=keras" />
</p>

