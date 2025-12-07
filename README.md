# 🌿 Plantify - Your Plant Doctor

**Plantify** is an end-to-end Computer Vision pipeline designed to detect and diagnose plant diseases from leaf images. By leveraging Deep Learning techniques, specifically Convolutional Neural Networks (CNNs) and Transfer Learning, this project automates the identification of crop pathologies to assist in timely agricultural intervention.
<img width="3999" height="1178" alt="image" src="https://github.com/user-attachments/assets/791bf623-f2eb-4f9f-bf44-b591838cf456" />

### 🚀 **Live Application**

Check out the deployed frontend application here: **[Plantify App](https://plantifyzs.lovable.app)**

-----

## 📂 Repository Structure

This repository contains the iterative development of the disease detection models, moving from initial experiments to a fully scaled solution.

| Notebook File | Description |
| :--- | :--- |
| **`Plant_Disease_Detection_Scaled.ipynb`** | **✨ The Final Model.** This is the robust, production-ready notebook trained on **29 classes** across **10 different plant species**.  |
| **`Plant_Disease_Detection_filtered.ipynb`** | **Logic Building & Proof of Concept.** A targeted implementation focusing on **9 classes** (2 plant species). This notebook was used to refine the data processing pipeline and model architecture before scaling up. It includes the full pipeline: Data Merging, ResNet50 Transfer Learning, Error Analysis, and Visualization. |
| **`PlantDiseaseDetection.ipynb`** | **Initial Prototype.** The sandbox notebook containing early experiments, exploratory data analysis (EDA), and baseline model attempts. |

-----

## 🛠️ How It Works

The project follows a rigorous Data Science lifecycle, from raw data acquisition to model deployment.

### 1\. Data Pipeline

  * **Source:** Integrates the **PlantVillage** dataset and an **Indoor Plant Disease** dataset from Kaggle.
  * **Preprocessing:** Images are resized to `128x128` (or `224x224` for ResNet), converted to tensors, and normalized using ImageNet mean/std values to accelerate convergence.
  * **Data Splitting:** A standard 80/20 train-validation split is used to ensure robust evaluation.

### 2\. Model Architectures

We benchmarked three distinct approaches to identify the best performer:

1.  **Baseline Custom CNN:** A lightweight architecture with 3 convolutional blocks to establish a performance baseline.
2.  **Optimized Custom CNN:** The baseline model tuned using **Grid Search** (optimizing Learning Rate and Batch Size).
3.  **Transfer Learning (ResNet50):** A heavy-weight industry-standard model pre-trained on ImageNet. We froze the feature extraction layers and fine-tuned the final classification head for our specific 29 disease classes.

### 3\. Training & Optimization

  * **Loss Function:** CrossEntropyLoss.
  * **Optimizer:** Adam Optimizer.
  * **Regularization:** Dropout (0.5) and **Early Stopping** (patience=3) were implemented to prevent overfitting.

### 4\. Advanced Visualization

To ensure the model is learning meaningful features (and not just memorizing background noise), we implemented:

  * **Saliency Maps:** Visualizing pixel intensity gradients to see *where* the model is looking on the leaf.
  * **t-SNE Clustering:** projecting the high-dimensional latent space into 2D to visualize how the model groups similar diseases.
  * **Confusion Matrices:** To identify specific disease pairs that confuse the model.

-----

## 💻 How to Use the Notebooks

To reproduce the results or train the model yourself, follow these steps:

### Prerequisites

  * A Google account (if using Colab) or a local Python environment with Jupyter.
  * A **Kaggle API Token** (`kaggle.json`).

### Step-by-Step Guide

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Shaguns26/PlantDiseaseDetection.git
    cd PlantDiseaseDetection
    ```

2.  **Open the Scaled Notebook**
    Open `Plant_Disease_Detection_Scaled.ipynb` in Jupyter Notebook or Google Colab.

3.  **Setup Kaggle Credentials**
    The notebook requires a `kaggle.json` file to download the datasets.

      * **In Colab:** The notebook has a cell to upload the file directly.
      * **Local:** Ensure `kaggle.json` is in your `~/.kaggle/` directory.

4.  **Run the Cells**
    Execute the blocks sequentially. The notebook will:

      * ⬇️ Download and unzip the datasets.
      * 🔄 Merge and filter the data into a clean structure.
      * 🧠 Train the models (Baseline, Optimized, and ResNet).
      * 📊 Generate evaluation metrics (Accuracy, RMSE) and plots.

-----

## 📊 Results

The project demonstrated the effectiveness of Transfer Learning for agricultural image classification.

  * **Custom CNN:** Achieved respectable accuracy but struggled with complex textures.
  * **ResNet50:** Significantly outperformed custom architectures, achieving **\>93% accuracy** on the validation set, proving that pre-trained features are highly effective for extracting leaf pathology patterns.

-----

## 👤 Author

**Shagun Sharma**

  * [GitHub Profile](https://github.com/Shaguns26)
  * [Plantify Application](https://plantifyzs.lovable.app)
