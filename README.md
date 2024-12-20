# AI_projects


<img width="1428" alt="Screen Shot 2024-12-20 at 12 24 29 PM" src="https://github.com/user-attachments/assets/23e64eda-5fad-4340-9e24-a0d85c442fec" />

# Breast Cancer Detection with K-Nearest Neighbors (KNN)

This project implements a **machine learning model** using the **K-Nearest Neighbors (KNN)** algorithm to predict whether breast cancer tumors are **benign** or **malignant** based on input features.

## Project Structure

- **/images**: Contains images for visualization (e.g., awareness images, diagnosis illustrations).
- **/models**: Contains the trained models and scaler used for prediction (`knn_model.joblib`, `scaler.joblib`).
- **/notebooks**: Contains the Jupyter notebook (`Project_Cancer.ipynb`) where the model was trained and evaluated.
- **/scripts**: Contains the main Python script (`project.py`) for running the predictions and analysis.
- **/data**: Contains the dataset (`features.csv`).

## Overview

The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** to train the KNN classifier. The dataset contains features like:
- Radius, texture, perimeter, area, and other measurements related to tumor shape and texture.

## Model Performance

- **Model Used**: K-Nearest Neighbors (KNN)
- **Accuracy**: 98.6% (Evaluated on the test set)
- **Precision**: 97.8%
- **Recall**: 98.3%
- **F1 Score**: 98.0%

- 
![image](https://github.com/user-attachments/assets/28e63b14-530c-48cb-a7d2-02ed3809463b)

knn_cv.best_params_

![image](https://github.com/user-attachments/assets/5af8aa16-ed36-4feb-8055-c3f8d9228aee)

Correlation matrix 

![image](https://github.com/user-attachments/assets/3ec2ef25-b6f2-4925-ae6e-599123c7a034)


## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/MuhammadjonR/breast-cancer-detection.git
