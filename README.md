# Breast-Cancer-Classification
This repository contains a comprehensive Jupyter Notebook designed as a hands-on workshop for Supervised and Unsupervised Machine Learning. Using the Breast Cancer Wisconsin Dataset, this project walks through the end-to-end pipeline of a classification project, from data preprocessing to hyperparameter tuning and clustering.

🚀 Overview
The goal of this analysis is to predict whether a tumor is malignant or benign based on features like radius, texture, and symmetry. It also explores finding hidden patterns in the data using unsupervised learning.

📊 Dataset
Name: Breast Cancer Wisconsin (Diagnostic) Dataset

Features: 30 numeric attributes (mean, standard error, and "worst" measurements of cell nuclei).

Target: Binary Classification (Malignant / Benign).

🛠️ Key Components
1. Data Preprocessing
Feature standardization using StandardScaler.

Data splitting into training (80%) and testing (20%) sets.

2. Supervised Learning: Model Comparison
We evaluate three popular classification algorithms:

Random Forest (Ensemble method)

K-Nearest Neighbors (KNN) (Distance-based)

Support Vector Machine (SVM) (Hyperplane-based)

3. Optimization
Hyperparameter Tuning: Utilizing GridSearchCV to optimize the Support Vector Machine (the top-performing model in this study).

Evaluation Metrics: Accuracy scores, Precision, Recall, and F1-score via classification_report.

4. Unsupervised Learning (Optional Section)
K-Means Clustering: Exploring the data without labels.

Elbow Method: Determining the optimal number of clusters.

Silhouette Score: Measuring the quality of the clusters.

💻 Requirements
To run this notebook, you need Python and the following libraries installed:

Bash
pip install numpy pandas matplotlib seaborn scikit-learn
📈 Results Summary
Top Model: Support Vector Machine (SVM) typically yields the highest accuracy (~98% after tuning).

Clustering: K-Means with K=2 aligns closely with the original labels, showing strong internal structure in the dataset.

🎓 Learning Objectives
Understanding the ML pipeline: Load → Explore → Preprocess → Model → Tune → Evaluate.

Differences between Supervised and Unsupervised learning.

How to use Scikit-learn for automated hyperparameter optimization.

💡 Suggested "Next Step" for your GitHub:
Would you like me to help you generate a high-quality "Model Comparison" bar chart image that you can embed directly into this README? (I can use your Nano Banana image tool for this).
