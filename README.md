# WINE-QUALITY-CLASSIFICATION-USING-ML-AND-DATA-ANALYSIS
## Overview

The **Wine Quality Classification** project leverages machine learning techniques to classify wine into "good" and "bad" quality categories based on its physicochemical properties. 
By analyzing features such as alcohol content, pH, acidity, and density, the project aims to automate wine quality assessment with high accuracy and consistency.

This project implements multiple machine learning models, including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naïve Bayes (NB), Random Forest, and XGBoost, to predict wine quality. 
The project uses the wine quality dataset from the UCI Machine Learning Repository.

---

## Features

- Data preprocessing and feature engineering.
- Comparative analysis of machine learning models.
- Hyperparameter tuning using Grid Search.
- Evaluation metrics like accuracy, precision, recall, and F1 score.
- Visualization of data distribution and model performance.

---

## Dataset

The dataset used in this project is publicly available on the **UCI Machine Learning Repository**:
- **Red Wine Dataset**: 1599 instances.
- Each dataset includes 11 input features and 1 output feature (`quality`).

---

## Installation
1. Install Python
Make sure you have Python 3.6 or higher installed on your system. If not, download and install Python from the official Python website.

Install Libraries Manually
If you don't have a requirements.txt file, you can manually install each required library using the following commands:

pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install xgboost

Project Workflow
Data Preprocessing:

Handling missing values.
Normalizing features for consistency.
Feature Engineering:

Analyzing correlations between features and the target variable.
Model Training:

Training models like KNN, SVM, NB, Random Forest, and XGBoost.
Hyperparameter Tuning:

Optimizing model parameters using Grid Search.
Evaluation:

Comparing models based on accuracy, precision, recall, and F1 score.
Results
Random Forest achieved the highest accuracy: ~94.5% (red wine) and ~93.2% (white wine).
Alcohol and pH were identified as the most significant features influencing wine quality.
Visualizations

Data Distribution: Visualized using histograms and bar plots.
Model Performance: Evaluated using confusion matrices and accuracy plots.
Feature Importance: Analyzed for Random Forest and XGBoost models.
License


Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

Contact
For any questions or feedback, feel free to contact:

Name: Om Uniyal
Email: omuniyal0@gmail.com
