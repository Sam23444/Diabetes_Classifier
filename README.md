# Diabetes_Classifier
DIABETES PREDICTION USING SUPERVISED CLASSIFICATION MODELS

This project applies various supervised classification algorithms to predict the likelihood of diabetes based on patient data. The dataset contains multiple features such as glucose levels, BMI, age, and more.

TABLE OF CONTENTS
Project Overview
Dataset Information
Supervised Models Implemented
Project Structure
How to Run
Results
License


PROJECT OVERVIEW
Diabetes is a chronic condition that affects millions of people worldwide. Early prediction and intervention can help manage the disease effectively. This project aims to:

EXPLORE AND PROCESS THE DATASET.
Train and evaluate various supervised classification models.
Compare model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

DATASETS INFORMATION
The dataset used for this project is stored in diabetes.csv and includes the following features:
Pregnancies: Number of times the patient was pregnant.
Glucose: Plasma glucose concentration.
BloodPressure: Diastolic blood pressure (mm Hg).
SkinThickness: Triceps skinfold thickness (mm).
Insulin: 2-Hour serum insulin (mu U/ml).
BMI: Body mass index (weight in kg/(height in m)^2).
DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history.
Age: Age of the patient.
Outcome: Target variable (0 for non-diabetic, 1 for diabetic).
The goal is to predict the Outcome (whether a patient is diabetic or not) based on the other features.

SUPERVISED MODELS IMPLEMENTED
The following supervised classification models were trained and evaluated:
Logistic Regression
Decision Trees
Random Forest
Support Vector Machines (SVM)
k-Nearest Neighbors (k-NN)
Naive Bayes
Gradient Boosting (XGBoost)
Each model is evaluated using accuracy and confusion matrix to determine the best-performing model.

PROJECT STRUCTURE
diabetes.csv: Dataset used for training and evaluation.
diabetes_classification.py: Python script containing the source code for data preprocessing, model training, and evaluation.
README.md: Project documentation (this file).
How to Run
PREREQUISITES
Ensure you have Python 3.8+ installed along with the following libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost (for Gradient Boosting)
You can install the necessary libraries using pip:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost

STEPS
Clone this repository to your local machine:
git clone <repository-url>
cd <repository-name>
Ensure the dataset file diabetes.csv is present in the project directory.

Run the Python script:
python diabetes_classification.py
The script will:
Load the dataset.
Preprocess the data (including feature scaling).
Train various supervised models.
Display the confusion matrix and accuracy for each model.
Identify the best-performing model and use it to predict whether a new patient is diabetic.
Results
The models are evaluated based on accuracy, confusion matrix, and classification metrics. The best model is selected based on the highest accuracy score.

Example Output:

Training and evaluating Logistic Regression...
Confusion Matrix for Logistic Regression:
[[120  15]
 [ 10  35]]
Accuracy for Logistic Regression: 0.8500

Training and evaluating Random Forest...
Confusion Matrix for Random Forest:
[[125  10]
 [  5  40]]
Accuracy for Random Forest: 0.9000

The best model is: Random Forest with an accuracy of 0.9000
The script will then predict whether the new input data represents a diabetic patient.

License
This project is licensed under the MIT License - see the LICENSE file for details.

