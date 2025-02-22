# data-science-student-admission

# Project Overview

This project analyzes factors influencing graduate school admission chances using machine learning models. The dataset includes GRE scores, TOEFL scores, university ratings, statement of purpose scores, letters of recommendation, CGPA, and research experience.

# Folder Structure
```
project-root/
│── data/             # Contains raw dataset
│── scripts/          # Scripts for analysis and modeling
│── outputs/          # Outputs from analysis (visualizations, models, reports)
│── requirements.txt  # Dependencies for the project
│── README.md         # Project documentation
```

# Usage

### 1. Setup the Project:
Clone the repository.
Ensure you have Python installed.
Install required dependencies using the requirements.txt file.
```sh
pip install -r requirements.txt
```

### 2. Preprocess the Data:
Run the data preprocessing script to clean and normalize the dataset.
```sh
python scripts/01_preprocess.py
```

### 3. Perform Exploratory Analysis:
Generate visualizations and summary statistics to understand the dataset.
```sh
python scripts/02_exploratory_analysis.py
```

### 4. Train the Machine Learning Model:
Train a regression model to predict admission chances.
```sh
python scripts/03_model_training.py
```

### 5. Make Predictions:
Use the trained model to predict admission chances based on user input.
```sh
python scripts/04_prediction.py
```

# Requirements

Ensure you have the required dependencies installed by running:
```sh
pip install -r requirements.txt
```

# Acknowledgments

dataset name: Graduate Admission 2  
dataset author: Mohan S Acharya  
dataset source: [Graduate Admission 2](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)  
citation: Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019