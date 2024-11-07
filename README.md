# Oil Spill Classification Project🛢️🌊

## Introduction🎯:

This machine learning project focuses on predicting the occurrence of oil spills based on various features provided in the dataset. The dataset consists of a set of masked parameters which influence the likelihood of an oil spill. The goal of this project is to build a machine learning model capable of accurately classifying the presence or absence of oil spills (`target` feature) using several classification algorithms. The challenge is to handle class imbalance, high-dimensional data, and ensure accurate predictions for both the majority and minority classes.

## Problem Statement🚨:

- **Objective:** Predict the occurrence of oil spills based on masked feature.
- **Target Variable:** `target` (indicating the occurrence of an oil spill).
- **Features:** A combination of numeric and categorical features representing environmental and operational factors.
- **Challenge:** The dataset suffers from class imbalance, with the majority of the data belonging to the `Category 0` (no spill).

## Tools & Technologies Used:🛠️

- **Programming Language:** Python🐍
- **Libraries & Frameworks:**
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical computing and array manipulation.
  - **Matplotlib & Seaborn**: For data visualization (graphs, plots, and statistical analysis).
  - **Scikit-learn**: For machine learning models, preprocessing (scaling, PCA), model evaluation, and hyperparameter tuning.
  - **XGBoost**: For gradient boosting and classification.
  - **Imbalanced-learn**: For handling imbalanced datasets using SMOTETomek.
  - **Statsmodels**: For calculating the Variance Inflation Factor (VIF) to check for multicollinearity.
  - **Pickle**: For model serialization and saving/loading models.

## ML Model Used:🤖
Various classification models were implemented to predict the target variable:
   - **Logistic Regression**
   - **Ridge (L2) Logistic Regression**
   - **Lasso (L1) Logistic Regression**
   - **K-Nearest Neighbors (KNN)**
   - **Decision Tree**
   - **Random Forest**
   - **XGBoost**

## Model Performance 🚀

The following models were tested, and their accuracy scores on the test data are as follows:

| Model                           | Accuracy |
|----------------------------------|----------|
| Logistic Regression              | 0.985    |
| Ridge (L2) Logistic Regression   | 0.987    |
| Lasso (L1) Logistic Regression   | 0.984    |
| K-Nearest Neighbors (KNN)        | 0.992    |
| Decision Tree                    | 0.956    |
| Random Forest                    | 0.995    |
| XGBoost                          | 0.995    |

## Conclusion📈

- **Best Performing Model:** XGBoost achieved the highest accuracy of 0.995, making it the preferred model for this task.
- **Model Deployment:** The final model was saved using `pickle` for deployment and future use in making predictions on new data.

## How to Use 🛠️
1. Clone the repository.
2. Install the necessary dependencies listed in `requirements.txt`.
3. Load the model using:
    ```python
    import pickle
    model = pickle.load(open('Oil_spill_app.pkl', 'rb'))
    ```
4. Preprocess your input data (scale it and apply PCA).
5. Predict oil spill occurrences:
    ```python
    predictions = model.predict(your_input_data)
    ```
