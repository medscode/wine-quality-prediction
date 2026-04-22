# Wine Quality Prediction using Random Forest

A machine learning web application that predicts whether a red wine is of Good Quality or Bad Quality based on 11 physicochemical features, using a Random Forest Classifier.

## Overview

This project uses the Red Wine Quality dataset to train a Random Forest model that classifies wines into two categories — Good Quality (quality score ≥ 7) and Bad Quality (quality score < 7). The project walks through data loading, exploratory analysis, visualization, label binarization, model training, and evaluation, and is deployed as an interactive Streamlit web application.

## Technologies Used

- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (Random Forest, train-test split, accuracy score)
- Streamlit (web deployment)

## Dataset

- **Source:** Red Wine Quality dataset
- **Samples:** 1,599 red wine samples
- **Features:** 11 physicochemical properties — Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides, Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol
- **Target:** Quality score (3 to 8), binarized into Good (≥7) and Bad (<7)

## Methodology

1. **Data Collection** — Loaded dataset as a Pandas DataFrame.
2. **Data Analysis & Visualization** — Statistical summaries, count plots, bar plots of key features vs. quality, and a correlation heatmap to identify feature relationships.
3. **Data Preprocessing** — Separated features (X) from the target label.
4. **Label Binarization** — Converted the multi-class quality score into a binary label (1 for Good, 0 for Bad) using a threshold of 7.
5. **Train/Test Split** — 80% training, 20% testing, with `random_state=3` for reproducibility.
6. **Model Training** — Trained a `RandomForestClassifier` with default parameters (100 estimators).
7. **Evaluation** — Measured accuracy on the test set.
8. **Predictive System** — Built an interactive Streamlit interface that accepts new wine samples via sliders and returns Good/Bad predictions.

## Results

- **Test Accuracy:** ~92.5%
- **Key Insight:** Alcohol content and volatile acidity were among the strongest predictors of wine quality based on the correlation heatmap.

## Running Locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

Deployed on Streamlit Community Cloud.
