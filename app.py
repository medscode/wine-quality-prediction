"""
Wine Quality Prediction using Random Forest Classifier
This project uses a Random Forest model to classify red wine as Good Quality (1) or Bad Quality (0)
based on 11 physicochemical features from the Wine Quality dataset.
Data collection --> Data analysis & visualization --> Label Binarization --> Train-Test Split --> Random Forest Training --> Prediction
Wines with quality score >= 7 are labeled as Good, others as Bad.
"""

# Importing the Dependencies
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ---------------- Streamlit Page Setup ----------------
st.set_page_config(page_title="Wine Quality Prediction", layout="wide")
st.title("🍷 Wine Quality Prediction using Random Forest")
st.write("A Random Forest Classifier that predicts whether a red wine is Good Quality or Bad Quality based on its physicochemical properties.")


# ---------------- Data Collection ----------------
# The @st.cache_data decorator caches the dataset so it's only loaded once
# @st.cache_resource caches the trained model so it's only trained once per session

@st.cache_data
def load_data():
    # loading the dataset to a Pandas DataFrame
    # Using a public URL since Streamlit Cloud can't access local /content/ paths
    wine_dataset = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv')
    return wine_dataset


@st.cache_resource
def train_model(wine_dataset):

    # ---------------- Data Preprocessing ----------------
    # separate the data and Label
    X = wine_dataset.drop('quality', axis=1)

    # ---------------- Label Binarization ----------------
    # Wine quality scores range from 3-8. We binarize: quality >= 7 means Good Quality (1), else Bad Quality (0)
    Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)


    # ---------------- Train & Test Split ----------------
    # 80% training, 20% testing. random_state=3 ensures consistent splits across runs.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


    # ---------------- Model Training: Random Forest Classifier ----------------
    # Random Forest is an ensemble of many decision trees. Each tree votes and the majority wins.
    # This gives it much better accuracy than a single decision tree and prevents overfitting.
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)


    # ---------------- Model Evaluation: Accuracy Score ----------------
    # accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    return model, test_data_accuracy, X, Y


# Load data and train model
with st.spinner("Loading data and training Random Forest..."):
    wine_dataset = load_data()
    model, test_data_accuracy, X, Y = train_model(wine_dataset)


# ---------------- Displaying Model Info ----------------
st.success(f"Model trained successfully! Test Accuracy: {test_data_accuracy:.2%}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", f"{wine_dataset.shape[0]}")
col2.metric("Features", "11")
col3.metric("Good Wines", f"{(Y == 1).sum()}")
col4.metric("Test Accuracy", f"{test_data_accuracy:.2%}")

st.divider()


# ---------------- Data Collection Display ----------------
st.subheader("Dataset Preview")
st.write(f"**Shape of the dataset:** {wine_dataset.shape}  (1599 rows, 12 columns)")

# first 5 rows of the dataset
st.write("**First 5 rows of the dataset:**")
st.dataframe(wine_dataset.head())

# checking for missing values
st.write("**Missing values in each column:**")
st.dataframe(wine_dataset.isnull().sum().to_frame(name='Missing Values').T)


st.divider()


# ---------------- Data Analysis and Visualization ----------------
st.subheader("Data Analysis and Visualization")

col_a, col_b = st.columns(2)

with col_a:
    # number of values for each quality
    st.write("**Distribution of Quality Scores**")
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    sns.countplot(x='quality', data=wine_dataset, ax=ax1)
    ax1.set_title('Count of each quality level')
    st.pyplot(fig1)

with col_b:
    # volatile acidity vs Quality
    st.write("**Volatile Acidity vs Quality**")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sns.barplot(x='quality', y='volatile acidity', data=wine_dataset, ax=ax2)
    st.pyplot(fig2)

col_c, col_d = st.columns(2)

with col_c:
    # citric acid vs Quality
    st.write("**Citric Acid vs Quality**")
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    sns.barplot(x='quality', y='citric acid', data=wine_dataset, ax=ax3)
    st.pyplot(fig3)

with col_d:
    # Correlation heatmap
    # 1. Positive Correlation - as one feature increases, the other also increases
    # 2. Negative Correlation - as one feature increases, the other decreases
    st.write("**Correlation Heatmap**")
    correlation = wine_dataset.corr()
    # constructing a heatmap to understand the correlation between the columns
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',
                annot=True, annot_kws={'size': 8}, cmap='Blues', ax=ax4)
    st.pyplot(fig4)


st.divider()


# ---------------- Building a Predictive System ----------------
st.subheader("🔮 Wine Quality Predictor")
st.write("Adjust the sliders in the sidebar to describe a wine sample, and the model will predict if it is Good Quality or Bad.")

# Sidebar sliders — user enters the 11 feature values
st.sidebar.header("Wine Sample Features")
st.sidebar.caption("Adjust the physicochemical properties below.")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.5, 0.1)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.0, 1.6, 0.5, 0.01)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.36, 0.01)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 16.0, 6.1, 0.1)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.62, 0.071, 0.001)
free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1.0, 72.0, 17.0, 1.0)
total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6.0, 289.0, 102.0, 1.0)
density = st.sidebar.slider("Density", 0.99, 1.01, 0.9978, 0.0001)
pH = st.sidebar.slider("pH", 2.7, 4.1, 3.35, 0.01)
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.8, 0.01)
alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.5, 0.1)


# collecting user input into the exact tuple format from your notebook
input_data = (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
              free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)


# ---------------- Displaying the Prediction Result ----------------
left, right = st.columns(2)

with left:
    st.subheader("Prediction Result")

    if (prediction[0] == 1):
        st.success("### ✅ Good Quality Wine")
        st.write("This wine has the physicochemical profile of a high-quality wine (quality score ≥ 7).")
    else:
        st.error("### ❌ Bad Quality Wine")
        st.write("This wine does not meet the threshold for good quality (quality score < 7).")

    st.write(f"**Raw prediction output:** `{prediction}`")

with right:
    st.subheader("Your Input Values")
    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                     'density', 'pH', 'sulphates', 'alcohol']
    input_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_data
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True)


# ---------------- Footer ----------------
st.divider()
with st.expander("ℹ️ About this Project"):
    st.markdown("""
    **Model:** Random Forest Classifier (default parameters, 100 trees)
    **Dataset:** Red Wine Quality (1599 samples, 11 features, quality score 3-8)
    **Preprocessing:** Label binarization — quality >= 7 becomes 1 (Good), else 0 (Bad)
    **Train/Test Split:** 80/20 with random_state=3
    **Accuracy:** ~92.5% on test data

    **Features used:**
    Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugar, Chlorides,
    Free Sulfur Dioxide, Total Sulfur Dioxide, Density, pH, Sulphates, Alcohol

    **Why Random Forest?** Random Forest combines multiple decision trees and takes
    a majority vote, making it much more robust than a single tree and resistant to overfitting.
    """)
