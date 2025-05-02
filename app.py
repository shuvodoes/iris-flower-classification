import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the model
model = joblib.load('iris_model.pkl')

# Load Iris dataset
iris = load_iris()

# Set the title for the Streamlit app
st.title('Iris Flower Prediction')
st.write('This is an interactive app to predict the species of Iris flower based on its features.')

# Input Sidebar
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 3.0)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# DataFrame for the input features
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_df = pd.DataFrame(input_data, columns=iris.feature_names)

# Display input features
st.write('Input Features:', input_df)

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display prediction
st.subheader('Prediction:')
predicted_class = iris.target_names[prediction][0]
st.markdown(f'<p style="background-color: #FE7743; padding: 10px; font-size: 16px; font-weight: bold;">The predicted Iris species is **{predicted_class}**.</p>', unsafe_allow_html=True)



st.write('Prediction Probabilities:')
for i, species in enumerate(iris.target_names):
    st.write(f'{species}: {prediction_proba[0][i]:.2f}')

# Accuracy and Classification Report
st.subheader('Model Evaluation on Test Set:')
# Load the dataset again and split for evaluation
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Evaluation
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy of the model: **{accuracy * 100:.2f}%**')

# Classification Report
st.write('Classification Report:')
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
classification_rep_df = pd.DataFrame(classification_rep).transpose()
st.write(classification_rep_df)


