import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd

#load the trained model,scaler pickle,onehot
model=load_model('model.h5')
with open('encoder_geography.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)
with open('encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

import streamlit as st

# Input fields
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])  # <-- Fixed syntax here

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],  # <-- fixed closing bracket
    'HasCrCard': [has_cr_card],          # <-- fixed variable name (was: has cr_card)
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Assume OneHotEncoder was fit on: ['France', 'Germany', 'Spain']
# One-hot encode the geography input
geo_encoded = label_encoder_geo.transform([[geography]])  # geography is the selected string, e.g., 'France'
geo_encoded_array = geo_encoded.toarray()

# Create a DataFrame with proper column names
geo_encoded_df = pd.DataFrame(
    geo_encoded_array,
    columns=label_encoder_geo.get_feature_names_out(['Geography'])
)
   

# Merge the original input with the encoded geography dataframe
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
   # or keep if not present



# Scale the input data using the fitted scaler
input_data_scaled = scaler.transform(input_data)

# Predict using the loaded model
prediction = model.predict(input_data_scaled)  # <-- FIXED 'preddict' typo

# Get the predicted probability for the positive class (usually class 1)
prediction_proba = prediction[0][0]
st.write(f"probability:{prediction_proba:.2f}")
# Interpret the result
if prediction_proba > 0.5:
    st.write("user yerripuku")  # example for predicted class = 1
else:
    st.write("user mqnchodu")   # example for predicted class = 0












