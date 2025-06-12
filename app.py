import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model= tf.keras.models.load_model('model.h5')

# Load all the scalers and encoders

with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geography.pkl", 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)
    
    
#stream lit app
st.title('Customer Churn Prediciton')
 
#user input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
has_cr_card = st.selectbox('Has Credit Card?', [0,1])
is_active_member = st.selectbox('Is Active Member?', [0,1])

credit_score = st.slider('Credit Score', min_value=0, max_value=900, value=300)
age = st.slider('Age', min_value=18, max_value=100, value=24)
tenure = st.slider('Tenure (years)', min_value=0, max_value=10, value=3)

balance = st.number_input('Balance')
estimated_salary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
)

input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_probability =  prediction[0][0]

if prediction_probability > 0.5:
    st.write("The customer is more likely to churn.")
else:
    st.write("The customer is not likely to churn.")
    