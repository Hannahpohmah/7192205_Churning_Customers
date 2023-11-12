import streamlit as st
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from create_model import create_model

from joblib import load
model_path= 'D:\\Ashesi Edu\\Sophomore Year\\INTRO TO AI\\Pohmahmbuh_ASS3\\final_best_model.plk'

# Load the Keras model

with open(model_path, 'rb') as f:
    best_model= pickle.load(f)

scaler_path = 'D:\\Ashesi Edu\\Sophomore Year\\INTRO TO AI\\Pohmahmbuh_ASS3\\scaler.joblib'
label_path='D:\\Ashesi Edu\\Sophomore Year\\INTRO TO AI\\Pohmahmbuh_ASS3\\label_encoder.joblib'

scaler = load(scaler_path)

label_encoder = load(label_path)


st.title("Nueral network model using TensorFlow's Keras API for Churn prediction")

def main():
    senior_citizen = st.number_input('Senior Citizen', min_value=0, max_value=1)
    tenure = st.number_input('Tenure', min_value=0)
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check'])
    contract = st.selectbox('Contract', ['Month-to-month'])


    if st.button("Predict"):
        categorical_features = [gender, partner, dependents, multiple_lines, internet_service,
                                online_security, online_backup, device_protection, tech_support,
                                paperless_billing]
        categorical_features_encoded = label_encoder.fit_transform(categorical_features)       
        
        data = {
            'PaymentMethod': [payment_method],
            'Contract': [contract]
        }
        df = pd.DataFrame(data)
        encoded_df = pd.get_dummies(df, columns=['PaymentMethod', 'Contract'])

        # Flatten the encoded_df to make it a 1D array
        flattened_encoded_df = encoded_df.values.flatten()

        input_features = np.concatenate([
            np.array([senior_citizen, tenure, monthly_charges, total_charges, *categorical_features_encoded]),
            flattened_encoded_df
        ]).reshape(1, -1)

        # Scale the input features
        input_features_scaled = scaler.transform(input_features)

        # Make predictions
        prediction = best_model.predict(input_features_scaled)

        label_mapping = {1: 'Yes', 0: 'No'}

        predicted_churn_label =int(prediction[0])
        # Map the predicted label using the dictionary
        predicted_churn = label_mapping[predicted_churn_label]
        # Display the prediction
        st.write(f"Predicted Churn: {predicted_churn}")   

if __name__ == "__main__":
    main()
 








