# Churn Prediction using Neural Network (Streamlit App)
# Overview
This Streamlit app serves as a predictive tool for telecom companies to anticipate customer churn. Customer churn, the phenomenon where customers switch from one service provider to another, significantly impacts a company's revenue. The app leverages a pre-trained neural network model built with TensorFlow's Keras API to analyze various customer attributes and forecast the likelihood of churn.

# Key Objectives
Churn Prediction: Provide a user-friendly interface for users to input customer-specific attributes and receive real-time churn predictions.
Insight Generation: Assist telecom companies in identifying potential churners among their customer base to take proactive retention measures.
Neural Network Modeling: Utilize advanced neural network architecture to accurately predict churn based on historical customer data. 

#Functionalities
User Input Collection: Enable users to input a range of customer attributes encompassing demographic information (e.g., 'Senior Citizen', 'Gender'), service-related details ('Tenure', 'Monthly Charges'), and plan specifics ('Contract', 'Payment Method').

Neural Network Prediction: Employ a pre-trained neural network model (loaded from final_best_model.plk) to process the user-input data and generate churn predictions based on learned patterns from historical data.

Data Preprocessing: Perform essential data preprocessing steps, including encoding categorical features using a label encoder (label_encoder.joblib) and employing one-hot encoding for categorical variables like 'Payment Method' and 'Contract'. Numerical feature scaling is conducted using a scaler (scaler.joblib) to ensure alignment with the model's requirements.

Prediction Display: After inputting customer attributes, users can obtain churn predictions by clicking the "Predict" button. The app then processes the data, computes churn likelihoods using the neural network model, and displays the predicted churn status ('Yes' or 'No').

Streamlit Interface: Develop a user-friendly interface using the Streamlit library, allowing for a seamless user experience while interacting with the app for churn prediction.

# Usage
Launch the Streamlit app by visiting the deployed URL.
Fill in the customer attributes (Senior Citizen, Tenure, Monthly Charges, etc.).
Click the "Predict" button to get the churn prediction.

# Deployment
The app has been deployed on Streamlit Cloud and is accessible here: https://churningprediction.streamlit.app/

# Tutorial Video
Learn how to use the app by watching the tutorial video on YouTube:

# File Structure
app.py: Streamlit application script.

create_model.py: File containing the code for creating the neural network model.

final_best_model.plk: Serialized neural network model file.

scaler.joblib: Serialized scaler for feature scaling.

label_encoder.joblib: Serialized label encoder for categorical variables.

requirements.txt: File containing required dependencies.




