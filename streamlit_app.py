import streamlit as st
import pandas as pd
import joblib  # for loading your saved model

# --- Load the trained model ---
best_rf_model = joblib.load("best_xgb_model.joblib")

# --- Load training features to get default mean values ---
X_train_url = "https://drive.google.com/file/d/1yjM1opg0s9Lg6xlblx828o2xkaRbvou1/view?usp=sharing"
X_train = pd.read_csv(X_train_url)  # Make sure this file contains the exact training features

# --- Define top 10 features to take manual input ---
top_features = ['nitrogen', 'srad_min', 'srad_median', 'srad_mean',
                'tmin_median', 'tmax_median', 'tmean_median',
                'tmax_mean', 'tmin_mean', 'tmean_std']

def user_input_features():
    st.sidebar.header("Input Top 10 Features")

    input_data = {}
    
    # Manual input for top 10 features
    for feat in top_features:
        default_val = float(X_train[feat].mean())
        val = st.sidebar.number_input(f'Input {feat}', value=default_val)
        input_data[feat] = val

    # Auto-fill the remaining features with mean values
    for feat in X_train.columns:
        if feat not in top_features:
            input_data[feat] = float(X_train[feat].mean())

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches the training order
    input_df = input_df[best_rf_model.feature_names_in_]

    return input_df

# --- Streamlit App Interface ---
st.title("Foreign Crop Yield Prediction")

# Get user inputs as DataFrame
input_df = user_input_features()

# Display user inputs
st.subheader("User Input Features")
st.write(input_df)

# Make prediction
prediction = best_rf_model.predict(input_df)

# Display prediction
st.subheader("Predicted Yield")
st.write(f"{prediction[0]:.4f}")
