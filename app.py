import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Global model path
MODEL_PATH = "fake_profile_model.h5"

# Function to load dataset
@st.cache_data
def import_data():
    file_path = "dataset.txt"  # Ensure the dataset is in the same directory
    try:
        data = pd.read_csv(file_path)
        
        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        
        # Convert object columns to numeric using Label Encoding
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to split dataset
def split_dataset(data):
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values.reshape(-1, 1)  # Labels
    
    encoder = LabelEncoder()
    Y = encoder.fit_transform(y)  # Convert labels to 0 and 1
    
    return train_test_split(X, Y, test_size=0.2, random_state=42)

# Function to train model
def train_model():
    data = import_data()
    if data is None:
        return None
    
    train_x, test_x, train_y, test_y = split_dataset(data)
    
    model = Sequential([
        Dense(200, input_shape=(train_x.shape[1],), activation='relu', name='fc1'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(200, activation='relu', name='fc2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid', name='output')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=50)  # Reduced epochs for quick training
    accuracy = model.evaluate(test_x, test_y)[1] * 100
    
    # Save the trained model
    model.save(MODEL_PATH)
    
    return accuracy

# Function to load the trained model
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

# Streamlit UI
st.title("🚀 Fake Profile Detection System")

# Sidebar for navigation
menu = st.sidebar.radio("Navigation", ["Home", "Train Model", "Predict Profile", "View Dataset"])

if menu == "Home":
    st.write("### Welcome to the Fake Profile Detection System")
    st.write("This application uses a Machine Learning model to detect whether a social media profile is genuine or fake.")

elif menu == "Train Model":
    st.subheader("Train the ML Model")
    if st.button("Train Now"):
        accuracy = train_model()
        if accuracy is not None:
            st.success(f"✅ Model Trained Successfully with {accuracy:.2f}% Accuracy")
        else:
            st.error("❌ Training failed. Check dataset format.")

elif menu == "Predict Profile":
    st.subheader("Enter Profile Details for Prediction")
    account_age = st.number_input("Account Age", min_value=0, max_value=50, value=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    user_age = st.number_input("User Age", min_value=13, max_value=100, value=25)
    link_desc = st.text_input("Link Description")
    status_count = st.number_input("Status Count", min_value=0, value=0)
    friend_count = st.number_input("Friend Count", min_value=0, value=0)
    location = st.text_input("Location")
    location_ip = st.text_input("Location IP")

    if st.button("Predict"):
        model = load_trained_model()
        if model is None:
            st.error("❌ Model is not trained yet. Please train the model first.")
        else:
            input_data = np.array([[account_age, 1 if gender == "Male" else 0, user_age, len(link_desc), 
                                    status_count, friend_count, len(location), len(location_ip)]], dtype=np.float32)
            
            prediction = (model.predict(input_data) > 0.5).astype(int)
            result = "✅ Genuine Profile" if prediction[0][0] == 0 else "❌ Fake Profile"
            st.success(f"Prediction: {result}")

elif menu == "View Dataset":
    st.subheader("Dataset Preview")
    df = import_data()
    if df is not None:
        st.write(df.head())
