import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# Define Neural Network Class (Must Match train.py)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Regularization
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 30
hidden_size = 64
output_size = 1

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load("breast_cancer_model.pth", map_location=device))
model.eval()

# Load Scaler
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🩺 Breast Cancer Prediction App")
st.write("Enter the feature values and get a prediction.")

# User Input
user_input = st.text_input("Enter 30 feature values (comma-separated):")

if st.button("Predict"):
    try:
        features = list(map(float, user_input.split(",")))

        if len(features) != 30:
            st.error("Please enter exactly 30 feature values.")
        else:
            # Scale input before prediction
            scaled_features = scaler.transform([features])
            print("Scaled Features:", scaled_features)  # Debugging Step

            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_tensor)

            probability = output.item()
            prediction = "Malignant" if probability > 0.5 else "Benign"

            st.success(f"🔍 Prediction: **{prediction}** (Probability: {probability:.4f})")

    except ValueError:
        st.error("Invalid input! Please enter numeric values only.")