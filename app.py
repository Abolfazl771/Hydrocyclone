import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Define the MLP model architecture (same as in Colab)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.0)
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.dropout(self.fc2(x)))
        x = torch.relu(self.dropout(self.fc3(x)))
        x = self.fc4(x)
        return x

# Load the saved scaler
scaler_pickle_path = 'scaler.pkl'
with open(scaler_pickle_path, 'rb') as f:
    loaded_scaler = pickle.load(f)

# Load the saved model with weights_only=True for future compatibility
model = MLP()
model.load_state_dict(torch.load('m.pth', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Streamlit page title
st.title('(MLP) پیش‌بینی با مدل شبکه عصبی ')

# User inputs (received as text inputs)
x1 = st.text_input('دبی:', value="0.0")
x2 = st.text_input('قطر سرریز:', value="0.0")
x3 = st.text_input('قطر ته ریز:', value="0.0")
x4 = st.text_input('زاویه مخروطی اصلی:', value="0.0")

# Convert input to a float array and prepare it for the scaler
input_values = [float(x1), float(x2), float(x3), float(x4)]
input_array = np.array(input_values).reshape(1, -1)

# Pass input as a DataFrame with the correct feature names for the scaler
input_df = pd.DataFrame(input_array, columns=['x1', 'x2', 'x3', 'x4'])
input_scaled = loaded_scaler.transform(input_df)  # Scale the input

# Convert the scaled input to a PyTorch tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# When the user presses the 'Predict' button
if st.button('پیش‌بینی'):
    with torch.no_grad():  # Disable gradient calculations
        prediction = model(input_tensor).item()  # Make the prediction and get the result
        st.write(f'پیش‌بینی: {prediction}')
