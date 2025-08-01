# -*- coding: utf-8 -*-
# =============================================================================
# BP_Model_Training_and_Saving.py
#
# Description:
# This script trains a Backpropagation (BP) neural network for a classification
# task. It performs the following steps:
# 1. Loads data from an Excel file.
# 2. Preprocesses the data using Standardization for features and Label Encoding for targets.
# 3. Defines a 3-layer BP neural network using PyTorch.
# 4. Splits the data into training and testing sets.
# 5. Trains the model on the training set.
# 6. Evaluates the model's performance on the test set.
# 7. Saves the trained model weights, the feature scaler, and the label encoder
#    to disk for future inference.
#
# Author: [Your Name/Lab Name]
# Date: 2025-08-01
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# --- Data Loading and Configuration ---
# !!! IMPORTANT !!!
# Please update the 'excel_path' to the absolute or relative path of your dataset file.
excel_path = './BP_model/training_set.xlsx'

# --- Load and Preprocess Data ---
try:
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    print(f"Error: The file was not found at the specified path: {excel_path}")
    print("Please make sure the excel_path variable is set correctly.")
    exit()

# Extract features (X) and labels (y)
X = df[['Frequency', 'Returnloss']].values
y = df['label'].values

# Feature Standardization: Scale features to have zero mean and unit variance.
# The scaler is saved to apply the same transformation to new data during inference.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Label Encoding: Convert string labels into a numerical format.
# The encoder is saved for decoding model predictions back to original labels.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# ---Data Splitting ---
# Split the dataset into 80% for training and 20% for testing.
# 'stratify=y_encoded' ensures the same proportion of labels in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# --- Model Definition ---
# A simple BP neural network with two hidden layers.
class BPNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, num_classes=num_classes):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = BPNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Model Training ---
print("--- Model training started ---")
epochs = 50
batch_size = 8
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        avg_loss = epoch_loss / (len(X_train_tensor) / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
print("--- Training completed ---")

# --- Model Evaluation ---
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_preds = torch.max(test_outputs, 1)
test_acc = accuracy_score(y_test_tensor.numpy(), test_preds.numpy())
print(f"\nAccuracy on test set: {test_acc:.4f}")

# --- Save Model and Preprocessing Objects ---
# Define paths for the saved files. They will be saved in the current working directory.
model_save_path = "../models/bp_net_model.pth"
scaler_save_path = "../models/scaler.joblib"
le_save_path = "../models/label_encoder.joblib"

# Save the model's learned parameters (state dictionary)
torch.save(model.state_dict(), model_save_path)
# Save the StandardScaler instance
joblib.dump(scaler, scaler_save_path)
# Save the LabelEncoder instance
joblib.dump(le, le_save_path)

print(f"\nModel saved to: {model_save_path}")
print(f"Scaler saved to: {scaler_save_path}")
print(f"Label Encoder saved to: {le_save_path}")
