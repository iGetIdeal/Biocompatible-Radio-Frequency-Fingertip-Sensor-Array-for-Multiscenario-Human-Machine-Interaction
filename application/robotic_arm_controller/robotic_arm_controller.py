# =======================================================================================
# Real-time Direct Command Transmission System for STM32 via Serial
#
# All special command logic (like start/stop) is removed. The system acts as a pure
# bridge between the BP model's output and the serial port.
#
# Required library: pySerial (install via: pip install pyserial)
# =======================================================================================

import os
import time
import threading
import queue
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
import serial

# --- System Configuration: File Paths and Serial Port Settings ---
# Users must modify these paths and settings according to their environment.

# File System Paths
DATA_FOLDER_PATH = os.path.join('D:', 'path', 'to', 'your', 'data', 'folder') # Path to the folder containing VNA output .xlsx files
MODEL_PATH = "../models/bp_net_model.pth"                                             # Path to the pre-trained BP Neural Network model
SCALER_PATH = "../models/scaler.joblib"                                               # Path to the fitted StandardScaler object
LABEL_ENCODER_PATH = "../models/label_encoder.joblib"                                 # Path to the fitted LabelEncoder object

# Serial Port Configuration
SERIAL_PORT = 'COM3'  # The serial port to use (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
BAUD_RATE = 9600      # The baud rate for the serial communication (must match the STM32 setting)


# --- Core Component: Neural Network Definition ---

class BPNet(nn.Module):
    """
    Defines the structure of a Backpropagation Neural Network (BPNet) with two hidden layers.
    """
    def __init__(self, num_classes):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Defines the forward pass logic of the model."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- Data Processing and Prediction Module (Unchanged) ---

def find_latest_xlsx_file(folder_path):
    """Finds and returns the most recently modified .xlsx file in a specified directory."""
    try:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        if not files: return None
        return max(files, key=os.path.getmtime)
    except FileNotFoundError:
        return None

def extract_features_from_excel(file_path):
    """Reads a VNA data file and extracts key features."""
    try:
        df = pd.read_excel(file_path, header=1).iloc[1:].reset_index(drop=True)
        if 'Frequency (Hz)' not in df.columns or 'Returnloss (dB)' not in df.columns: return None
        df_filtered = df[(df['Frequency (Hz)'] > 400 * 10**6) & (df['Frequency (Hz)'] < 700 * 10**6)]
        if df_filtered.empty: return None
        min_returnloss_row = df_filtered.loc[df_filtered['Returnloss (dB)'].idxmin()]
        return (min_returnloss_row['Frequency (Hz)'], min_returnloss_row['Returnloss (dB)'])
    except Exception:
        return None

def predict_command(features_tuple, model, scaler, label_encoder):
    """Processes features using the loaded model to predict a command."""
    try:
        features_np = np.array([list(features_tuple)])
        scaled_features = scaler.transform(features_np)
        input_tensor = torch.FloatTensor(scaled_features)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output.data, 1)
        return label_encoder.inverse_transform(predicted_idx.numpy())[0]
    except Exception:
        return None

# --- Background Worker Thread (Unchanged) ---

def data_processing_thread(command_queue):
    """
    This function runs as a background thread, monitoring for new files,
    predicting commands, and placing them into a thread-safe queue.
    """
    try:
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        model = BPNet(num_classes=len(label_encoder.classes_))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Worker Thread: BP model and transformers loaded successfully.")
    except Exception as e:
        print(f"Worker Thread FATAL ERROR: Could not load model or files: {e}")
        return

    last_processed_file = None
    print("\n--- Worker Thread: Started monitoring data folder ---")
    while True:
        latest_file = find_latest_xlsx_file(DATA_FOLDER_PATH)
        if latest_file and latest_file != last_processed_file:
            features = extract_features_from_excel(latest_file)
            if features:
                command = predict_command(features, model, scaler, label_encoder)
                if command is not None:
                    print(f"Worker Thread: Predicted command '{command}'")
                    command_queue.put(command)
            last_processed_file = latest_file
        time.sleep(1)

# --- Main Program Entry Point and Serial Communication Logic ---

def main():
    """
    Initializes the system, opens the serial port, starts the background thread,
    and enters a loop to transmit every predicted command to the STM32.
    """
    # Initialize the queue for inter-thread communication.
    command_queue = queue.Queue()
    
    # Create and start the data processing thread as a daemon.
    data_thread = threading.Thread(target=data_processing_thread, args=(command_queue,), daemon=True)
    data_thread.start()

    # --- Initialize Serial Port ---
    ser = None
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"Main Program: Successfully opened serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
        print("Main Program: Now transmitting all predicted commands...")
    except serial.SerialException as e:
        print(f"Main Program FATAL ERROR: Could not open serial port '{SERIAL_PORT}'. Error: {e}")
        print("Please check the port name, permissions, and ensure the device is connected.")
        return # Exit if the port cannot be opened.

    # --- Main Transmission Loop ---
    try:
        while True:
            # Get a command from the queue. This call will block until a command is available.
            command = command_queue.get()
            
            # Encode the command string to bytes and append a newline character '\n'.
            # This acts as a reliable delimiter for the receiver (STM32).
            data_to_send = (command + '\n').encode('utf-8')
            
            # Write the data to the serial port.
            ser.write(data_to_send)
            
            # Print a confirmation to the console.
            print(f"Transmitted to {SERIAL_PORT}: '{command}'")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down...")
    finally:
        # Ensure the serial port is properly closed upon exiting.
        if ser and ser.is_open:
            ser.close()
            print(f"Main Program: Serial port {SERIAL_PORT} closed.")
        print("Main Program has shut down.")

if __name__ == '__main__':
    main()
