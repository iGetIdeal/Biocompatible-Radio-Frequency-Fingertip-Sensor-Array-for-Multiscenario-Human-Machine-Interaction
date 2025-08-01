# =======================================================================================
# Real-time Morse Code Translation System Driven by a BP Neural Network
#
# The system employs a multi-threaded architecture to decouple data acquisition and
# model inference (worker thread) from the main program's command processing and
# translation logic (main thread), ensuring real-time responsiveness.
#
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

# --- System Configuration: File Paths and Constants ---
# Users must modify the file paths in this section according to their
# deployment environment. The os.path.join() method is used for
# cross-platform compatibility.
DATA_FOLDER_PATH = os.path.join('D:', 'path', 'to', 'your', 'data', 'folder') # Path to the folder containing VNA output .xlsx files
MODEL_PATH = "../models/bp_net_model.pth"                                             # Path to the pre-trained BP Neural Network model
SCALER_PATH = "../models/scaler.joblib"                                               # Path to the fitted StandardScaler object for feature normalization
LABEL_ENCODER_PATH = "../models/label_encoder.joblib"                                 # Path to the fitted LabelEncoder object for label decoding


# --- Static Data: Morse Code to Character Translation Dictionary ---
# This dictionary defines the mapping from Morse code sequences to their
# corresponding alphabetic or numeric characters.
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9'
}


# --- Core Component: Neural Network Definition ---

class BPNet(nn.Module):
    """
    Defines the structure of a Backpropagation Neural Network (BPNet) with two hidden layers.
    
    This network model maps the extracted two-dimensional physical features to
    pre-defined command categories. Its architecture is as follows:
    Input(2) -> Hidden1(16) -> ReLU -> Hidden2(16) -> ReLU -> Output(num_classes).
    """
    def __init__(self, num_classes):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)           # Input layer to first hidden layer
        self.fc2 = nn.Linear(16, 16)          # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(16, num_classes) # Second hidden layer to output layer
        self.relu = nn.ReLU()                 # Activation function

    def forward(self, x):
        """
        Defines the forward pass logic of the model.
        
        Args:
            x (torch.Tensor): The input feature tensor.
        Returns:
            torch.Tensor: The raw output logits from the model.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- Data Processing and Prediction Module ---

def find_latest_xlsx_file(folder_path):
    """
    Finds and returns the most recently modified .xlsx file in a specified directory.

    Args:
        folder_path (str): The directory path to search.
    Returns:
        str or None: The full path to the latest file if found; otherwise, None.
    """
    try:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except FileNotFoundError:
        print(f"Error: The specified folder path does not exist: '{folder_path}'")
        return None

def extract_features_from_excel(file_path):
    """
    Reads data from a VNA data file (.xlsx) and extracts key features.
    The extraction logic finds the point with the minimum return loss within a specific
    frequency range (400-700 MHz) and returns its corresponding frequency and loss value.

    Args:
        file_path (str): The path to the input .xlsx file.
    Returns:
        tuple or None: A tuple of (frequency, return_loss) on success; otherwise, None.
    """
    try:
        df = pd.read_excel(file_path, header=1).iloc[1:].reset_index(drop=True)
        if 'Frequency (Hz)' not in df.columns or 'Returnloss (dB)' not in df.columns:
            return None
        df_filtered = df[(df['Frequency (Hz)'] > 400 * 10**6) & (df['Frequency (Hz)'] < 700 * 10**6)]
        if df_filtered.empty:
            return None
        min_returnloss_row = df_filtered.loc[df_filtered['Returnloss (dB)'].idxmin()]
        return (min_returnloss_row['Frequency (Hz)'], min_returnloss_row['Returnloss (dB)'])
    except Exception:
        return None

def predict_command(features_tuple, model, scaler, label_encoder):
    """
    Processes the extracted features using the loaded model to predict a command.
    The process includes: feature scaling -> tensor conversion -> model inference -> result decoding.

    Args:
        features_tuple (tuple): A tuple containing (frequency, return_loss).
        model (BPNet): The pre-trained PyTorch model.
        scaler (StandardScaler): The fitted feature scaler.
        label_encoder (LabelEncoder): The fitted label encoder.
    Returns:
        str or None: The predicted command string on success; otherwise, None.
    """
    try:
        features_np = np.array([list(features_tuple)])
        scaled_features = scaler.transform(features_np)
        input_tensor = torch.FloatTensor(scaled_features)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output.data, 1)
        return label_encoder.inverse_transform(predicted_idx.numpy())[0]
    except Exception as e:
        print(f"An error occurred during model prediction: {e}")
        return None

# --- Background Worker Thread ---

def data_processing_thread(command_queue):
    """
    This function runs as a background thread, responsible for continuously monitoring
    and processing new files and making predictions. It decouples the time-consuming
    data processing from the main program's UI/logic, communicating results
    asynchronously via a thread-safe queue.

    Args:
        command_queue (queue.Queue): The queue instance for inter-thread communication.
    """
    try:
        # Load all necessary models and transformers at the start of the thread.
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        num_classes = len(label_encoder.classes_)
        model = BPNet(num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Switch model to evaluation mode
        print("Worker Thread: BP model and transformers loaded successfully.")
    except FileNotFoundError as e:
        print(f"Worker Thread FATAL ERROR: Could not find model or transformer files: {e}")
        return

    last_processed_file = None
    print("\n--- Worker Thread: Started monitoring data folder ---")
    while True:
        latest_file = find_latest_xlsx_file(DATA_FOLDER_PATH)
        if latest_file and latest_file != last_processed_file:
            print("-" * 30)
            features = extract_features_from_excel(latest_file)
            if features:
                command = predict_command(features, model, scaler, label_encoder)
                if command is not None:
                    # Put the predicted command into the queue for the main thread to consume.
                    command_queue.put(command)
            last_processed_file = latest_file
        time.sleep(1) # Polling interval of 1 second

# --- Main Program Entry Point and Control Logic ---

def main():
    """
    The main function initializes the system, starts the background thread,
    and handles the main control loop. This loop retrieves commands from the queue,
    manages system state, and performs Morse code translation and output.
    """
    # Initialize the queue for inter-thread communication.
    command_queue = queue.Queue()
    
    # Create and start the background data processing thread as a daemon.
    data_thread = threading.Thread(target=data_processing_thread, args=(command_queue,), daemon=True)
    data_thread.start()

    print("Main Program: Initialized. Waiting for 'start' (990-2) command...")
    
    # Initialize state variables
    is_started = False           # Whether the system has received the start command
    current_morse_code = ""      # Accumulator for the Morse code of the current character

    try:
        while True:
            # Get a command from the queue. This call will block if the queue is empty.
            command = command_queue.get()

            # Ignore all commands until the 'start' command is received.
            if not is_started:
                if command == '990-2':
                    is_started = True
                    print("\n=== Main Program: 'start' command received. System is now active. ===")
                else:
                    print(f"(Awaiting start) Main Program ignored command: {command}")
                continue

            # --- Core Translation Logic ---
            if command == '500-2':    # Command represents a Morse dot (·)
                current_morse_code += '.'
            elif command == '500-1':  # Command represents a Morse dash (-)
                current_morse_code += '-'
            elif command == '800-2':  # Command represents the end of a character
                if current_morse_code:
                    # Translate the accumulated code, defaulting to '?' if not found.
                    letter = MORSE_CODE_DICT.get(current_morse_code, '?')
                    print(f"Translation Output: {letter} (from code: {current_morse_code})")
                    print("-" * 30)
                    # Reset the accumulator for the next character.
                    current_morse_code = ""
            elif command == '1200-2': # Command represents program exit
                print("=== Main Program: 'exit' command received. Shutting down. ===")
                break
            elif command == '990-2':  # Ignore redundant start commands
                print("Main Program Info: System is already active. Ignoring redundant 'start' command.")
            else:
                # Ignore any other unknown commands
                print(f"Main Program Info: Unknown command '{command}' received and ignored.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down...")
    finally:
        print("Main Program has shut down.")

if __name__ == '__main__':
    main()
