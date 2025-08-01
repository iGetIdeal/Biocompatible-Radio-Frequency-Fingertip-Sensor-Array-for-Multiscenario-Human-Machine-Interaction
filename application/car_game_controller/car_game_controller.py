# =============================================================================
#   BP-NET REAL-TIME VIRTUAL CAR CONTROLLER
#   Description:
#   This script runs a virtual car simulation using Pygame, where the car's
#   movements are controlled in real-time by a pre-trained BP neural network.
#
#   A background thread monitors a folder for new .xlsx data files, processes
#   them to extract features, and feeds them into the neural network. The
#   model's predicted command is then used to move the car on the screen.
#
#   Dependencies: pygame, torch, pandas, scikit-learn, joblib
#
# =============================================================================

import os
import time
import threading
import pygame
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np

# --- Configuration: File Paths and Constants ---
# Note: These paths should be configured by the user.
# Using os.path.join for cross-platform compatibility.
DATA_FOLDER_PATH = os.path.join('D:', 'path', 'to', 'your', 'data', 'folder') # Path to the folder containing VNA output .xlsx files
MODEL_PATH = "../models/bp_net_model.pth"                                             # Path to the trained BP Neural Network model
SCALER_PATH = "../models/scaler.joblib"                                               # Path to the fitted StandardScaler object
LABEL_ENCODER_PATH = "../models/label_encoder.joblib"                                 # Path to the fitted LabelEncoder object

# Asset paths for the game visuals
ASSET_DIR = "./application/car_game_controller/asset"
BACKGROUND_IMAGE_PATH = os.path.join(ASSET_DIR, "background.jpg")
CAR_IMAGE_PATH = os.path.join(ASSET_DIR, "car_red.png")

# Screen dimensions for the Pygame window
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# Custom Pygame event for model-driven commands
CUSTOM_COMMAND_EVENT = pygame.USEREVENT + 1


# --- Neural Network Model Definition ---

class BPNet(nn.Module):
    """
    A simple Backpropagation Neural Network (BPNet) with two hidden layers.
    This network takes 2 input features and predicts one of the predefined classes.
    """
    def __init__(self, num_classes):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)       # Input layer to first hidden layer
        self.fc2 = nn.Linear(16, 16)      # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(16, num_classes) # Second hidden layer to output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        """Defines the forward pass of the neural network."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- Data Processing and Prediction Functions ---

def find_latest_xlsx_file(folder_path):
    """
    Finds the most recently modified .xlsx file in a specified directory.

    Args:
        folder_path (str): The path to the directory to search.

    Returns:
        str or None: The full path to the latest .xlsx file, or None if no such file is found.
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
    Extracts key features from a given VNA data file (.xlsx).
    It reads the file, filters for a specific frequency range, and finds the minimum return loss.

    Args:
        file_path (str): The path to the .xlsx file.

    Returns:
        tuple or None: A tuple containing (frequency, return_loss) if successful, otherwise None.
    """
    print(f"Processing file: {os.path.basename(file_path)}")
    try:
        # Read the Excel file, skipping the original header row.
        df = pd.read_excel(file_path, header=1)
        df = df.iloc[1:].reset_index(drop=True)

        if 'Frequency (Hz)' not in df.columns or 'Returnloss (dB)' not in df.columns:
            print("Error: The file is missing required columns: 'Frequency (Hz)' or 'Returnloss (dB)'.")
            return None

        # Filter the DataFrame for frequencies between 400 MHz and 700 MHz.
        df_filtered = df[(df['Frequency (Hz)'] > 400 * 10**6) & (df['Frequency (Hz)'] < 700 * 10**6)]
        if df_filtered.empty:
            print("No valid data found in the 400-700 MHz frequency range.")
            return None
        
        # Find the row with the minimum return loss (most negative value).
        min_returnloss_row = df_filtered.loc[df_filtered['Returnloss (dB)'].idxmin()]
        min_loss = min_returnloss_row['Returnloss (dB)']
        corresponding_freq = min_returnloss_row['Frequency (Hz)']
        
        print(f"Features extracted successfully: Frequency = {corresponding_freq / 1e6:.2f} MHz, Return Loss = {min_loss:.2f} dB")
        return (corresponding_freq, min_loss)

    except Exception as e:
        print(f"An error occurred while reading or processing the Excel file: {e}")
        return None

def predict_command(features_tuple, model, scaler, label_encoder):
    """
    Uses the loaded BPNet model to predict a control command from the extracted features.

    Args:
        features_tuple (tuple): A tuple containing the features (frequency, return_loss).
        model (BPNet): The trained PyTorch model.
        scaler (StandardScaler): The fitted scaler for feature normalization.
        label_encoder (LabelEncoder): The fitted encoder for decoding the prediction.

    Returns:
        str or None: The predicted command string if successful, otherwise None.
    """
    try:
        # Prepare features for the model: scale and convert to a tensor.
        features_np = np.array([list(features_tuple)])
        scaled_features = scaler.transform(features_np)
        input_tensor = torch.FloatTensor(scaled_features)

        # Perform inference.
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output.data, 1)

        # Decode the predicted index back to the original command label.
        predicted_command = label_encoder.inverse_transform(predicted_idx.numpy())[0]
        print(f"Model predicted command: {predicted_command}")
        return predicted_command

    except Exception as e:
        print(f"An error occurred during model prediction: {e}")
        return None


# --- Pygame Simulation Classes and Functions ---

class Car:
    """
    Represents the player-controlled car in the simulation.
    Manages the car's state (position) and handles movement commands.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.move_step = 10 # Movement speed in pixels per command

    def handle_command(self, command):
        """
        Updates the car's position based on a given command string.
        The command rules are based on the trained model's output labels.
        """
        if command == '500-2':    # Up
            self.y -= self.move_step
        elif command == '1000-2': # Down
            self.y += self.move_step
        elif command == '800-2':  # Left
            self.x -= self.move_step
        elif command == '1200-2': # Right
            self.x += self.move_step
        
        # Enforce screen boundaries to keep the car visible.
        self.x = max(0, min(SCREEN_WIDTH - 60, self.x)) # 60 is car width
        self.y = max(0, min(SCREEN_HEIGHT - 35, self.y)) # 35 is car height

    def draw(self, surface, image):
        """Draws the car sprite onto the specified surface."""
        surface.blit(image, (self.x, self.y))

def data_processing_thread():
    """
    A worker thread that continuously monitors the data folder for new files,
    processes them, and posts the resulting command to the Pygame event queue.
    """
    try:
        # Load the machine learning model and pre-processing objects.
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        num_classes = len(label_encoder.classes_)
        model = BPNet(num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Set the model to evaluation mode.
        print("BP model and transformers loaded successfully.")
    except FileNotFoundError as e:
        print(f"Fatal Error: Could not find model or transformer files: {e}")
        print("The data processing thread will now exit.")
        return

    last_processed_file = None
    print("\n--- Starting to monitor the data folder ---")
    while True:
        latest_file = find_latest_xlsx_file(DATA_FOLDER_PATH)
        if latest_file and latest_file != last_processed_file:
            print(f"New file detected: {os.path.basename(latest_file)}")
            features = extract_features_from_excel(latest_file)
            if features:
                command = predict_command(features, model, scaler, label_encoder)
                if command is not None:
                    # Post the command as a custom event to the main Pygame thread.
                    event = pygame.event.Event(CUSTOM_COMMAND_EVENT, {'command': command})
                    pygame.event.post(event)
            last_processed_file = latest_file
        time.sleep(1) # Wait for 1 second before checking for new files again.


# --- Main Application Entry Point ---

def main():
    """
    Initializes Pygame, sets up the game window, and runs the main game loop.
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BPNet Real-time Virtual Car Control")
    clock = pygame.time.Clock()

    # Load and scale game assets.
    try:
        background_img = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
        background_img = pygame.transform.scale(background_img, (SCREEN_WIDTH, SCREEN_HEIGHT))
        car_img = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
        car_img = pygame.transform.scale(car_img, (60, 35))
    except pygame.error as e:
        print(f"Fatal Error: Failed to load game assets: {e}")
        print("Please ensure the 'asset' folder and its contents are in the correct location.")
        return

    # Create the car object and start the data processing thread.
    car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    
    # Start the data processing thread as a daemon so it exits when the main program does.
    data_thread = threading.Thread(target=data_processing_thread, daemon=True)
    data_thread.start()

    running = True
    while running:
        # Process all events in the Pygame event queue.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle the custom command event posted by the data thread.
            elif event.type == CUSTOM_COMMAND_EVENT:
                command = event.dict['command']
                car.handle_command(command)

        # Drawing operations.
        screen.blit(background_img, (0, 0))
        car.draw(screen, car_img)

        # Update the display.
        pygame.display.flip()
        
        # Limit the frame rate to 60 FPS.
        clock.tick(60)

    pygame.quit()
    print("Pygame simulation has been shut down.")

if __name__ == '__main__':
    main()

