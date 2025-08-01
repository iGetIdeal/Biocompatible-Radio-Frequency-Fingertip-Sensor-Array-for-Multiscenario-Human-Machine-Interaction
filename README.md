# Biocompatible-Radio-Frequency-Fingertip-Sensor-Array-for-Multiscenario-Human-Machine-Interaction

This repository serves as the official code appendix for the paper **"Biocompatible Radio Frequency Fingertip Sensor Array for Multi-scenario Human-Machine Interaction"**. It contains the scripts for training the Backpropagation (BP) neural network model, the dataset used, the pre-trained model files, and several proof-of-concept applications demonstrating the model's utility.

## Project Structure

The repository is organized to separate data, models, source code, and applications, ensuring clarity and reproducibility.

```
.
├── 📂 data/
│   └── 📄 training_set.xlsx
│
├── 📂 models/
│   ├── 📄 bp_net_model.pth
│   ├── 📄 label_encoder.joblib
│   └── 📄 scaler.joblib
│
├── 📂 src/
│   └── 📄 train_model.py
│
├── 📂 applications/
│   ├── 📂 car_game_controller/
│   │   ├── 📄 car_game_controller.py
│   │   └── 📂 assets/
│   │       ├── 📄 background.jpg
│   │       └── 📄 car_red.png
│   │
│   ├── 📂 morse_code_translator/
│   │   └── 📄 morse_code_translator.py
│   │
│   └── 📂 robotic_arm_controller/
│       └── 📄 robotic_arm_controller.py
│
└── 📄 README.md
```

- **`/data`**: Contains the dataset (`training_set.xlsx`) used for training and evaluating the model.

- `/models`

  : Contains the pre-trained model files. These are the outputs of the training script.

  - `bp_net_model.pth`: The saved state dictionary of the trained PyTorch neural network.
  - `scaler.joblib`: The saved `StandardScaler` object for normalizing input features.
  - `label_encoder.joblib`: The saved `LabelEncoder` object for handling class labels.

- `/src`

  : Contains the core source code for model training.

  - `train_model.py`: The script to train the BP neural network from scratch using the data in `/data` and save the resulting artifacts into `/models`.

- **`/applications`**: Contains standalone Python scripts that load the pre-trained models from `/models` to perform specific tasks. Each sub-folder represents a distinct application.

------

## Requirements

**Python Version:** This project requires **Python 3.10 or newer**. This is because the `pygame` library, a core dependency for the `car_game_controller` application, requires a modern version of Python.

**Dependencies:** You will also need the following Python libraries. The `pyserial` package is required for the `robotic_arm_controller`.

You can install all required dependencies using a single `pip` command:

```bash
pip install torch pandas scikit-learn joblib numpy pygame pyserial openpyxl
```

It is recommended to create a virtual environment to manage these dependencies.

------

## Usage Guide

This project allows for both re-training the model and running applications with the provided pre-trained models.

### 1. Re-training the Model

If you wish to re-train the model from scratch, you can run the training script.

**Note:** Running this script will overwrite the existing files in the `/models` directory.

From the project root directory, execute the following command:

```bash
python src/train_model.py
```

The script will load `data/training_set.xlsx`, train the model, and save the new model, scaler, and encoder to the `/models` directory.

### 2. Running the Applications

The pre-trained models are located in the `/models` directory, allowing you to run the applications directly. Each application script is self-contained and loads the necessary model files.

From the project root directory, run the desired application using one of the following commands:

- **Car Game Controller:**

  ```bash
  python applications/car_game_controller/car_game_controller.py
  ```

- **Morse Code Translator:**

  ```bash
  python applications/morse_code_translator/morse_code_translator.py
  ```

- **Robotic Arm Controller:**

  ```bash
  python applications/robotic_arm_controller/robotic_arm_controller.py
  ```

------

## Citation

This code is the official appendix for our paper, which is currently under review. If you find this code or our methodology useful for your research, we kindly ask that you cite our work upon its publication.

**Paper Title:** Biocompatible Radio Frequency Fingertip Sensor Array for Multi-scenario Human-Machine Interaction
**Author(s):** Zhiqing Gao, Hao Wen, Yibing Qiu, Jiawei Jin, Feiyang Huang, Rujing Sun, Xin Li, Zijian An, and Qingjun Liu

A full, formal citation will be added here as soon as the paper is accepted and published.