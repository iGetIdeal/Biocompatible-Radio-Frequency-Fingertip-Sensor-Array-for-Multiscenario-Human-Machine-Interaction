# Biocompatible-Radio-Frequency-Fingertip-Sensor-Array-for-Multiscenario-Human-Machine-Interaction

This repository serves as the official code appendix for the paper **"Biocompatible Radio Frequency Fingertip Sensor Array for Multi-scenario Human-Machine Interaction"**. It contains the scripts for training the Backpropagation (BP) neural network model, the dataset used, the pre-trained model files, and several proof-of-concept applications demonstrating the model's utility.

## Project Structure

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
│       ├── 📄 robotic_arm_controller.py
│       └── 📂 stm32_keil_project/
│           ├── 📂 CORE/
│           ├── 📂 STM32F10x_FWLib/
│           ├── 📂 USER/
│           ├── 📄 OpenArmSTM32.uvoptx
│           └── 📄 OpenArmSTM32.uvprojx
│
├── 📄 README.md
└── 📄 requirements.txt
```

## Modules Description

### data/

Contains the dataset used for training the neural network.

- `training_set.xlsx`: The raw data for model training.

### models/

Stores the pre-trained machine learning model and associated data processors.

- `bp_net_model.pth`: The trained PyTorch model weights.
- `scaler.joblib`: The fitted StandardScaler for normalizing input data.
- `label_encoder.joblib`: The fitted LabelEncoder for converting between numerical predictions and class labels.

###  src/

Source code for model training.

- `train_model.py`: The Python script used to train the BP neural network and save the model files.

### applications/

Contains the standalone applications that utilize the trained model.

#### car_game_controller/

A virtual car simulation powered by the BP network.

- `car_game_controller.py`: The main script that reads data, predicts commands, and controls the car in a Pygame window.
- `assets/`: Contains image files for the game.

#### Morse Code Translator

A simple command-line tool for Morse code.

- `morse_code_translator.py`: The script for translation logic.

#### robotic_arm_controller/

A dual-component system for controlling a physical robotic arm.

- `robotic_arm_controller.py`: The Python host script that runs on a PC. It handles model inference and sends control commands via serial communication.
- `stm32_keil_project/`: The embedded firmware for the STM32 microcontroller. This Keil MDK-ARM v5 project receives commands from the host and actuates the arm's motors.

------

## Installation and Usage

### 1. Prerequisites

- Python 3.10+
- Keil MDK-ARM (v5 or later) for the embedded project.

### 2. Setup

Clone the repository and install the required Python packages:

```
git clone https://your-repository-url/
cd your-repository-name/
pip install -r requirements.txt
```

### 3. Model Training (Optional)

The repository includes pre-trained models. However, if you wish to retrain the model on your own data, run the training script:

```
python src/train_model.py
```

### 4. Running the Applications

**Virtual Car Controller:**

```
python applications/car_game_controller/car_game_controller.py
```

**Morse Code Translator:**

```
python applications/morse_code_translator/morse_code_translator.py
```

**Robotic Arm Controller:**

1. **Flash Firmware**: Open the `applications/robotic_arm_controller/stm32_keil_project/OpenArmSTM32.uvprojx` project in Keil MDK, compile it, and flash the firmware to your STM32 hardware.

2. Run Host Script

   : Connect the hardware to your PC and run the host controller script. You may need to configure the serial port within the script.

   ```
   python applications/robotic_arm_controller/robotic_arm_controller.py
   ```

------

## Citation

This code is the official appendix for our paper, which is currently under review. If you find this code or our methodology useful for your research, we kindly ask that you cite our work upon its publication.

**Paper Title:** Biocompatible Radio Frequency Fingertip Sensor Array for Multi-scenario Human-Machine Interaction
**Author(s):** Zhiqing Gao, Hao Wen, Yibing Qiu, Jiawei Jin, Feiyang Huang, Rujing Sun, Xin Li, Zijian An, and Qingjun Liu

A full, formal citation will be added here as soon as the paper is accepted and published.