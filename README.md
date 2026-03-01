# sip_model_class
This repository contains a deep learning model for classifying respiratory diseases (Asthma, Bronchiectasis, COPD) based on Volatile Organic Compound (VOC) profiles. The model is built with TensorFlow/Keras and is trained on a combination of real and synthetically generated data.

An automatic versioning system is implemented. Each time the training script is run, it increments the model version and saves the corresponding artifacts (weights, info, performance plot) into their respective directories.

## Project Structure

For the scripts to function correctly, the repository must be placed within a specific project structure. The `sip_model_class` directory should be a sibling to `sip_data` (containing the real dataset) and `sip_model_gen` (containing the synthetic dataset).
Refer to [sip_model_gen](https://github.com/seongjunkang123/sip_model_gen) and [sip_data](https://github.com/seongjunkang123/sip_data). 

```
.
├── sip_data/
│   └── res1/
│       ├── combined_data.csv
│       └── combined_synthetic_and_real.csv (Generated)
├── sip_model_class/  <-- This repository
└── sip_model_gen/
    └── synthetic_data/
        └── synthetic_data_*.csv
```

## Installation

1.  Clone the repository and arrange your project directories as shown in the "Project Structure" section.

2.  Navigate to the `sip_model_class` directory:
    ```bash
    cd sip_model_class
    ```
3.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Follow these steps to prepare the data and train the model.

### 1. Combine Datasets

This script merges synthetic data from `../sip_model_gen/synthetic_data/` with real data from `../sip_data/res1/combined_data.csv`. The output is saved to `../sip_data/res1/combined_synthetic_and_real.csv`, which is required for training.

```bash
python combine_synthetic_and_real.py
```

### 2. Train the Model

The main script loads the combined data, builds the classification model, and starts the training process. It automatically versions the model and its outputs.

```bash
python main.py
```

This script will:
*   Determine the next model version (e.g., `v1`, `v2`).
*   Save the model architecture and parameters to `model_info/mod_v<version>.txt`.
*   Train the model and save the best weights to `model_weights/mod_v<version>.keras`.
*   Save the training history to `model_performance/mod_v<version>.pkl`.
*   Generate and save a loss plot to `model_performance/mod_v<version>.png`.

### 3. View Training History

To visualize the training loss for a previously trained model, run this script. It will prompt you to enter the model's version number.

```bash
python history.py
```
When prompted, enter the version number (e.g., `1`). This will load the data from the corresponding `.pkl` file and display the loss plot.

## Docker

You can build and run the training process within a Docker container.

**Note:** The Docker build context must be the parent directory containing `sip_model_class`, `sip_data`, and `sip_model_gen` to ensure the container has access to all necessary data.

1.  From the parent project directory, build the Docker image:
    ```bash
    docker build -t sip-model-class -f sip_model_class/Dockerfile .
    ```

2.  Run the container. This will execute the `main.py` script to train the model. The output artifacts will be saved inside the container.
    ```bash
    docker run --rm sip-model-class
    ```

## File & Directory Descriptions

*   `main.py`: The main script for building, training, and saving the classification model.
*   `combine_synthetic_and_real.py`: A utility script to merge real and synthetically generated datasets.
*   `history.py`: A script to visualize the training loss of a saved model.
*   `utils.py`: Contains a helper function for automatic model versioning.
*   `requirements.txt`: A list of Python libraries required for the project.
*   `Dockerfile`: Defines the Docker container for running the training script.
*   `model_weights/`: Stores trained model weights (`.keras` files).
*   `model_info/`: Contains text files with model architecture summaries and hyperparameters.
*   `model_performance/`: Stores serialized training history (`.pkl` files) and loss plots (`.png` files).