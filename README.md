# MNIST Classification with MLflow Tracking

This repository contains a machine learning workflow for digit recognition (MNIST) using Random Forest. It demonstrates how to use MLflow for tracking experiments, managing hyperparameters via YAML, and organizing pipelines into nested stages.

## Project Structure

- `config.yaml`: Centralized configuration file for data paths and model hyperparameters.
- `experiment_flat.py`: Script for comparing different hyperparameters in a flat structure (ideal for side-by-side comparison).
- `experiment_nested.py`: Script implementing a hierarchical logging structure, dividing the workflow into data preparation, training, and evaluation stages.

## Key Features

- **Centralized Configuration**: All model settings (n_estimators, max_depth) and data parameters are managed through a single YAML file.
- **Dual Tracking Approaches**: 
    - **Flat**: Best for hyperparameter tuning.
    - **Nested**: Best for production-like pipeline monitoring.
- **Automated Caching**: The dataset is downloaded once and cached locally to speed up subsequent runs.
- **Rich Artifacts**: Both scripts automatically log trained models and confusion matrix plots to the MLflow server.

## Getting Started

### 1. Requirements
Ensure you have the necessary libraries installed:
```bash
pip install mlflow scikit-learn joblib pyyaml matplotlib
```
### 2. Launch the MLflow Tracking Server
Start the local server using SQLite as the backend store:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```
### 3. Execute Experiments
Run the Python scripts to train the models and log data to the server:
```bash
python experiment_flat.py
python experiment_nested.py
```
To change model parameters, edit the config.yaml file.

## Access the UI
Open your browser and navigate to http://127.0.0.1:5000 to visualize the results, compare runs, and download artifacts.
