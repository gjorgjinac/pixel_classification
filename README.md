# Pixel classification

## Overview

This repository provides a framework for training and evaluating machine learning models
for classifying satellite image pixels. It includes scripts for data preprocessing, splitting data into cross-validation
folds, training models, and analyzing results. The workflow is designed to handle classification problems with optional
label binarization and feature importance analysis.

## Features

- **Data Preprocessing**: Handles numeric and categorical feature preprocessing.
- **Cross-Validation**: Splits data into 10 folds using `GroupKFold` based on product IDs.
- **Model Training**: Trains classification models on each fold and evaluates their performance.
- **Result Analysis**: Aggregates results across folds and visualizes metrics.

## Repository Structure

### Exploratory Data Analysis (EDA) scripts
- `EDA1_analyze_class_distribution.py`: Analyzes the distribution of features/classes in the dataset.
- `EDA2_analyze_missing_values.py`: Analyzes missing values

### Main model training scripts
- `download_data.sh`: Script to download the dataset.
- `R1_split_folds.py`: Splits the dataset into folds for cross-validation.
- `R2_train_model_single_fold.py`: Trains and evaluates models for a specific fold. Expects as arguments --fold, --use_mlflow, --binarize_labels
  - `--fold`: Specifies the fold number to train on (0-9).
  - `--use_mlflow`: If set to non-zero value, logs training metrics to MLflow.
  - `--binarize_labels`: If set to non-zero value, binarizes labels for binary classification tasks.
  - `--do_resampling`: Expects string values 'none'/'over'/'under', indicating the type of resampling to use. 'none' means no resampling, 'under' means to use random undersampling, 'over' means to use oversampling.
- `run_experiments.sh`: Runs models for all folds, all combinations of binarize_labels and resampling options (works on linux and requires task spooler for task scheduling)
- `R3_analyze_results.py`: Aggregates and visualizes results across folds.

### Utility Files
- `utils/utils_visualization.py`: Contains utility functions for EDA visualizations
- `utils/utils_model_training.py`: Contains utility functions for preprocessing, training, and evaluation.
- `utils/utils_read_data.py`: Reads data into a pandas DataFrame.
- `globals.py`: Defines global variables used across scripts.

## Installation
### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Create a python environment. We use python 3.12
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install required packages:
   ```bash
    pip install -r requirements.txt
    ```
4. Download the dataset:
   ```bash
   wget -O "20170710_s2_manual_classification_data.h5" "https://git.gfz-potsdam.de/EnMAP/sentinel2_manual_classification_clouds/-/raw/master/20170710_s2_manual_classification_data.h5"
    ```
5. Run the data splitting script:
   ```bash
   python R1_split_folds.py
   ```
6. Train models for a single fold:
   ```bash
    python R2_train_model_single_fold.py --fold 0 --use_mlflow 0 --binarize_labels 1 --do_resampling none
    ```
7. Run models for all folds, all combinations of binarize_labels and resampling options (works on linux and requires task spooler for task scheduling)
```bash
./run_experiments.sh
```
7. Analyze results across all folds:
```bash
   python R3_analyze_results.py
   ```