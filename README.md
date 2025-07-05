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


### Main model training scripts
- `download_data.sh`: Script to download the dataset.
- `R1_split_folds.py`: Splits the dataset into folds for cross-validation.
- `R2_train_model_single_fold.py`: Trains and evaluates models for a specific fold.
- `R3_analyze_results.py`: Aggregates and visualizes results across folds.

### Utility Files

- `utils_model_training.py`: Contains utility functions for preprocessing, training, and evaluation.
- `utils_read_data.py`: Reads data into a pandas DataFrame.
- `globals.py`: Defines global variables used across scripts.

## Installation

### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install required packages:
   ```bash
    pip install -r requirements.txt
    ```
3. Download the dataset:
   ```bash
   ./download_data.sh
    ```
4. Run the data splitting script:
   ```bash
   python R1_split_folds.py
   ```
5. Train models for a specific fold:
   ```bash
    python R2_train_model_single_fold.py --fold 0
    ```
6. Analyze results across all folds:
```bash
   python R3_analyze_results.py
   ```