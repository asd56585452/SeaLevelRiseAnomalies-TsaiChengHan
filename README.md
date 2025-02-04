# SeaLevelRiseAnomalies-TsaiChengHan

This repository contains the final training code and trained models for the competition.

## Directory Structure

```
.
├── optuna_train.py         # Structure search and hyperparameter tuning using Optuna
├── SLAV8-K=5.ipynb         # Final training script with 5-fold cross-validation
├── sla_dataset.py          # CPU version of SLA dataset loader
├── sla_dataset_GPU.py      # GPU-accelerated version of SLA dataset loader
├── README.md               # This README file
```

## File Descriptions

### `optuna_train.py`
- Uses [Optuna](https://optuna.org/) for structure search and hyperparameter optimization.
- Searches for optimal CNN architecture, including convolutional layer channels, dropout rates, learning rates, weight decay, etc.
- Loads data using `sla_dataset_GPU.py` and trains on a GPU.
- Uses `BCEWithLogitsLoss` as the loss function and evaluates models based on the `F1-score (macro)`.
- Stores training progress in `sqlite:///optuna_study.db` for tracking the best parameters.

### `SLAV8-K=5.ipynb`
- This is the final training script utilizing 5-fold cross-validation to ensure model generalization.
- Loads data using `sla_dataset.py` or `sla_dataset_GPU.py`.
- During training, each fold uses different dataset splits for training and testing.
- The optimization strategy is the same as in `optuna_train.py`, but with a fixed model structure and hyperparameters.

### `sla_dataset.py` & `sla_dataset_GPU.py`
- These files are used to load the Sea Level Anomalies (SLA) dataset:
  - `sla_dataset.py`: CPU-compatible version, slower but works on all systems.
  - `sla_dataset_GPU.py`: Loads data directly onto the GPU for faster training.
- The dataset consists of `.nc` (NetCDF) files containing SLA data and corresponding anomaly labels (`.csv`).
- `sla_dataset_GPU.py` preloads data onto the GPU to speed up training.

## Usage

### 1. Run Structure Search
```bash
python optuna_train.py
```
This will start the Optuna optimization process and store the best results in `sqlite:///optuna_study.db`.

### 2. Perform 5-Fold Cross-Validation
Open `SLAV8-K=5.ipynb` in Jupyter Notebook and run the cells sequentially to perform full training.

## Environment Requirements
- Python 3.8+
- PyTorch
- Optuna
- NumPy, Pandas, Xarray
- Matplotlib (optional, for data visualization)

## Contact
Author: **Tsai Cheng-Han**  
For any questions, please contact the author or submit an issue on GitHub.
