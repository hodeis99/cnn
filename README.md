# CNN for Backorder Prediction - Supply Chain Forecasting

A state-of-the-art 1D CNN model for predicting backorders in supply chain management, designed to capture local temporal patterns in inventory data sequences.

## Key Features

- **Temporal CNN Architecture**: 3-layer 1D CNN with MaxPooling and BatchNorm
- **Advanced Sequence Handling**: Converts tabular data to 10-step time sequences
- **Class Imbalance Solution**: Custom class weighting (2.2:1) and AUC optimization
- **Robust Training**: Gradient clipping, progressive dropout, and early stopping
- **Full Evaluation Suite**: Precision, Recall, AUC, and bilingual reports

## Dataset Requirements

Preprocessed CSV files from the data cleaning pipeline:
- `Train_Preprocess.csv` (balanced dataset)
- `Test_Preprocess.csv` (original distribution)

## Installation

```bash
git clone https://github.com/hodeis99/cnn.git
cd cnn
pip install -r requirements.txt
