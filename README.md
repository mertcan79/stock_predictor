# Machine Learning Engineer Assignment

## Overview

This project contains a stock price prediction model using LSTM network. The model analyzes historical stock data, extracts meaningful features, and predicts future prices. In notebooks folder, a jupyter notebook contains EDA and modeling. In src folder, main.py can be executed to run the model based on the config.yaml file. The logs, visuals and the model file can be found in their folders. The presentation PDF file shows the steps taken and the modeling approach.

## Installation

Clone the repository
```bash
git clone <repo-url>
cd <repo-folder>
```
Create and activate a virtual environment
```bash
python -m venv env
source env/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration 

Modify config.yaml to adjust parameters for data loading, feature engineering, and model training.

To run

```bash
python src/main.py
```


