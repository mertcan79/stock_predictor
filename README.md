# Machine Learning Engineer Assignment

## Overview

This project contains a stock price prediction model using an LSTM model. 
The model loads historical stock data, creates and extracts features, and predicts future prices. 
A jupyter notebook in the notebooks folder contains EDA and modeling steps. 
In src folder, main.py can be executed to run the model and output results based on the config.yaml file. 
The logs, visuals and the model file can be found in their folders. 
The presentation PDF file shows the steps taken and the modeling approach.

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

## To run

```bash
python src/main.py
```


