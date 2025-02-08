import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Tuple

class DataLoader:
    """
    Handles downloading, processing, and splitting stock data.
    """
    def __init__(self, config: Dict):
        self.config = config
        logger.add("logs/data_loader.log", rotation="500 MB", enqueue=True)
        self.logger = logger

    def load_data(self) -> pd.DataFrame:
        """
        Downloads stock data, fills missing values, and creates lag features.

        Returns processed data with additional lag features.
        """
        start_date = datetime.now() - timedelta(days=self.config["data"]["history_days"])
        end_date = datetime.now()
        symbol = self.config["data"]["stock_symbol"]
        
        self.logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        df = yf.download(symbol, start=start_date, end=end_date)
        
        # Fill missing values forward and backward
        df = df.ffill().bfill()
        
        # If columns are multi-indexed, reduce to a single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        
        # Create lag features for the 'Close' column
        for lag in [1, 2, 3, 5, 7, 10, 14]:
            df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        
        df.dropna(inplace=True)
        self.logger.info(f"Data loaded successfully with shape {df.shape}")
        return df

    def prepare_train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data into training, validation, and test sets, ensuring continuity.

        Returns train, validation, and test dfs.
        """
        seq_len = self.config["data"]["sequence_length"]
        n = len(df)
        train_idx = int(n * self.config["data"]["train_size"])
        val_idx = train_idx + int(n * self.config["data"]["val_size"])
        
        # Training data start to train_idx
        train_df = df.iloc[:train_idx].copy()
        # include last sequence length rows from training for continuity
        val_df = df.iloc[train_idx - seq_len:val_idx].copy()
        # include last sequence length rows from validation to preserve sequence
        test_df = df.iloc[val_idx - seq_len:].copy()
        
        self.logger.info(f"Data split: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
        return train_df, val_df, test_df
