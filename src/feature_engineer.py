from loguru import logger
from sklearn.feature_selection import RFE
import ta
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import yfinance as yf
from typing import Dict, Tuple, List, Any

class FeatureEngineer:
    """
    Performs feature engineering including technical indicators, scaling, and sequence generation.
    """
    def __init__(self, config: Dict[str, Any]) -> None:

        self.config = config
        self.sequence_length: int = config["data"]["sequence_length"]
        self.scalers: Dict[str, RobustScaler] = {}
        self.target_scaler: RobustScaler | None = None
        logger.add("logs/feature_engineering.log", rotation="500 MB", enqueue=True)
        self.logger = logger

    def create_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Create technical indicator features, define the target, and scale the data.
        
        Returns dataframe with engineered features.
        """
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("Date", inplace=True) 
        
        df["target"] = df["Close"].shift(-1)  # Next day's closing price
        
        # Compute technical indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["bollinger_high"] = bb.bollinger_hband()
        df["bollinger_low"] = bb.bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["acc_dist"] = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
        df["mfi"] = ta.volume.MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]).money_flow_index()
        df["cci"] = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()
        
        # Volume trend
        df["vol_ma_20"] = df["Volume"].rolling(window=20).mean()
        df["vol_mom"] = df["Volume"] / df["vol_ma_20"] - 1
        
        # Fetch fundamental data
        symbol = self.config["data"]["stock_symbol"]
        eps = yf.Ticker(symbol).info.get("trailingEps", np.nan)
        div_yield = yf.Ticker(symbol).info.get("dividendYield", np.nan)
        df["eps"] = eps
        df["dividend_yield"] = div_yield
        
        df.dropna(inplace=True)
        
        if is_training:
            self._fit_scalers(df)
        df = self._scale_features(df)
        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df

    def _fit_scalers(self, df: pd.DataFrame) -> None:
        """
        Fit a RobustScaler for each feature and for the target.
        """
        for col in df.columns:
            if col != "target":
                scaler = RobustScaler()
                scaler.fit(df[[col]])
                self.scalers[col] = scaler
        self.target_scaler = RobustScaler()
        self.target_scaler.fit(df[["target"]])

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scalers to the data.
        
        Returns scaled DataFrame.
        """
        for col, scaler in self.scalers.items():
            df[col] = scaler.transform(df[[col]])
        if self.target_scaler is not None:
            df["target"] = self.target_scaler.transform(df[["target"]].values)
        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sequences and corresponding targets for time series modeling.
        
        Returns arrays of sequences and target
        """
        X, y = [], []
        features = [col for col in df.columns if col != "target"]
        for i in range(len(df) - self.sequence_length):
            seq = df[features].iloc[i: i + self.sequence_length].values
            target_value = df["target"].iloc[i + self.sequence_length]
            X.append(seq)
            y.append(target_value)
        self.logger.info(f"Generated {len(X)} sequences from data shape {df.shape}")
        return np.array(X), np.array(y)
    
    def perform_rfe(
        self,
        X_df: pd.DataFrame,
        y: Any,
        n_features_to_select: int = 12
    ) -> Tuple[List[str], np.ndarray]:
        """
        Perform recursive feature elimination and ensure essential features are retained.
        
        Returns selected feature names and their rankings.
        """
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(model, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_df, y)
        selected = X_df.columns[rfe.support_].tolist()
        self.logger.info(f"Selected features after RFE:{selected}")
        return selected, rfe.ranking_
