from loguru import logger
from sklearn.feature_selection import RFE
import ta
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import yfinance as yf

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config["data"]["sequence_length"]
        self.scalers = {}
        self.target_scaler = None
        logger.add("logs/feature_engineering.log", rotation="500 MB", enqueue=True)
        self.logger = logger

    def create_features(self, df: pd.DataFrame, is_training: bool=False) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("Date", inplace=True)
        
        # Target definition
        df["target"] = df["Close"].shift(-1)
        
        # Technical indicators
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
        acc_dist = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
        df["acc_dist"] = acc_dist
        df["mfi"] = ta.volume.MFIIndicator(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]).money_flow_index()
        df["cci"] = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()
        
        # Volume trend: volume momentum relative to 20-day MA
        df["vol_ma_20"] = df["Volume"].rolling(window=20).mean()
        df["vol_mom"] = df["Volume"] / df["vol_ma_20"] - 1
        
        symbol = self.config["data"]["stock_symbol"]
        eps = yf.Ticker(symbol).info.get("trailingEps", np.nan)
        div_yield = yf.Ticker(symbol).info.get("dividendYield", np.nan)
        df["eps"] = eps
        df["dividend_yield"] = div_yield
        
        df.dropna(inplace=True)
        # Select features – include essential ones
        selected_features = [
            "Close_lag_1", "rsi", "macd", "macd_signal", "bollinger_high", "bollinger_low",
            "atr", "adx", "obv", "stoch_k", "stoch_d", "acc_dist", "mfi", "cci",
            "vol_mom", "eps", "dividend_yield", "target"
        ]
        df = df[selected_features]
        
        if is_training:
            self._fit_scalers(df)
        df = self._scale_features(df)
        self.logger.info(f"Feature engineering complete. New shape: {df.shape}")
        return df

    def _fit_scalers(self, df: pd.DataFrame):
        for col in df.columns:
            if col != "target":
                scaler = RobustScaler()
                scaler.fit(df[[col]])
                self.scalers[col] = scaler
        self.target_scaler = RobustScaler()
        self.target_scaler.fit(df[["target"]])
        
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, scaler in self.scalers.items():
            df[col] = scaler.transform(df[[col]])
        if self.target_scaler is not None:
            df["target"] = self.target_scaler.transform(df[["target"]].values)
        return df

    def create_sequences(self, df: pd.DataFrame):
        X, y = [], []
        features = [col for col in df.columns if col != "target"]
        for i in range(len(df) - self.sequence_length):
            seq = df[features].iloc[i: i + self.sequence_length].values
            target_value = df["target"].iloc[i + self.sequence_length]
            X.append(seq)
            y.append(target_value)
        self.logger.info(f"Generated {len(X)} sequences from data shape {df.shape}")
        return np.array(X), np.array(y)
    
    def perform_rfe(self, X_df: pd.DataFrame, y, essential_features, n_features_to_select=12):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(model, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_df, y)
        selected = X_df.columns[rfe.support_].tolist()
        for feat in essential_features:
            if feat not in selected:
                selected.append(feat)
        return selected, rfe.ranking_