import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger
plt.style.use('Solarize_Light2')

class LSTMModel:
    """
    LSTM model for time series forecasting.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        logger.add("logs/model.log", rotation="500 MB", enqueue=True)
        self.logger = logger

    def build_model(self, input_shape: tuple) -> Model:
        inputs = Input(shape=input_shape)
        x = LSTM(self.config["model"]["lstm"]["units"][0], return_sequences=False, 
                 kernel_initializer='glorot_uniform')(inputs)
        outputs = Dense(1, activation="linear")(x)
        self.model = Model(inputs, outputs)
        optimizer = Adam(learning_rate=self.config["model"]["training"]["learning_rate"])
        self.model.compile(optimizer=optimizer, 
                           loss=self.config["model"]["optimization"]["loss_function"],
                           metrics=["mae"])
        self.logger.info("LSTM model built successfully.")
        return self.model

    def train(self, train_data: tuple, val_data: tuple = None) -> dict:
        callbacks = [
            EarlyStopping(monitor="val_loss", 
                          patience=self.config["model"]["training"]["early_stopping"]["patience"],
                          restore_best_weights=True, verbose=0),
            ModelCheckpoint("models/best_model.keras", monitor="val_loss", save_best_only=True, verbose=0)
        ]
        history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=self.config["model"]["training"]["epochs"],
            batch_size=self.config["model"]["training"]["batch_size"],
            callbacks=callbacks,
            shuffle=False,
            verbose=0
        )
        self.logger.info("Training complete.")
        return history.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, target_scaler) -> dict:
        preds = self.model.predict(X_test, verbose=0)
        y_pred = target_scaler.inverse_transform(preds.reshape(-1, 1))
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def time_series_cv_evaluation(self, X, y, target_scaler, n_splits=3):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_list = []
        fold = 1
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model_instance = LSTMModel(self.config)
            model_instance.build_model(input_shape=X_tr.shape[1:])
            history = model_instance.train((X_tr, y_tr), (X_val, y_val))
            m = model_instance.evaluate(X_val, y_val, target_scaler)
            metrics_list.append(m)
            if fold == 1:
                plt.figure(figsize=(8, 4))
                plt.plot(history['loss'], label="Train Loss")
                plt.plot(history['val_loss'], label="Validation Loss")
                plt.title("Learning Curve - Fold 1")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig("results/learning_curve_fold1.png")
                plt.close()
            fold += 1
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        self.logger.info(f"imeSeries CV Average Metrics:{avg_metrics}")
        return avg_metrics