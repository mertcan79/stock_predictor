import os
import yaml
import matplotlib.pyplot as plt
from typing import Tuple, Any, Dict

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from lstm_model import LSTMModel

def main(config_path: str = None) -> Tuple[LSTMModel, Dict[str, Any], Dict[str, float]]:
    """
    Main function to load data, engineer features, train, and evaluate the LSTM model.
    
    Returns:
        - final_model: The trained LSTMModel instance.
        - cv_metrics: Cross-validation metrics.
        - test_metrics: Evaluation metrics on the test set.
    """
    # Build an absolute path if none is provided
    if config_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "..", "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Instantiate loader and feature engineer
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)

    df = data_loader.load_data()
    train_df, val_df, test_df = data_loader.prepare_train_val_test_split(df)
    
    # Create features for each split
    train_features = feature_engineer.create_features(train_df, is_training=True)
    val_features = feature_engineer.create_features(val_df, is_training=False)
    test_features = feature_engineer.create_features(test_df, is_training=False)
    
    # Perform RFE on training features
    X_train_df = train_features.drop("target", axis=1)
    y_train = train_features["target"]
    selected_feats, rankings = feature_engineer.perform_rfe(
        X_train_df, y_train, n_features_to_select=12
    )
    
    # Retain only selected features and target in all splits
    cols_to_keep = selected_feats + ["target"]
    train_features = train_features[cols_to_keep].copy()
    val_features = val_features[cols_to_keep].copy()
    test_features = test_features[cols_to_keep].copy()
    
    # Create sequences for modeling
    X_train, y_train = feature_engineer.create_sequences(train_features)
    X_val, y_val = feature_engineer.create_sequences(val_features)
    X_test, y_test = feature_engineer.create_sequences(test_features)
    
    input_shape = (config["data"]["sequence_length"], X_train.shape[2])
    
    # Time series cross-validation evaluation
    model_instance = LSTMModel(config)
    cv_metrics = model_instance.time_series_cv_evaluation(
        X_train, y_train, feature_engineer.target_scaler, n_splits=3
    )
    
    # Build, train, and evaluate final model on test set
    final_model = LSTMModel(config)
    final_model.build_model(input_shape)
    final_model.train((X_train, y_train), (X_val, y_val))
    test_metrics = final_model.evaluate(
        X_test, y_test, target_scaler=feature_engineer.target_scaler
    )
    
    # Save predicted vs. actual plot
    preds = final_model.model.predict(X_test, verbose=0)
    y_pred = feature_engineer.target_scaler.inverse_transform(preds.reshape(-1, 1))
    y_true = feature_engineer.target_scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.figure(figsize=(8, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Test Set: Actual vs Predicted")
    plt.xlabel("Sample")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("results/test_predictions.png")
    plt.close()
    
    return final_model, cv_metrics, test_metrics

if __name__ == "__main__":
    main()
