"""
GRU-based CPU Usage Predictor
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CPUUsagePredictor:
    """Simple GRU predictor for CPU usage"""
    
    def __init__(self, sequence_length=24, prediction_horizon=12):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def build_model(self):
        """Build GRU architecture"""
        model = keras.Sequential([
            layers.GRU(64, return_sequences=True, input_shape=(self.sequence_length, 1)),
            layers.Dropout(0.2),
            layers.GRU(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def prepare_sequences(self, data):
        """Create training sequences"""
        # Scale data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(data_scaled) - self.sequence_length - self.prediction_horizon):
            X.append(data_scaled[i:i + self.sequence_length])
            y.append(data_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)
        
        return X, y
    
    def train(self, train_data, epochs=30, batch_size=32, validation_split=0.2):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X_train, y_train = self.prepare_sequences(train_data)
        
        print(f"Training on {len(X_train)} sequences...")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def predict(self, sequence):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Scale sequence
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1)).flatten()
        X = sequence_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        # Predict
        prediction_scaled = self.model.predict(X, verbose=0)[0]
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()
        
        # Calculate confidence (simple version)
        confidence = 1.0 / (1.0 + np.std(prediction))
        
        return prediction, confidence
    
    def evaluate(self, test_data):
        """Evaluate model"""
        X_test, y_test = self.prepare_sequences(test_data)
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        y_pred_inv = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        
        # Metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        
        return {'mse': mse, 'mae': mae, 'rmse': rmse}

