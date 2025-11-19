"""
GRU Model Comparison for VM CPU Prediction
Compares: Standard GRU, CNN-GRU, BiGRU, Parallel GRU
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("="*70)
print("VM CPU PREDICTOR - MODEL COMPARISON")
print("="*70)

# Generate synthetic data
print("\n1. Generating synthetic CPU usage data...")
np.random.seed(42)
time_steps = 500

# Create realistic CPU pattern with daily cycles
t = np.linspace(0, 10, time_steps)
cpu_data = (
    50 +  # Base load
    20 * np.sin(t * 2 * np.pi) +  # Daily cycle
    10 * np.sin(t * 8 * np.pi) +  # High-frequency pattern
    np.random.normal(0, 5, time_steps)  # Noise
)
cpu_data = np.clip(cpu_data, 0, 100)

print(f"✓ Generated {len(cpu_data)} data points")
print(f"  Mean CPU: {cpu_data.mean():.2f}%")


# Data preparation
class DataPreparator:
    def __init__(self, sequence_length=24, prediction_horizon=12):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_sequences(self, data):
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(data_scaled) - self.sequence_length - self.prediction_horizon):
            X.append(data_scaled[i:i + self.sequence_length])
            y.append(data_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)
        
        return X, y


# Model 1: Standard GRU (Your current model)
class StandardGRU:
    def __init__(self, seq_len=24, pred_horizon=12):
        self.name = "Standard GRU"
        self.model = models.Sequential([
            layers.GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
            layers.Dropout(0.2),
            layers.GRU(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(pred_horizon)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def get_params(self):
        return self.model.count_params()


# Model 2: CNN-GRU Hybrid
class CNNGRU:
    def __init__(self, seq_len=24, pred_horizon=12):
        self.name = "CNN-GRU Hybrid"
        self.model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_len, 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.GRU(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(pred_horizon)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def get_params(self):
        return self.model.count_params()


# Model 3: Bidirectional GRU
class BidirectionalGRU:
    def __init__(self, seq_len=24, pred_horizon=12):
        self.name = "Bidirectional GRU"
        self.model = models.Sequential([
            layers.Bidirectional(layers.GRU(64, return_sequences=True), input_shape=(seq_len, 1)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.GRU(32, return_sequences=False)),
            layers.Dropout(0.2),
            layers.Dense(pred_horizon)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def get_params(self):
        return self.model.count_params()


# Model 4: Parallel GRU (Multi-scale)
class ParallelGRU:
    def __init__(self, seq_len=24, pred_horizon=12):
        self.name = "Parallel GRU (Multi-scale)"
        
        # Input
        input_layer = layers.Input(shape=(seq_len, 1))
        
        # Three parallel GRU branches
        gru1 = layers.GRU(32)(input_layer)
        
        input_short = layers.Lambda(lambda x: x[:, -12:, :])(input_layer)
        gru2 = layers.GRU(32)(input_short)
        
        input_long = input_layer
        gru3 = layers.GRU(32)(input_long)
        
        # Concatenate
        concat = layers.Concatenate()([gru1, gru2, gru3])
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu')(concat)
        dropout = layers.Dropout(0.2)(dense1)
        output = layers.Dense(pred_horizon)(dropout)
        
        self.model = models.Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
    def get_params(self):
        return self.model.count_params()


# Prepare data
print("\n2. Preparing training data...")
prep = DataPreparator(sequence_length=24, prediction_horizon=12)
X, y = prep.prepare_sequences(cpu_data)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")


# Train and evaluate all models
print("\n" + "="*70)
print("TRAINING AND EVALUATING MODELS")
print("="*70)

models_list = [
    StandardGRU(),
    CNNGRU(),
    BidirectionalGRU(),
    ParallelGRU()
]

results = []

for model_wrapper in models_list:
    print(f"\n{'='*70}")
    print(f"Model: {model_wrapper.name}")
    print(f"{'='*70}")
    
    # Model info
    params = model_wrapper.get_params()
    print(f"Parameters: {params:,}")
    
    # Training
    print("\nTraining...")
    start_time = time.time()
    
    history = model_wrapper.model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    training_time = time.time() - start_time
    print(f"✓ Training time: {training_time:.2f} seconds")
    
    # Prediction
    print("\nEvaluating...")
    start_time = time.time()
    y_pred = model_wrapper.model.predict(X_test, verbose=0)
    prediction_time = (time.time() - start_time) / len(X_test) * 1000
    
    # Inverse transform for metrics
    y_test_inv = prep.scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    y_pred_inv = prep.scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-10))) * 100
    
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"✓ MAE: {mae:.2f}%")
    print(f"✓ RMSE: {rmse:.2f}%")
    print(f"✓ MAPE: {mape:.2f}%")
    print(f"✓ Prediction time: {prediction_time:.2f} ms/sample")
    
    results.append({
        'Model': model_wrapper.name,
        'Parameters': params,
        'Training Time (s)': round(training_time, 2),
        'Prediction Time (ms)': round(prediction_time, 2),
        'MAE (%)': round(mae, 2),
        'RMSE (%)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'Final Loss': round(final_loss, 6),
        'Final Val Loss': round(final_val_loss, 6)
    })


# Results summary
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# Save results
df_results.to_csv('reports/model_comparison_results.csv', index=False)
print("\n✓ Results saved to: reports/model_comparison_results.csv")

# Best model analysis
best_mae_idx = df_results['MAE (%)'].idxmin()
best_speed_idx = df_results['Prediction Time (ms)'].idxmin()
best_mae = df_results.iloc[best_mae_idx]
best_speed = df_results.iloc[best_speed_idx]

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"\n🏆 Best Accuracy: {best_mae['Model']}")
print(f"   MAE: {best_mae['MAE (%)']}%")
print(f"   RMSE: {best_mae['RMSE (%)']}%")

print(f"\n⚡ Fastest Prediction: {best_speed['Model']}")
print(f"   Speed: {best_speed['Prediction Time (ms)']} ms")

print(f"\n💡 For Production Deployment:")
if best_mae['Model'] == best_speed['Model']:
    print(f"   Recommended: {best_mae['Model']}")
    print(f"   Reason: Best accuracy AND speed")
else:
    print(f"   If accuracy is priority: {best_mae['Model']}")
    print(f"   If speed is priority: {best_speed['Model']}")

# Performance improvements
standard_mae = df_results[df_results['Model'] == 'Standard GRU']['MAE (%)'].values[0]
print(f"\n📊 Improvements over Standard GRU:")
for _, row in df_results.iterrows():
    if row['Model'] != 'Standard GRU':
        improvement = ((standard_mae - row['MAE (%)']) / standard_mae) * 100
        symbol = "↓" if improvement > 0 else "↑"
        print(f"   {row['Model']}: {abs(improvement):.1f}% {symbol}")

print("\n" + "="*70)
print("✅ Model comparison complete!")
print("="*70)

