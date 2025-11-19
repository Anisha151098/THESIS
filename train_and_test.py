"""
Train and Test VM Predictor with Synthetic Data
"""

import numpy as np
import matplotlib.pyplot as plt
from models.gru_predictor import CPUUsagePredictor

print("="*60)
print("VM CPU PREDICTOR - Training with Synthetic Data")
print("="*60)

# Generate synthetic CPU data
print("\n1. Generating synthetic CPU usage data...")
np.random.seed(42)

# Create 500 data points (about 2 days of 5-min intervals)
time_steps = 500

# Pattern 1: High utilization VM (70-90% CPU)
high_cpu = 80 + 10 * np.sin(np.linspace(0, 8*np.pi, time_steps)) + np.random.normal(0, 5, time_steps)
high_cpu = np.clip(high_cpu, 0, 100)

# Pattern 2: Idle VM (5-15% CPU)
idle_cpu = 10 + 3 * np.sin(np.linspace(0, 4*np.pi, time_steps)) + np.random.normal(0, 2, time_steps)
idle_cpu = np.clip(idle_cpu, 0, 100)

# Pattern 3: Variable VM (transitions from high to low)
variable_cpu = np.concatenate([
    np.ones(200) * 75 + np.random.normal(0, 8, 200),  # High usage
    np.ones(100) * 40 + np.random.normal(0, 10, 100), # Medium
    np.ones(200) * 10 + np.random.normal(0, 3, 200)   # Low (idle)
])
variable_cpu = np.clip(variable_cpu, 0, 100)

print(f"✓ Generated 3 VM patterns with {time_steps} data points each")

# Train models for each pattern
patterns = {
    'High CPU VM': high_cpu,
    'Idle VM': idle_cpu,
    'Variable VM (becomes idle)': variable_cpu
}

results = {}

for vm_name, cpu_data in patterns.items():
    print(f"\n{'='*60}")
    print(f"2. Training model for: {vm_name}")
    print(f"{'='*60}")
    
    # Split into train/test
    train_size = int(len(cpu_data) * 0.8)
    train_data = cpu_data[:train_size]
    test_data = cpu_data[train_size:]
    
    print(f"   Training samples: {train_size}")
    print(f"   Test samples: {len(test_data)}")
    print(f"   Mean CPU: {cpu_data.mean():.2f}%")
    
    # Create and train model
    predictor = CPUUsagePredictor(sequence_length=24, prediction_horizon=12)
    
    print("\n   Training GRU model...")
    history = predictor.train(
        train_data=train_data,
        epochs=30,
        batch_size=32,
        validation_split=0.2
    )
    
    print(f"   ✓ Training complete!")
    print(f"   Final loss: {history.history['loss'][-1]:.4f}")
    
    # Evaluate
    print("\n3. Evaluating model...")
    metrics = predictor.evaluate(test_data)
    print(f"   MAE: {metrics['mae']:.2f}%")
    print(f"   RMSE: {metrics['rmse']:.2f}%")
    
    # Make prediction on recent sequence
    recent_sequence = cpu_data[-24:]
    predictions, confidence = predictor.predict(recent_sequence)
    
    print("\n4. Making predictions...")
    print(f"   Recent CPU average: {recent_sequence.mean():.2f}%")
    print(f"   Predicted CPU average: {predictions.mean():.2f}%")
    print(f"   Prediction confidence: {confidence:.3f}")
    
    # Determine if idle
    is_idle = (predictions.mean() < 10) and (recent_sequence.mean() < 10)
    print(f"\n   {'🔴 IDLE VM DETECTED!' if is_idle else '✅ ACTIVE VM'}")
    
    results[vm_name] = {
        'data': cpu_data,
        'recent': recent_sequence,
        'predictions': predictions,
        'confidence': confidence,
        'is_idle': is_idle,
        'metrics': metrics
    }

# Summary
print("\n" + "="*60)
print("SUMMARY - IDLE VM DETECTION")
print("="*60)

for vm_name, result in results.items():
    status = "🔴 IDLE" if result['is_idle'] else "✅ ACTIVE"
    print(f"{vm_name:30s} {status:10s} (confidence: {result['confidence']:.2f})")

print("\n✓ Training and testing completed successfully!")
print("\nNext steps:")
print("  - Review the results above")
print("  - Model is ready for real AWS data")
print("  - Run 'python train_and_test.py' again to re-train")
