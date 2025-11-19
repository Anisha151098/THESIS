"""
Quick Test with Current Data
Works with 30+ data points
"""

import boto3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from models.gru_predictor import CPUUsagePredictor

print("="*60)
print("VM CPU PREDICTOR - Quick Test Mode")
print("="*60)

REGION = 'us-east-1'

print("\n1. Connecting to AWS...")
ec2 = boto3.client('ec2', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
print(f"✓ Connected to AWS")

print("\n2. Finding EC2 instances...")
response = ec2.describe_instances(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)

instances = []
for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        name = 'Unknown'
        for tag in instance.get('Tags', []):
            if tag['Key'] == 'Name':
                name = tag['Value']
        instances.append({
            'id': instance['InstanceId'],
            'name': name
        })

print(f"✓ Found {len(instances)} instance(s)")

for instance in instances:
    instance_id = instance['id']
    instance_name = instance['name']
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {instance_id} ({instance_name})")
    print('='*60)
    
    # Collect metrics
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Average']
    )
    
    datapoints = response['Datapoints']
    
    if not datapoints:
        print("⚠ No data available")
        continue
    
    datapoints.sort(key=lambda x: x['Timestamp'])
    cpu_values = np.array([d['Average'] for d in datapoints])
    
    print(f"\nData Statistics:")
    print(f"  Data points: {len(cpu_values)}")
    print(f"  Mean CPU: {cpu_values.mean():.2f}%")
    print(f"  Max CPU: {cpu_values.max():.2f}%")
    print(f"  Min CPU: {cpu_values.min():.2f}%")
    print(f"  Std Dev: {cpu_values.std():.2f}%")
    
    # Check if we have minimum data
    if len(cpu_values) < 30:
        print(f"\n⚠ Only {len(cpu_values)} data points")
        print("  Need at least 30 points")
        continue
    
    # Simple idle detection (no ML for very small data)
    if len(cpu_values) < 50:
        print("\n📊 Using Simple Rule-Based Detection:")
        print("    (Not enough data for ML model)")
        
        is_consistently_low = (cpu_values < 15).mean() > 0.9
        mean_is_low = cpu_values.mean() < 10
        max_is_low = cpu_values.max() < 20
        
        print(f"\n  ✓ {(cpu_values < 15).mean()*100:.1f}% of time below 15% CPU")
        print(f"  ✓ Mean CPU: {cpu_values.mean():.2f}%")
        print(f"  ✓ Max CPU: {cpu_values.max():.2f}%")
        
        if is_consistently_low and mean_is_low:
            print("\n" + "="*60)
            print("🔴 IDLE VM DETECTED!")
            print("="*60)
            print("This instance shows very low utilization:")
            print(f"  - Average: {cpu_values.mean():.2f}%")
            print(f"  - 90%+ of time below 15% CPU")
            print("\n💡 Recommendation: CANDIDATE FOR SHUTDOWN")
            print("   This instance could be stopped to save costs")
        else:
            print("\n" + "="*60)
            print("✅ ACTIVE VM")
            print("="*60)
            print("Instance shows normal activity")
        
        continue
    
    # Train ML model (if enough data)
    print("\n🧠 Training ML Model...")
    
    predictor = CPUUsagePredictor(
        sequence_length=min(20, len(cpu_values)//3),
        prediction_horizon=12
    )
    
    predictor.train(
        train_data=cpu_values,
        epochs=10,
        batch_size=8,
        validation_split=0.2
    )
    
    print("✓ Training complete!")
    
    # Predict
    recent = cpu_values[-20:]
    predictions, confidence = predictor.predict(recent)
    
    print(f"\nPrediction Results:")
    print(f"  Recent avg: {recent.mean():.2f}%")
    print(f"  Predicted avg: {predictions.mean():.2f}%")
    print(f"  Confidence: {confidence:.3f}")
    
    is_idle = (predictions.mean() < 10) and (recent.mean() < 10)
    
    if is_idle:
        print("\n🔴 IDLE VM DETECTED!")
        print("💡 Recommendation: Consider shutdown")
    else:
        print("\n✅ ACTIVE VM")
        print("💡 Recommendation: Keep running")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)

