"""
Real AWS Data Collection and Prediction
"""

import boto3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from models.gru_predictor import CPUUsagePredictor

print("="*60)
print("VM CPU PREDICTOR - Real AWS Data")
print("="*60)

# Set your region
REGION = 'us-east-1'  # CHANGE THIS to your region!

print("\n1. Connecting to AWS...")
try:
    ec2 = boto3.client('ec2', region_name=REGION)
    cloudwatch = boto3.client('cloudwatch', region_name=REGION)
    
    # Test connection
    ec2.describe_instances(MaxResults=5)
    print(f"✓ Connected to AWS region: {REGION}")
    
except Exception as e:
    print(f"✗ Error connecting to AWS: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure IAM role is attached to this EC2")
    print("  2. Or run: aws configure")
    exit(1)

# Get running instances
print("\n2. Finding EC2 instances...")
try:
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
                'type': instance['InstanceType'],
                'name': name,
                'launch_time': instance['LaunchTime']
            })
    
    if not instances:
        print("✗ No running instances found!")
        print("  Make sure you have EC2 instances running in this region")
        exit(1)
    
    print(f"✓ Found {len(instances)} running instance(s):")
    for i, inst in enumerate(instances, 1):
        print(f"  {i}. {inst['id']} - {inst['name']} ({inst['type']})")
        
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Process each instance
print("\n" + "="*60)
print("ANALYZING INSTANCES")
print("="*60)

results_summary = []

for idx, instance in enumerate(instances, 1):
    instance_id = instance['id']
    instance_name = instance['name']
    
    print(f"\n[{idx}/{len(instances)}] Instance: {instance_id} ({instance_name})")
    print("-" * 60)
    
    # Collect metrics
    print("Collecting CPU metrics from CloudWatch...")
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    try:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5 minutes
            Statistics=['Average']
        )
        
        datapoints = response['Datapoints']
        
        if not datapoints:
            print("⚠ No CloudWatch data available yet")
            print("  Reasons:")
            print("  - Instance just launched (wait 10-15 minutes)")
            print("  - Detailed monitoring not enabled")
            print("  - Wrong region selected")
            results_summary.append({
                'instance': instance_id,
                'status': 'NO DATA',
                'reason': 'CloudWatch has no metrics yet'
            })
            continue
        
        # Sort and extract data
        datapoints.sort(key=lambda x: x['Timestamp'])
        cpu_values = np.array([d['Average'] for d in datapoints])
        timestamps = [d['Timestamp'] for d in datapoints]
        
        print(f"✓ Collected {len(cpu_values)} data points")
        print(f"  Time span: {(timestamps[-1] - timestamps[0]).total_seconds()/3600:.1f} hours")
        print(f"  Mean CPU: {cpu_values.mean():.2f}%")
        print(f"  Max CPU: {cpu_values.max():.2f}%")
        print(f"  Min CPU: {cpu_values.min():.2f}%")
        
        # Check if enough data
        if len(cpu_values) < 40:
            print(f"\n⚠ Only {len(cpu_values)} data points (need 40+)")
            print("  Please wait a few more hours and try again")
            results_summary.append({
                'instance': instance_id,
                'status': 'INSUFFICIENT DATA',
                'datapoints': len(cpu_values)
            })
            continue
        
        # Train model
        print("\nTraining GRU model on this instance's data...")
        
        predictor = CPUUsagePredictor(
            sequence_length=min(24, len(cpu_values)//3),
            prediction_horizon=12
        )
        
        # Train with less epochs for speed
        history = predictor.train(
            train_data=cpu_values,
            epochs=15,
            batch_size=16,
            validation_split=0.2
        )
        
        final_loss = history.history['loss'][-1]
        print(f"✓ Training complete! (Final loss: {final_loss:.4f})")
        
        # Evaluate
        metrics = predictor.evaluate(cpu_values)
        print(f"  Model MAE: {metrics['mae']:.2f}%")
        
        # Predict
        print("\nMaking predictions...")
        recent = cpu_values[-min(24, len(cpu_values)//2):]
        predictions, confidence = predictor.predict(recent)
        
        print(f"  Recent average: {recent.mean():.2f}%")
        print(f"  Predicted average: {predictions.mean():.2f}%")
        print(f"  Confidence: {confidence:.3f}")
        
        # Idle detection
        is_idle = (predictions.mean() < 10) and (recent.mean() < 10) and confidence > 0.7
        
        print("\n" + "="*40)
        if is_idle:
            print("🔴 IDLE VM DETECTED!")
            print("Recommendation: Consider shutdown")
            status = "IDLE"
        else:
            print("✅ ACTIVE VM")
            print("Recommendation: Keep running")
            status = "ACTIVE"
        print("="*40)
        
        # Save data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_utilization': cpu_values
        })
        filename = f'data/{instance_id}_metrics.csv'
        df.to_csv(filename, index=False)
        
        results_summary.append({
            'instance': f"{instance_id} ({instance_name})",
            'status': status,
            'recent_cpu': f"{recent.mean():.2f}%",
            'predicted_cpu': f"{predictions.mean():.2f}%",
            'confidence': f"{confidence:.3f}",
            'mae': f"{metrics['mae']:.2f}%",
            'datapoints': len(cpu_values)
        })
        
    except Exception as e:
        print(f"✗ Error processing instance: {e}")
        results_summary.append({
            'instance': instance_id,
            'status': 'ERROR',
            'error': str(e)
        })
        continue

# Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

if results_summary:
    print("\nInstance Analysis Results:")
    print("-" * 60)
    for result in results_summary:
        print(f"\n{result['instance']}")
        for key, value in result.items():
            if key != 'instance':
                print(f"  {key}: {value}")
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('data/analysis_summary.csv', index=False)
    print(f"\n✓ Summary saved to: data/analysis_summary.csv")
else:
    print("\nNo instances were analyzed successfully.")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)

