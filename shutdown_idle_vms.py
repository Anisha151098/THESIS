"""
Shutdown Idle VMs - IMMEDIATE MODE (No Confirmation)
"""

import boto3
from datetime import datetime, timedelta
import numpy as np
from models.gru_predictor import CPUUsagePredictor

print("="*60)
print("SHUTDOWN IDLE VMS - IMMEDIATE MODE")
print("="*60)

REGION = 'us-east-1'
PREDICTOR_INSTANCE = 'i-06439ca9baab008dd'  # GRUMODEL - Protected

print(f"\n🛡️  Protected instance: {PREDICTOR_INSTANCE} (GRUMODEL)")

# Connect to AWS
print("\n1. Connecting to AWS...")
ec2 = boto3.client('ec2', region_name=REGION)
cloudwatch = boto3.client('cloudwatch', region_name=REGION)
print("✓ Connected")

# Get all running instances
print("\n2. Finding all running instances...")
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
            'name': name,
            'type': instance['InstanceType']
        })

# Filter out predictor
instances_to_analyze = [i for i in instances if i['id'] != PREDICTOR_INSTANCE]

print(f"✓ Found {len(instances_to_analyze)} instances to analyze")

# Analyze and shutdown
print("\n" + "="*60)
print("ANALYZING AND SHUTTING DOWN IDLE INSTANCES")
print("="*60)

idle_instances = []
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=24)

for instance in instances_to_analyze:
    instance_id = instance['id']
    instance_name = instance['name']
    
    print(f"\n[{instance_id}] {instance_name}")
    
    try:
        # Get metrics
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
        
        if not datapoints or len(datapoints) < 30:
            print(f"  ⚠ Insufficient data - SKIP")
            continue
        
        datapoints.sort(key=lambda x: x['Timestamp'])
        cpu_values = np.array([d['Average'] for d in datapoints])
        
        mean_cpu = cpu_values.mean()
        print(f"  Mean CPU: {mean_cpu:.2f}%")
        
        # Check if idle
        is_idle = (mean_cpu < 10) and ((cpu_values < 15).mean() > 0.85)
        
        if is_idle and len(cpu_values) >= 50:
            # ML Confirmation
            predictor = CPUUsagePredictor(
                sequence_length=min(20, len(cpu_values)//3),
                prediction_horizon=12
            )
            predictor.train(cpu_values, epochs=5)
            
            recent = cpu_values[-20:]
            predictions, confidence = predictor.predict(recent)
            
            print(f"  Predicted: {predictions.mean():.2f}% (confidence: {confidence:.3f})")
            
            if predictions.mean() < 10 and confidence > 0.7:
                print(f"  🔴 IDLE CONFIRMED - SHUTTING DOWN...")
                
                # SHUTDOWN IMMEDIATELY
                try:
                    ec2.stop_instances(InstanceIds=[instance_id])
                    print(f"  ✅ STOPPED")
                    idle_instances.append({
                        'id': instance_id,
                        'name': instance_name,
                        'type': instance['type'],
                        'mean_cpu': mean_cpu,
                        'predicted_cpu': predictions.mean(),
                        'confidence': confidence
                    })
                except Exception as e:
                    print(f"  ✗ Failed to stop: {e}")
            else:
                print(f"  ✅ ACTIVE - Keep running")
        elif is_idle:
            # Rule-based (not enough data for ML)
            print(f"  🔴 IDLE (rule-based) - SHUTTING DOWN...")
            try:
                ec2.stop_instances(InstanceIds=[instance_id])
                print(f"  ✅ STOPPED")
                idle_instances.append({
                    'id': instance_id,
                    'name': instance_name,
                    'type': instance['type'],
                    'mean_cpu': mean_cpu,
                    'predicted_cpu': mean_cpu,
                    'confidence': 0.85
                })
            except Exception as e:
                print(f"  ✗ Failed to stop: {e}")
        else:
            print(f"  ✅ ACTIVE - Keep running")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

# Summary
print("\n" + "="*60)
print("SHUTDOWN COMPLETE")
print("="*60)

if idle_instances:
    print(f"\n✅ Successfully stopped {len(idle_instances)} idle instance(s):")
    for inst in idle_instances:
        print(f"  • {inst['name']} ({inst['id']})")
        print(f"    CPU: {inst['mean_cpu']:.2f}% | Confidence: {inst['confidence']:.1%}")
    
    # Calculate savings
    monthly_savings = len(idle_instances) * 67.74
    annual_savings = monthly_savings * 12
    
    print(f"\n💰 Cost Savings:")
    print(f"   Monthly: ${monthly_savings:.2f}")
    print(f"   Annual: ${annual_savings:.2f}")
else:
    print("\n✅ No idle instances detected")

print("\n" + "="*60)

