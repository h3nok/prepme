# AWS Services for ML & AI

## 🎯 Core AWS ML Services

### Amazon SageMaker
**Primary ML Platform on AWS**

#### SageMaker Components
```python
# SageMaker Training Job Example
import boto3
import sagemaker
from sagemaker.estimator import Estimator

def create_training_job():
    """Create a SageMaker training job for large model training."""
    
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define estimator for distributed training
    estimator = Estimator(
        image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker',
        role=role,
        instance_count=8,  # Multi-node training
        instance_type='ml.p4d.24xlarge',  # High-memory GPU instances
        volume_size=1000,  # Large EBS volume for data
        max_run=7*24*3600,  # 7 days max runtime
        
        # Distributed training configuration
        distribution={
            'torch_distributed': {
                'enabled': True
            }
        },
        
        # Environment variables
        environment={
            'NCCL_DEBUG': 'INFO',
            'NCCL_TREE_THRESHOLD': '0',
        },
        
        # Hyperparameters
        hyperparameters={
            'epochs': 10,
            'learning_rate': 5e-5,
            'batch_size': 32,
            'model_name': 'llama-7b',
            'gradient_accumulation_steps': 4
        }
    )
    
    # Start training
    estimator.fit({
        'training': 's3://my-bucket/training-data/',
        'validation': 's3://my-bucket/validation-data/'
    })
    
    return estimator
```

#### SageMaker Inference
```python
# Real-time inference endpoint
def deploy_model_endpoint(estimator):
    """Deploy trained model to real-time endpoint."""
    
    predictor = estimator.deploy(
        initial_instance_count=2,
        instance_type='ml.g5.2xlarge',
        
        # Auto-scaling configuration
        auto_scaling_target_value=70.0,  # Target CPU utilization
        auto_scaling_min_capacity=1,
        auto_scaling_max_capacity=10,
        
        # Model data configuration
        model_data_download_timeout=1800,
        container_startup_health_check_timeout=600
    )
    
    return predictor

# Batch inference for large datasets
def create_batch_transform_job():
    """Create batch transform job for inference on large datasets."""
    
    transformer = estimator.transformer(
        instance_count=5,
        instance_type='ml.m5.2xlarge',
        output_path='s3://my-bucket/batch-inference-output/',
        
        # Batch strategy
        strategy='MultiRecord',
        max_payload=6,  # MB
        max_concurrent_transforms=10
    )
    
    transformer.transform(
        data='s3://my-bucket/batch-inference-input/',
        content_type='application/json',
        split_type='Line'
    )
    
    return transformer
```

### Amazon Bedrock
**Managed Generative AI Service**

#### Bedrock Usage Patterns
```python
import boto3
import json

class BedrockClient:
    def __init__(self, region='us-east-1'):
        self.bedrock = boto3.client('bedrock-runtime', region_name=region)
        self.model_configs = {
            'claude-3': {
                'model_id': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'max_tokens': 4096
            },
            'titan': {
                'model_id': 'amazon.titan-text-express-v1',
                'max_tokens': 8192
            }
        }
    
    def generate_text(self, prompt, model='claude-3', **kwargs):
        """Generate text using Bedrock models."""
        
        config = self.model_configs[model]
        
        if model == 'claude-3':
            body = {
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': kwargs.get('max_tokens', config['max_tokens']),
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            }
        elif model == 'titan':
            body = {
                'inputText': prompt,
                'textGenerationConfig': {
                    'maxTokenCount': kwargs.get('max_tokens', config['max_tokens']),
                    'temperature': kwargs.get('temperature', 0.7),
                    'topP': kwargs.get('top_p', 0.9)
                }
            }
        
        response = self.bedrock.invoke_model(
            modelId=config['model_id'],
            body=json.dumps(body)
        )
        
        return self._parse_response(response, model)
    
    def _parse_response(self, response, model):
        """Parse response based on model type."""
        response_body = json.loads(response['body'].read())
        
        if model == 'claude-3':
            return response_body['content'][0]['text']
        elif model == 'titan':
            return response_body['results'][0]['outputText']

# Bedrock for RAG applications
class BedrockRAG:
    def __init__(self):
        self.bedrock_client = BedrockClient()
        self.vector_store = self._setup_vector_store()
    
    def answer_question(self, question, context_docs=None):
        """Answer question using RAG with Bedrock."""
        
        if context_docs is None:
            # Retrieve relevant documents
            context_docs = self.vector_store.similarity_search(question, k=5)
        
        # Create RAG prompt
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        rag_prompt = f"""
        Context information:
        {context}
        
        Question: {question}
        
        Please answer the question based on the provided context. If the answer 
        is not in the context, please say so.
        
        Answer:
        """
        
        return self.bedrock_client.generate_text(rag_prompt, model='claude-3')
```

## 🗄️ Data & Storage Services

### Amazon S3 for ML Data
```python
import boto3
from concurrent.futures import ThreadPoolExecutor
import os

class S3DataManager:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
    
    def upload_dataset(self, local_path, s3_prefix, parallel_uploads=10):
        """Upload large datasets to S3 efficiently."""
        
        def upload_file(file_info):
            local_file, s3_key = file_info
            self.s3.upload_file(local_file, self.bucket_name, s3_key)
            return s3_key
        
        # Collect all files to upload
        upload_tasks = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                upload_tasks.append((local_file, s3_key))
        
        # Parallel upload
        with ThreadPoolExecutor(max_workers=parallel_uploads) as executor:
            results = list(executor.map(upload_file, upload_tasks))
        
        return results
    
    def create_data_versioning(self, dataset_name, version):
        """Create versioned dataset structure."""
        
        version_prefix = f"datasets/{dataset_name}/v{version}/"
        
        # Create metadata
        metadata = {
            'dataset_name': dataset_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'schema': self._extract_schema(),
            'statistics': self._compute_statistics()
        }
        
        # Upload metadata
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=f"{version_prefix}metadata.json",
            Body=json.dumps(metadata),
            ContentType='application/json'
        )
        
        return version_prefix
```

### Amazon EFS for Shared Model Storage
```python
# EFS configuration for shared model storage across instances
def setup_efs_for_distributed_training():
    """Setup EFS for shared model weights and checkpoints."""
    
    efs_config = {
        'mount_target': '/opt/ml/model',
        'file_system_id': 'fs-0123456789abcdef0',
        'performance_mode': 'generalPurpose',  # or 'maxIO' for high throughput
        'throughput_mode': 'bursting',  # or 'provisioned'
        
        # For large model training
        'provisioned_throughput': 500  # MiB/s if using provisioned mode
    }
    
    # Mount script for training instances
    mount_script = f"""
    sudo mount -t efs -o tls {efs_config['file_system_id']}:/ {efs_config['mount_target']}
    sudo chmod 777 {efs_config['mount_target']}
    """
    
    return efs_config, mount_script
```

## 🚀 Compute Services

### Amazon EC2 for Custom ML Workloads
```python
# EC2 instance configurations for different ML workloads
ML_INSTANCE_CONFIGS = {
    'training_large_models': {
        'instance_type': 'p4d.24xlarge',
        'gpu_count': 8,  # A100 GPUs
        'memory_gb': 1152,
        'network': '400 Gbps',
        'storage': 'NVMe SSD',
        'use_case': 'Large model training (>10B parameters)'
    },
    
    'inference_serving': {
        'instance_type': 'g5.2xlarge', 
        'gpu_count': 1,  # A10G GPU
        'memory_gb': 32,
        'network': '25 Gbps',
        'use_case': 'Real-time inference serving'
    },
    
    'batch_processing': {
        'instance_type': 'c6i.32xlarge',
        'cpu_count': 128,
        'memory_gb': 256,
        'network': '50 Gbps',
        'use_case': 'CPU-intensive preprocessing'
    }
}

def launch_training_cluster(num_instances=4):
    """Launch multi-node training cluster."""
    
    ec2 = boto3.client('ec2')
    
    # User data script for setup
    user_data_script = """#!/bin/bash
    # Install NVIDIA drivers and Docker
    sudo yum update -y
    sudo yum install -y docker
    sudo systemctl start docker
    sudo usermod -a -G docker ec2-user
    
    # Install NVIDIA container toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Setup distributed training environment
    export MASTER_ADDR=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)
    export MASTER_PORT=29500
    export WORLD_SIZE={num_instances}
    """.format(num_instances=num_instances)
    
    # Launch instances
    response = ec2.run_instances(
        ImageId='ami-0abcdef1234567890',  # Deep Learning AMI
        MinCount=num_instances,
        MaxCount=num_instances,
        InstanceType='p4d.24xlarge',
        KeyName='my-key-pair',
        SecurityGroupIds=['sg-12345678'],
        SubnetId='subnet-12345678',
        UserData=user_data_script,
        
        # Placement group for high-bandwidth networking
        Placement={
            'GroupName': 'training-cluster',
            'Strategy': 'cluster'
        },
        
        # EBS optimization
        EbsOptimized=True,
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 500,
                    'VolumeType': 'gp3',
                    'Iops': 16000,
                    'Throughput': 1000
                }
            }
        ]
    )
    
    return response['Instances']
```

### AWS Batch for Scalable ML Workloads
```python
def setup_batch_compute_environment():
    """Setup AWS Batch for scalable ML processing."""
    
    batch_client = boto3.client('batch')
    
    # Compute environment configuration
    compute_env_config = {
        'computeEnvironmentName': 'ml-processing-env',
        'type': 'MANAGED',
        'state': 'ENABLED',
        'computeResources': {
            'type': 'EC2',
            'minvCpus': 0,
            'maxvCpus': 1000,
            'desiredvCpus': 0,
            'instanceTypes': ['p3.2xlarge', 'p3.8xlarge', 'p3.16xlarge'],
            'subnets': ['subnet-12345678', 'subnet-87654321'],
            'securityGroupIds': ['sg-12345678'],
            'instanceRole': 'arn:aws:iam::123456789012:instance-profile/ecsInstanceRole',
            'tags': {
                'Project': 'ML-Training',
                'Environment': 'Production'
            },
            
            # Spot instances for cost optimization
            'bidPercentage': 50,
            'ec2Configuration': [{
                'imageType': 'ECS_AL2_NVIDIA'
            }]
        }
    }
    
    # Create compute environment
    response = batch_client.create_compute_environment(**compute_env_config)
    
    return response
```

## 📊 Analytics & Data Processing

### Amazon EMR for Big Data ML
```python
def setup_emr_cluster_for_ml():
    """Setup EMR cluster optimized for ML workloads."""
    
    emr_client = boto3.client('emr')
    
    cluster_config = {
        'Name': 'ML-Data-Processing-Cluster',
        'ReleaseLabel': 'emr-6.15.0',
        'Applications': [
            {'Name': 'Spark'},
            {'Name': 'Hadoop'},
            {'Name': 'JupyterHub'},
            {'Name': 'Zeppelin'}
        ],
        
        'Instances': {
            'MasterInstanceType': 'm5.xlarge',
            'SlaveInstanceType': 'm5.2xlarge',
            'InstanceCount': 10,
            'KeepJobFlowAliveWhenNoSteps': True,
            'TerminationProtected': False,
            
            # Mixed instance types for cost optimization
            'InstanceFleets': [
                {
                    'Name': 'Master',
                    'InstanceFleetType': 'MASTER',
                    'TargetOnDemandCapacity': 1,
                    'InstanceTypeConfigs': [
                        {
                            'InstanceType': 'm5.xlarge',
                            'WeightedCapacity': 1
                        }
                    ]
                },
                {
                    'Name': 'Core',
                    'InstanceFleetType': 'CORE', 
                    'TargetOnDemandCapacity': 2,
                    'TargetSpotCapacity': 8,
                    'InstanceTypeConfigs': [
                        {
                            'InstanceType': 'm5.2xlarge',
                            'WeightedCapacity': 1,
                            'BidPriceAsPercentageOfOnDemandPrice': 50
                        },
                        {
                            'InstanceType': 'm5.4xlarge',
                            'WeightedCapacity': 2,
                            'BidPriceAsPercentageOfOnDemandPrice': 50
                        }
                    ]
                }
            ]
        },
        
        'Configurations': [
            {
                'Classification': 'spark-defaults',
                'Properties': {
                    'spark.sql.adaptive.enabled': 'true',
                    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                    'spark.dynamicAllocation.enabled': 'true',
                    'spark.dynamicAllocation.minExecutors': '1',
                    'spark.dynamicAllocation.maxExecutors': '100'
                }
            }
        ],
        
        'ServiceRole': 'EMR_DefaultRole',
        'JobFlowRole': 'EMR_EC2_DefaultRole',
        'LogUri': 's3://my-emr-logs/',
        
        'BootstrapActions': [
            {
                'Name': 'Install ML Libraries',
                'ScriptBootstrapAction': {
                    'Path': 's3://my-bootstrap-scripts/install-ml-libs.sh'
                }
            }
        ]
    }
    
    response = emr_client.run_job_flow(**cluster_config)
    return response['JobFlowId']

# Spark ML pipeline on EMR
def create_spark_ml_pipeline():
    """Create Spark ML pipeline for feature processing."""
    
    spark_code = """
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.classification import LogisticRegression
    
    spark = SparkSession.builder.appName("MLPipeline").getOrCreate()
    
    # Load data
    df = spark.read.parquet("s3://my-bucket/ml-data/")
    
    # Feature engineering pipeline
    assembler = VectorAssembler(
        inputCols=["feature1", "feature2", "feature3"],
        outputCol="features"
    )
    
    scaler = StandardScaler(
        inputCol="features", 
        outputCol="scaledFeatures"
    )
    
    lr = LogisticRegression(
        featuresCol="scaledFeatures",
        labelCol="label"
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    
    # Train model
    model = pipeline.fit(df)
    
    # Save model
    model.write().overwrite().save("s3://my-bucket/models/spark-ml-model")
    """
    
    return spark_code
```

## 🔐 Security & Compliance

### IAM for ML Workloads
```python
def create_ml_iam_roles():
    """Create IAM roles for ML workloads with least privilege."""
    
    iam = boto3.client('iam')
    
    # SageMaker execution role
    sagemaker_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::my-ml-bucket/*",
                    "arn:aws:s3:::my-ml-bucket"
                ]
            },
            {
                "Effect": "Allow", 
                "Action": [
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
            }
        ]
    }
    
    # Create policy
    policy_response = iam.create_policy(
        PolicyName='SageMakerMLPolicy',
        PolicyDocument=json.dumps(sagemaker_policy)
    )
    
    # Create role
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    role_response = iam.create_role(
        RoleName='SageMakerExecutionRole',
        AssumeRolePolicyDocument=json.dumps(assume_role_policy)
    )
    
    # Attach policies
    iam.attach_role_policy(
        RoleName='SageMakerExecutionRole',
        PolicyArn=policy_response['Policy']['Arn']
    )
    
    return role_response['Role']['Arn']
```

### VPC Configuration for ML
```python
def setup_ml_vpc_configuration():
    """Setup VPC configuration for secure ML workloads."""
    
    vpc_config = {
        'vpc_id': 'vpc-12345678',
        'private_subnets': ['subnet-12345678', 'subnet-87654321'],
        'security_groups': {
            'training': {
                'group_id': 'sg-training-12345',
                'rules': [
                    {
                        'type': 'ingress',
                        'protocol': 'tcp',
                        'port_range': '22-22',
                        'source': '10.0.0.0/8'
                    },
                    {
                        'type': 'ingress', 
                        'protocol': 'tcp',
                        'port_range': '443-443',
                        'source': '0.0.0.0/0'
                    }
                ]
            },
            'inference': {
                'group_id': 'sg-inference-12345',
                'rules': [
                    {
                        'type': 'ingress',
                        'protocol': 'tcp', 
                        'port_range': '8080-8080',
                        'source': '10.0.0.0/8'
                    }
                ]
            }
        },
        
        # VPC Endpoints for AWS services
        'vpc_endpoints': [
            's3',
            'ecr.api',
            'ecr.dkr', 
            'logs',
            'monitoring'
        ]
    }
    
    return vpc_config
```

## 📈 Monitoring & Optimization

### CloudWatch for ML Monitoring
```python
def setup_ml_monitoring():
    """Setup comprehensive monitoring for ML workloads."""
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Custom metrics for model performance
    def put_model_metrics(model_name, accuracy, latency, throughput):
        """Put custom model performance metrics."""
        
        cloudwatch.put_metric_data(
            Namespace='ML/Models',
            MetricData=[
                {
                    'MetricName': 'ModelAccuracy',
                    'Dimensions': [
                        {
                            'Name': 'ModelName',
                            'Value': model_name
                        }
                    ],
                    'Value': accuracy,
                    'Unit': 'Percent'
                },
                {
                    'MetricName': 'InferenceLatency',
                    'Dimensions': [
                        {
                            'Name': 'ModelName', 
                            'Value': model_name
                        }
                    ],
                    'Value': latency,
                    'Unit': 'Milliseconds'
                },
                {
                    'MetricName': 'Throughput',
                    'Dimensions': [
                        {
                            'Name': 'ModelName',
                            'Value': model_name
                        }
                    ],
                    'Value': throughput,
                    'Unit': 'Count/Second'
                }
            ]
        )
    
    # CloudWatch alarms
    def create_ml_alarms():
        """Create alarms for ML workload monitoring."""
        
        alarms = [
            {
                'AlarmName': 'HighInferenceLatency',
                'MetricName': 'InferenceLatency',
                'Threshold': 1000,  # ms
                'ComparisonOperator': 'GreaterThanThreshold'
            },
            {
                'AlarmName': 'LowModelAccuracy',
                'MetricName': 'ModelAccuracy', 
                'Threshold': 85,  # percent
                'ComparisonOperator': 'LessThanThreshold'
            },
            {
                'AlarmName': 'HighGPUUtilization',
                'MetricName': 'GPUUtilization',
                'Threshold': 90,  # percent
                'ComparisonOperator': 'GreaterThanThreshold'
            }
        ]
        
        for alarm in alarms:
            cloudwatch.put_metric_alarm(
                AlarmName=alarm['AlarmName'],
                ComparisonOperator=alarm['ComparisonOperator'],
                EvaluationPeriods=2,
                MetricName=alarm['MetricName'],
                Namespace='ML/Models',
                Period=300,
                Statistic='Average',
                Threshold=alarm['Threshold'],
                ActionsEnabled=True,
                AlarmActions=[
                    'arn:aws:sns:us-east-1:123456789012:ml-alerts'
                ],
                AlarmDescription=f'Alarm for {alarm["AlarmName"]}'
            )
    
    return put_model_metrics, create_ml_alarms
```

## 💰 Cost Optimization

### Spot Instances for Training
```python
def setup_spot_training():
    """Setup spot instances for cost-effective training."""
    
    spot_config = {
        'sagemaker_spot': {
            'use_spot_instances': True,
            'max_wait_time': 24 * 3600,  # 24 hours
            'spot_stopping_condition': {
                'max_runtime_in_seconds': 7 * 24 * 3600  # 7 days
            }
        },
        
        'ec2_spot': {
            'spot_fleet_config': {
                'IamFleetRole': 'arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-tagging-role',
                'AllocationStrategy': 'diversified',
                'TargetCapacity': 8,
                'SpotPrice': '2.50',
                'LaunchSpecifications': [
                    {
                        'ImageId': 'ami-0abcdef1234567890',
                        'InstanceType': 'p3.2xlarge',
                        'KeyName': 'my-key-pair',
                        'SecurityGroups': [{'GroupId': 'sg-12345678'}],
                        'SubnetId': 'subnet-12345678',
                        'WeightedCapacity': 1
                    },
                    {
                        'ImageId': 'ami-0abcdef1234567890', 
                        'InstanceType': 'p3.8xlarge',
                        'KeyName': 'my-key-pair',
                        'SecurityGroups': [{'GroupId': 'sg-12345678'}],
                        'SubnetId': 'subnet-12345678',
                        'WeightedCapacity': 4
                    }
                ],
                'Type': 'maintain'
            }
        }
    }
    
    return spot_config

def implement_cost_monitoring():
    """Implement cost monitoring and alerts."""
    
    budgets_client = boto3.client('budgets')
    
    # Create budget for ML workloads
    budget_config = {
        'AccountId': '123456789012',
        'Budget': {
            'BudgetName': 'ML-Training-Budget',
            'BudgetLimit': {
                'Amount': '10000',
                'Unit': 'USD'
            },
            'TimeUnit': 'MONTHLY',
            'BudgetType': 'COST',
            'CostFilters': {
                'Service': ['Amazon SageMaker', 'Amazon EC2-Instance']
            }
        },
        'NotificationsWithSubscribers': [
            {
                'Notification': {
                    'NotificationType': 'ACTUAL',
                    'ComparisonOperator': 'GREATER_THAN',
                    'Threshold': 80,
                    'ThresholdType': 'PERCENTAGE'
                },
                'Subscribers': [
                    {
                        'SubscriptionType': 'EMAIL',
                        'Address': 'ml-team@company.com'
                    }
                ]
            }
        ]
    }
    
    response = budgets_client.create_budget(**budget_config)
    return response
```

## 🎯 Interview Questions & Answers

### Q1: "How would you architect a system for training and serving a 100B parameter model on AWS?"

**Expected Answer Framework**:
```python
# Architecture for 100B parameter model
architecture_design = {
    'training': {
        'compute': 'SageMaker with p4d.24xlarge instances (8x A100 GPUs)',
        'storage': 'S3 for data, EFS for shared checkpoints',
        'distribution': '3D parallelism (data + model + pipeline)',
        'networking': 'Placement groups for high bandwidth',
        'cost_optimization': 'Spot instances with checkpointing'
    },
    
    'serving': {
        'inference': 'SageMaker multi-model endpoints with auto-scaling',
        'optimization': 'Model quantization, tensor parallelism',
        'caching': 'ElastiCache for frequent requests',
        'monitoring': 'CloudWatch + custom metrics'
    },
    
    'mlops': {
        'orchestration': 'SageMaker Pipelines',
        'model_registry': 'SageMaker Model Registry', 
        'deployment': 'Blue/green deployment with SageMaker',
        'monitoring': 'SageMaker Model Monitor'
    }
}
```

### Q2: "Design a cost-optimized solution for processing 10TB of text data daily for LLM training."

**Expected Solution**:
```python
# Cost-optimized data processing pipeline
processing_pipeline = {
    'ingestion': {
        'service': 'S3 with Intelligent Tiering',
        'format': 'Parquet for efficient storage',
        'partitioning': 'By date and source for optimal queries'
    },
    
    'processing': {
        'service': 'EMR with Spot instances (70% cost savings)',
        'auto_scaling': 'Based on queue depth',
        'instance_mix': 'Mix of On-Demand and Spot instances'
    },
    
    'optimization': {
        'compression': 'LZ4 for fast decompression',
        'caching': 'S3 Transfer Acceleration',
        'scheduling': 'Process during off-peak hours'
    }
}
```

### Q3: "How would you implement real-time model monitoring and automated retraining?"

**Expected Implementation**:
```python
# Real-time monitoring and retraining system
monitoring_system = {
    'data_drift_detection': {
        'service': 'SageMaker Model Monitor',
        'metrics': 'Statistical distance measures',
        'alerts': 'CloudWatch alarms → SNS → Lambda'
    },
    
    'performance_monitoring': {
        'custom_metrics': 'Accuracy, latency, throughput',
        'dashboards': 'CloudWatch dashboards',
        'automated_alerts': 'Based on thresholds'
    },
    
    'automated_retraining': {
        'trigger': 'Performance degradation or data drift',
        'pipeline': 'SageMaker Pipelines with Step Functions',
        'validation': 'A/B testing before deployment'
    }
}
```

---

## 📝 Study Checklist

- [ ] Understand SageMaker components and capabilities
- [ ] Know when to use different compute options (EC2, Batch, EMR)
- [ ] Familiar with storage options for ML workloads
- [ ] Can design secure, cost-effective ML architectures
- [ ] Understand monitoring and optimization strategies
- [ ] Know integration patterns between AWS services
- [ ] Can estimate costs for large-scale ML workloads

**Next**: [Production ML →](../03-aws-production/09-production-ml.md)
