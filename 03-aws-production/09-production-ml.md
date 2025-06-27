# Production ML Systems for Generative AI

## 🎯 Overview
Comprehensive guide to deploying, monitoring, and scaling generative AI models in production environments, covering infrastructure, optimization, monitoring, and operational best practices.

## 🏗️ Production Architecture Patterns

### Microservices Architecture for ML
```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import time
from typing import List, Dict, Optional
import logging

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class GenerationResponse(BaseModel):
    generated_text: str
    tokens_used: int
    latency_ms: float
    model_version: str

# ML Service Class
class MLService:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
            logging.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text asynchronously"""
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                request.prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            latency = (time.time() - start_time) * 1000
            
            return GenerationResponse(
                generated_text=generated_text,
                tokens_used=len(generated_ids),
                latency_ms=latency,
                model_version="v1.0"
            )
            
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail="Generation failed")

# FastAPI Application
app = FastAPI(title="Generative AI Service")
ml_service = MLService("gpt2")  # Replace with your model

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text endpoint"""
    return await ml_service.generate(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": ml_service.model_name}

@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint for monitoring"""
    # Implement metrics collection
    return {"requests_processed": 1000, "avg_latency_ms": 250}
```

### Load Balancing and Auto-scaling
```python
import asyncio
import aiohttp
from typing import List
import random
import time

class LoadBalancer:
    def __init__(self, service_endpoints: List[str]):
        self.endpoints = service_endpoints
        self.health_status = {endpoint: True for endpoint in service_endpoints}
        self.request_counts = {endpoint: 0 for endpoint in service_endpoints}
        self.response_times = {endpoint: [] for endpoint in service_endpoints}
        
    async def route_request(self, request_data: dict) -> dict:
        """Route request to best available endpoint"""
        available_endpoints = [
            endpoint for endpoint, healthy in self.health_status.items() 
            if healthy
        ]
        
        if not available_endpoints:
            raise Exception("No healthy endpoints available")
        
        # Choose endpoint based on strategy (round-robin, least-connections, etc.)
        endpoint = self._select_endpoint(available_endpoints)
        
        return await self._send_request(endpoint, request_data)
    
    def _select_endpoint(self, endpoints: List[str]) -> str:
        """Select endpoint using weighted round-robin based on performance"""
        # Calculate weights based on inverse of average response time
        weights = []
        for endpoint in endpoints:
            avg_response_time = (
                sum(self.response_times[endpoint][-10:]) / 
                len(self.response_times[endpoint][-10:])
                if self.response_times[endpoint] else 1.0
            )
            weight = 1.0 / max(avg_response_time, 0.1)  # Avoid division by zero
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return endpoints[i]
        
        return endpoints[-1]  # Fallback
    
    async def _send_request(self, endpoint: str, request_data: dict) -> dict:
        """Send request to specific endpoint"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{endpoint}/generate", json=request_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update metrics
                        response_time = time.time() - start_time
                        self.response_times[endpoint].append(response_time)
                        self.request_counts[endpoint] += 1
                        
                        return result
                    else:
                        raise Exception(f"Request failed with status {response.status}")
                        
        except Exception as e:
            # Mark endpoint as unhealthy and retry with another
            self.health_status[endpoint] = False
            logging.error(f"Request to {endpoint} failed: {e}")
            raise
    
    async def health_check_loop(self):
        """Periodic health checks for all endpoints"""
        while True:
            for endpoint in self.endpoints:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{endpoint}/health", timeout=5) as response:
                            self.health_status[endpoint] = response.status == 200
                except:
                    self.health_status[endpoint] = False
            
            await asyncio.sleep(30)  # Check every 30 seconds

# Auto-scaling Controller
class AutoScaler:
    def __init__(self, min_instances: int = 2, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.target_utilization = 0.7
        
    async def monitor_and_scale(self, load_balancer: LoadBalancer):
        """Monitor load and scale instances accordingly"""
        while True:
            current_load = self._calculate_current_load(load_balancer)
            
            if current_load > self.target_utilization and self.current_instances < self.max_instances:
                await self._scale_up()
            elif current_load < (self.target_utilization * 0.5) and self.current_instances > self.min_instances:
                await self._scale_down()
            
            await asyncio.sleep(60)  # Check every minute
    
    def _calculate_current_load(self, load_balancer: LoadBalancer) -> float:
        """Calculate current system load"""
        active_endpoints = sum(load_balancer.health_status.values())
        if active_endpoints == 0:
            return 1.0
        
        # Simple load calculation based on response times
        avg_response_times = []
        for endpoint, times in load_balancer.response_times.items():
            if times and load_balancer.health_status[endpoint]:
                avg_response_times.append(sum(times[-10:]) / len(times[-10:]))
        
        if not avg_response_times:
            return 0.0
        
        avg_response_time = sum(avg_response_times) / len(avg_response_times)
        # Normalize to 0-1 range (assuming 1 second is maximum acceptable)
        return min(avg_response_time / 1.0, 1.0)
    
    async def _scale_up(self):
        """Add new instance"""
        # Implementation would depend on your orchestration platform
        # (Kubernetes, ECS, etc.)
        logging.info(f"Scaling up from {self.current_instances} to {self.current_instances + 1}")
        self.current_instances += 1
    
    async def _scale_down(self):
        """Remove instance"""
        logging.info(f"Scaling down from {self.current_instances} to {self.current_instances - 1}")
        self.current_instances -= 1
```

## 🚀 Model Optimization for Production

### Model Quantization
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

class ModelQuantizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization"""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Target layers
            dtype=torch.qint8
        )
        return quantized_model
    
    def quantize_static(self, model: nn.Module, calibration_data) -> nn.Module:
        """Apply static quantization with calibration"""
        model.eval()
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with representative data
        with torch.no_grad():
            for batch in calibration_data:
                model_prepared(**batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        return quantized_model
    
    def quantize_qat(self, model: nn.Module, train_dataloader, num_epochs: int = 3):
        """Quantization Aware Training"""
        model.train()
        
        # Set QAT config
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        model_prepared = torch.quantization.prepare_qat(model)
        
        # Train with quantization simulation
        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-5)
        
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                outputs = model_prepared(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        # Convert to quantized model
        model_prepared.eval()
        quantized_model = torch.quantization.convert(model_prepared)
        return quantized_model

# ONNX Optimization
class ONNXOptimizer:
    def __init__(self):
        self.optimization_level = "all"
    
    def export_to_onnx(self, model, tokenizer, output_path: str):
        """Export PyTorch model to ONNX format"""
        model.eval()
        
        # Create dummy input
        dummy_input = tokenizer(
            "Hello world", 
            return_tensors="pt", 
            max_length=128, 
            padding="max_length"
        )
        
        # Export
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=11
        )
    
    def optimize_onnx(self, model_path: str, optimized_path: str):
        """Optimize ONNX model"""
        import onnx
        from onnxruntime.tools.optimize_model import optimize_model
        
        # Load and optimize
        optimized_model = optimize_model(
            model_path,
            model_type="bert",  # or appropriate type
            optimization_level=self.optimization_level
        )
        
        optimized_model.save_model_to_file(optimized_path)

# TensorRT Optimization
class TensorRTOptimizer:
    def __init__(self):
        pass
    
    def optimize_with_tensorrt(self, onnx_path: str, engine_path: str):
        """Convert ONNX to TensorRT engine"""
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        return engine
```

### Model Serving with Batching
```python
import asyncio
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import torch

@dataclass
class BatchedRequest:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    future: asyncio.Future

class DynamicBatcher:
    def __init__(self, 
                 model, 
                 tokenizer,
                 max_batch_size: int = 8,
                 max_wait_time: float = 0.01,  # 10ms
                 device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.device = device
        
        self.pending_requests: List[BatchedRequest] = []
        self.processing = False
        
    async def add_request(self, request_id: str, prompt: str, 
                         max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Add request to batch and return result"""
        future = asyncio.Future()
        
        batched_request = BatchedRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            future=future
        )
        
        self.pending_requests.append(batched_request)
        
        # Trigger processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        # Wait for result
        return await future
    
    async def _process_batch(self):
        """Process pending requests in batches"""
        self.processing = True
        
        try:
            while self.pending_requests:
                # Wait for batch to fill or timeout
                await self._wait_for_batch()
                
                if not self.pending_requests:
                    break
                
                # Extract batch
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                current_batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                # Process batch
                await self._process_batch_requests(current_batch)
                
        finally:
            self.processing = False
    
    async def _wait_for_batch(self):
        """Wait for batch to fill or timeout"""
        start_time = time.time()
        
        while (len(self.pending_requests) < self.max_batch_size and 
               time.time() - start_time < self.max_wait_time):
            await asyncio.sleep(0.001)  # 1ms
    
    async def _process_batch_requests(self, batch: List[BatchedRequest]):
        """Process a batch of requests"""
        try:
            # Prepare batch inputs
            prompts = [req.prompt for req in batch]
            max_tokens = max(req.max_tokens for req in batch)
            
            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate for batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=batch[0].temperature,  # Use first request's temperature
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode results
            for i, request in enumerate(batch):
                # Extract generated tokens for this request
                input_length = inputs['input_ids'][i].shape[0]
                generated_ids = outputs[i][input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Set result
                request.future.set_result(generated_text)
                
        except Exception as e:
            # Set exception for all requests in batch
            for request in batch:
                request.future.set_exception(e)

# Usage with FastAPI
from fastapi import FastAPI
import uuid

app = FastAPI()
# Initialize batcher with your model
batcher = DynamicBatcher(model, tokenizer)

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100, temperature: float = 0.7):
    request_id = str(uuid.uuid4())
    result = await batcher.add_request(request_id, prompt, max_tokens, temperature)
    return {"generated_text": result, "request_id": request_id}
```

## 📊 Monitoring and Observability

### Comprehensive Monitoring System
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time
import logging
from typing import Dict, Any
import json

class MLMetricsCollector:
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'ml_requests_total', 
            'Total ML requests', 
            ['model_name', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'ml_request_duration_seconds',
            'ML request duration',
            ['model_name', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'ml_active_requests',
            'Currently active ML requests',
            ['model_name']
        )
        
        self.model_memory_usage = Gauge(
            'ml_model_memory_bytes',
            'Model memory usage',
            ['model_name', 'device']
        )
        
        self.token_throughput = Histogram(
            'ml_tokens_per_second',
            'Token generation throughput',
            ['model_name']
        )
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, Any] = {}
        
    def record_request(self, model_name: str, endpoint: str, 
                      duration: float, status: str, tokens_generated: int = 0):
        """Record a request completion"""
        self.request_count.labels(
            model_name=model_name, 
            endpoint=endpoint, 
            status=status
        ).inc()
        
        self.request_duration.labels(
            model_name=model_name, 
            endpoint=endpoint
        ).observe(duration)
        
        if tokens_generated > 0:
            tokens_per_second = tokens_generated / duration
            self.token_throughput.labels(model_name=model_name).observe(tokens_per_second)
    
    def update_active_requests(self, model_name: str, delta: int):
        """Update active request count"""
        self.active_requests.labels(model_name=model_name).inc(delta)
    
    def update_memory_usage(self, model_name: str, device: str, memory_bytes: int):
        """Update memory usage"""
        self.model_memory_usage.labels(model_name=model_name, device=device).set(memory_bytes)
    
    def record_custom_metric(self, name: str, value: Any, labels: Dict[str, str] = None):
        """Record custom metric"""
        timestamp = time.time()
        self.custom_metrics[name] = {
            "value": value,
            "timestamp": timestamp,
            "labels": labels or {}
        }

class RequestTracker:
    def __init__(self, metrics_collector: MLMetricsCollector):
        self.metrics = metrics_collector
        
    def __call__(self, model_name: str, endpoint: str):
        """Decorator for tracking requests"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                self.metrics.update_active_requests(model_name, 1)
                
                try:
                    result = await func(*args, **kwargs)
                    status = "success"
                    
                    # Extract token count if available
                    tokens_generated = 0
                    if hasattr(result, 'tokens_used'):
                        tokens_generated = result.tokens_used
                    
                    return result
                    
                except Exception as e:
                    status = "error"
                    logging.error(f"Request failed: {e}")
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    self.metrics.record_request(
                        model_name, endpoint, duration, status, tokens_generated
                    )
                    self.metrics.update_active_requests(model_name, -1)
            
            return wrapper
        return decorator

# Anomaly Detection
class AnomalyDetector:
    def __init__(self, window_size: int = 100, threshold_multiplier: float = 2.0):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.metric_history: Dict[str, List[float]] = {}
        
    def detect_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect if a metric value is anomalous"""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        history = self.metric_history[metric_name]
        
        # Add new value
        history.append(value)
        if len(history) > self.window_size:
            history.pop(0)
        
        # Need at least 10 samples for detection
        if len(history) < 10:
            return False
        
        # Calculate statistics
        mean = sum(history[:-1]) / len(history[:-1])  # Exclude current value
        variance = sum((x - mean) ** 2 for x in history[:-1]) / len(history[:-1])
        std_dev = variance ** 0.5
        
        # Check if current value is anomalous
        threshold = mean + (self.threshold_multiplier * std_dev)
        return value > threshold
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for metric_name, values in self.metric_history.items():
            if len(values) >= 2:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                
                summary[metric_name] = {
                    "mean": mean,
                    "std_dev": variance ** 0.5,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary

# Log Analysis
class LogAnalyzer:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.error_patterns = [
            r"ERROR",
            r"CRITICAL",
            r"Failed to",
            r"Connection timeout",
            r"Memory error"
        ]
        
    def analyze_logs(self, hours_back: int = 1) -> Dict[str, Any]:
        """Analyze logs for errors and patterns"""
        import re
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        error_counts = {}
        total_lines = 0
        error_lines = 0
        
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    total_lines += 1
                    
                    # Check if line is within time window
                    # (assumes standard log format with timestamp)
                    
                    # Check for error patterns
                    for pattern in self.error_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            error_lines += 1
                            error_counts[pattern] = error_counts.get(pattern, 0) + 1
                            break
                            
        except FileNotFoundError:
            return {"error": "Log file not found"}
        
        return {
            "total_lines": total_lines,
            "error_lines": error_lines,
            "error_rate": error_lines / total_lines if total_lines > 0 else 0,
            "error_patterns": error_counts,
            "analysis_time": datetime.now().isoformat()
        }

# Usage Example
metrics_collector = MLMetricsCollector()
request_tracker = RequestTracker(metrics_collector)
anomaly_detector = AnomalyDetector()

@request_tracker(model_name="gpt-model", endpoint="generate")
async def generate_with_monitoring(request: GenerationRequest):
    # Your generation logic here
    result = await ml_service.generate(request)
    
    # Check for anomalies
    if anomaly_detector.detect_anomaly("latency_ms", result.latency_ms):
        logging.warning(f"Anomalous latency detected: {result.latency_ms}ms")
    
    return result
```

## 🔧 Infrastructure as Code

### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  labels:
    app: ml-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: your-registry/ml-service:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "your-model-name"
        - name: DEVICE
          value: "cuda"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
spec:
  selector:
    app: ml-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Terraform Configuration
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.21"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    gpu_nodes = {
      desired_capacity = 2
      max_capacity     = 10
      min_capacity     = 1
      
      instance_types = ["p3.2xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        Application = "ml-inference"
      }
    }
  }
}

# S3 Bucket for Model Storage
resource "aws_s3_bucket" "model_storage" {
  bucket = "${var.cluster_name}-ml-models"
  
  versioning {
    enabled = true
  }
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# ElastiCache for Caching
resource "aws_elasticache_cluster" "ml_cache" {
  cluster_id           = "${var.cluster_name}-cache"
  engine               = "redis"
  node_type           = "cache.r5.large"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis6.x"
  port                = 6379
  subnet_group_name   = aws_elasticache_subnet_group.ml_cache.name
  security_group_ids  = [aws_security_group.ml_cache.id]
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ml_service_logs" {
  name              = "/aws/eks/${var.cluster_name}/ml-service"
  retention_in_days = 7
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "ml-inference-cluster"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}
```

## 🔒 Security and Compliance

### API Security Implementation
```python
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import hashlib
import hmac
import time
from typing import Optional

class SecurityManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
        """Verify JWT token"""
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid authentication credentials")
            return username
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    def rate_limit(self, max_requests: int = 100, window_seconds: int = 3600):
        """Rate limiting decorator"""
        def decorator(func):
            request_counts = {}
            
            async def wrapper(*args, **kwargs):
                # Get client identifier (could be IP, user ID, etc.)
                client_id = "default"  # Implement proper client identification
                
                current_time = time.time()
                window_start = current_time - window_seconds
                
                # Clean old entries
                request_counts[client_id] = [
                    timestamp for timestamp in request_counts.get(client_id, [])
                    if timestamp > window_start
                ]
                
                # Check rate limit
                if len(request_counts[client_id]) >= max_requests:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Record request
                request_counts[client_id].append(current_time)
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator

class InputValidator:
    @staticmethod
    def validate_prompt(prompt: str) -> str:
        """Validate and sanitize input prompt"""
        # Check length
        if len(prompt) > 10000:
            raise HTTPException(status_code=400, detail="Prompt too long")
        
        # Check for potentially harmful content
        harmful_patterns = [
            "injection",
            "script",
            "<script>",
            "javascript:",
            "eval(",
            "exec("
        ]
        
        prompt_lower = prompt.lower()
        for pattern in harmful_patterns:
            if pattern in prompt_lower:
                raise HTTPException(status_code=400, detail="Invalid input content")
        
        return prompt.strip()
    
    @staticmethod
    def validate_parameters(max_tokens: int, temperature: float) -> tuple:
        """Validate generation parameters"""
        if max_tokens < 1 or max_tokens > 2048:
            raise HTTPException(status_code=400, detail="Invalid max_tokens value")
        
        if temperature < 0.0 or temperature > 2.0:
            raise HTTPException(status_code=400, detail="Invalid temperature value")
        
        return max_tokens, temperature

class DataPrivacy:
    def __init__(self):
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b'  # Phone number
        ]
    
    def detect_pii(self, text: str) -> bool:
        """Detect PII in text"""
        import re
        
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize PII in text"""
        import re
        
        anonymized = text
        
        # Replace patterns with placeholders
        replacements = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b': '[CREDIT_CARD]',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
            r'\b\d{3}-\d{3}-\d{4}\b': '[PHONE]'
        }
        
        for pattern, replacement in replacements.items():
            anonymized = re.sub(pattern, replacement, anonymized)
        
        return anonymized

# Usage in FastAPI
app = FastAPI()
security_manager = SecurityManager("your-secret-key")
input_validator = InputValidator()
data_privacy = DataPrivacy()

@app.post("/generate")
async def secure_generate(
    request: GenerationRequest,
    current_user: str = Depends(security_manager.verify_token)
):
    # Validate inputs
    prompt = input_validator.validate_prompt(request.prompt)
    max_tokens, temperature = input_validator.validate_parameters(
        request.max_tokens, request.temperature
    )
    
    # Check for PII
    if data_privacy.detect_pii(prompt):
        # Optionally anonymize or reject
        prompt = data_privacy.anonymize_text(prompt)
    
    # Process request
    result = await ml_service.generate(GenerationRequest(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    ))
    
    return result
```

## 🎯 Interview Questions & Answers

### Q1: How would you design a system to serve a large language model to millions of users?
**Answer**:
1. **Load Balancing**: Use application load balancers with health checks
2. **Auto-scaling**: Horizontal pod autoscaling based on CPU/memory/custom metrics
3. **Caching**: Redis/Memcached for frequent queries, model output caching
4. **Batching**: Dynamic batching to improve throughput
5. **Model Optimization**: Quantization, ONNX/TensorRT optimization
6. **Monitoring**: Comprehensive metrics, alerting, distributed tracing
7. **Security**: Authentication, rate limiting, input validation

### Q2: What are the key challenges in deploying generative AI models in production?
**Answer**:
- **Latency Requirements**: Real-time inference vs batch processing trade-offs
- **Resource Management**: GPU memory optimization, model sharding
- **Consistency**: Ensuring reproducible outputs across deployments
- **Safety**: Content filtering, prompt injection prevention
- **Cost**: Balancing performance with infrastructure costs
- **Monitoring**: Detecting model drift, performance degradation

### Q3: How do you handle model versioning and rollback in production?
**Answer**:
1. **Blue-Green Deployment**: Maintain two identical environments
2. **Canary Releases**: Gradual traffic shifting to new versions
3. **A/B Testing**: Statistical comparison of model versions
4. **Feature Flags**: Control model routing at runtime
5. **Automated Rollback**: Triggered by performance thresholds
6. **Model Registry**: Centralized versioning and metadata management

### Q4: Describe your approach to monitoring ML model performance in production.
**Answer**:
- **Real-time Metrics**: Latency, throughput, error rates
- **Model Quality**: Accuracy, confidence scores, output analysis
- **Data Drift**: Input distribution changes over time
- **Resource Usage**: GPU utilization, memory consumption
- **Business Metrics**: User satisfaction, task completion rates
- **Alerting**: Automated alerts for anomalies and degradation

### Q5: How would you optimize inference costs for a large-scale deployment?
**Answer**:
1. **Model Optimization**: Quantization, pruning, distillation
2. **Infrastructure**: Spot instances, reserved capacity
3. **Auto-scaling**: Scale down during low traffic
4. **Caching**: Reduce redundant computations
5. **Batching**: Improve GPU utilization
6. **Multi-tenancy**: Share resources across applications
7. **Edge Computing**: Reduce data transfer and latency costs

## 📋 Production Checklist

### Pre-deployment
- [ ] Model validation and testing
- [ ] Performance benchmarking
- [ ] Security assessment
- [ ] Disaster recovery plan
- [ ] Monitoring setup
- [ ] Documentation complete

### Deployment
- [ ] Blue-green deployment strategy
- [ ] Health checks configured
- [ ] Auto-scaling policies
- [ ] Load balancer configuration
- [ ] SSL/TLS certificates
- [ ] Backup procedures

### Post-deployment
- [ ] Performance monitoring
- [ ] Error tracking
- [ ] User feedback collection
- [ ] Cost optimization
- [ ] Regular model updates
- [ ] Security audits

## 🔗 Additional Resources

- **Tools**: Kubernetes, Docker, Terraform, Prometheus, Grafana
- **Platforms**: AWS EKS, GCP GKE, Azure AKS
- **Monitoring**: DataDog, New Relic, Elastic Stack
- **Security**: HashiCorp Vault, AWS IAM, cert-manager
