# Evaluation Metrics for Generative AI Models

## 🎯 Overview
Comprehensive guide to evaluation metrics for generative AI models, covering language models, diffusion models, multimodal systems, and production monitoring metrics.

## 📊 Language Model Evaluation

### Intrinsic Metrics

#### Perplexity
```python
import torch
import torch.nn.functional as F
import math

def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift for causal language modeling
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = attention_mask[:, 1:].contiguous()
            
            outputs = model(inputs, attention_mask=attention_mask[:, :-1])
            logits = outputs.logits
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                reduction='none'
            )
            
            # Apply mask and sum
            loss = loss.view(targets.size()) * mask
            total_loss += loss.sum().item()
            total_tokens += mask.sum().item()
    
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

# Usage
perplexity = calculate_perplexity(model, eval_dataloader, device)
print(f"Perplexity: {perplexity:.2f}")
```

#### Token-level Accuracy
```python
def calculate_token_accuracy(model, dataloader, device, top_k=1):
    """Calculate top-k token prediction accuracy"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = attention_mask[:, 1:].contiguous()
            
            outputs = model(inputs, attention_mask=attention_mask[:, :-1])
            logits = outputs.logits
            
            # Get top-k predictions
            _, top_k_indices = torch.topk(logits, top_k, dim=-1)
            
            # Check if target is in top-k
            targets_expanded = targets.unsqueeze(-1).expand(-1, -1, top_k)
            matches = (top_k_indices == targets_expanded).any(dim=-1)
            
            # Apply mask
            masked_matches = matches * mask.bool()
            correct_predictions += masked_matches.sum().item()
            total_predictions += mask.sum().item()
    
    accuracy = correct_predictions / total_predictions
    return accuracy
```

### Downstream Task Evaluation

#### GLUE Benchmark Implementation
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

class GLUEEvaluator:
    def __init__(self, task_name):
        self.task_name = task_name
        self.metric_functions = {
            'cola': self._matthews_correlation,
            'sst2': self._accuracy,
            'mrpc': self._f1_and_accuracy,
            'stsb': self._pearson_and_spearman,
            'qqp': self._f1_and_accuracy,
            'mnli': self._accuracy,
            'qnli': self._accuracy,
            'rte': self._accuracy,
            'wnli': self._accuracy
        }
    
    def evaluate(self, predictions, labels):
        if self.task_name not in self.metric_functions:
            raise ValueError(f"Unknown task: {self.task_name}")
        
        return self.metric_functions[self.task_name](predictions, labels)
    
    def _accuracy(self, predictions, labels):
        return {"accuracy": accuracy_score(labels, predictions)}
    
    def _f1_and_accuracy(self, predictions, labels):
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average='binary')
        }
    
    def _matthews_correlation(self, predictions, labels):
        return {"matthews_correlation": matthews_corrcoef(labels, predictions)}
    
    def _pearson_and_spearman(self, predictions, labels):
        pearson_corr, _ = pearsonr(predictions, labels)
        spearman_corr, _ = spearmanr(predictions, labels)
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2
        }

# Usage
evaluator = GLUEEvaluator('sst2')
metrics = evaluator.evaluate(predictions, labels)
print(f"SST-2 Accuracy: {metrics['accuracy']:.3f}")
```

#### Few-shot Learning Evaluation
```python
class FewShotEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_few_shot(self, examples, test_data, k_shot=5):
        """Evaluate few-shot performance"""
        results = []
        
        for test_item in test_data:
            # Create few-shot prompt
            prompt = self._create_few_shot_prompt(examples[:k_shot], test_item)
            
            # Generate prediction
            prediction = self._generate_prediction(prompt)
            
            # Evaluate prediction
            is_correct = self._check_prediction(prediction, test_item['label'])
            results.append(is_correct)
        
        accuracy = sum(results) / len(results)
        return {"few_shot_accuracy": accuracy}
    
    def _create_few_shot_prompt(self, examples, test_item):
        prompt_parts = []
        
        for example in examples:
            prompt_parts.append(f"Input: {example['input']}\nOutput: {example['output']}")
        
        prompt_parts.append(f"Input: {test_item['input']}\nOutput:")
        return "\n\n".join(prompt_parts)
    
    def _generate_prediction(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False
            )
        
        # Extract only the generated part
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    def _check_prediction(self, prediction, label):
        # Simple exact match (can be made more sophisticated)
        return prediction.strip().lower() == label.strip().lower()
```

## 🎨 Generative Model Evaluation

### Image Generation Metrics

#### Fréchet Inception Distance (FID)
```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from scipy import linalg

class FIDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.inception_model = self._load_inception_model()
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_inception_model(self):
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()  # Remove final classification layer
        model.eval()
        return model.to(self.device)
    
    def get_features(self, images):
        """Extract features from images using Inception-v3"""
        if not isinstance(images, torch.Tensor):
            images = torch.stack([self.transform(img) for img in images])
        
        images = images.to(self.device)
        
        with torch.no_grad():
            features = self.inception_model(images)
        
        return features.cpu().numpy()
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate FID between real and generated images"""
        # Get features
        real_features = self.get_features(real_images)
        gen_features = self.get_features(generated_images)
        
        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        fid = self._calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fid
    
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet distance between two multivariate Gaussians"""
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + 
                np.trace(sigma2) - 2 * tr_covmean)

# Usage
fid_calculator = FIDCalculator()
fid_score = fid_calculator.calculate_fid(real_images, generated_images)
print(f"FID Score: {fid_score:.2f}")
```

#### Inception Score (IS)
```python
class InceptionScoreCalculator:
    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.inception_model = self._load_inception_model()
    
    def _load_inception_model(self):
        model = models.inception_v3(pretrained=True)
        model.eval()
        return model.to(self.device)
    
    def calculate_inception_score(self, images, splits=10):
        """Calculate Inception Score"""
        N = len(images)
        
        # Get predictions
        predictions = []
        for i in range(0, N, self.batch_size):
            batch = images[i:i+self.batch_size]
            if not isinstance(batch, torch.Tensor):
                batch = torch.stack([self.transform(img) for img in batch])
            
            batch = batch.to(self.device)
            
            with torch.no_grad():
                pred = self.inception_model(batch)
                predictions.append(F.softmax(pred, dim=1).cpu())
        
        predictions = torch.cat(predictions, dim=0)
        
        # Calculate IS for each split
        scores = []
        for i in range(splits):
            part = predictions[i * (N // splits): (i + 1) * (N // splits)]
            
            # Calculate marginal distribution
            py = torch.mean(part, dim=0)
            
            # Calculate KL divergence
            kl_div = part * (torch.log(part) - torch.log(py.unsqueeze(0)))
            kl_div = torch.sum(kl_div, dim=1)
            
            # Calculate IS for this split
            score = torch.exp(torch.mean(kl_div))
            scores.append(score.item())
        
        return np.mean(scores), np.std(scores)

# Usage
is_calculator = InceptionScoreCalculator()
is_mean, is_std = is_calculator.calculate_inception_score(generated_images)
print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
```

### Text Generation Metrics

#### BLEU Score Implementation
```python
from collections import Counter
import math

class BLEUCalculator:
    def __init__(self, max_n=4):
        self.max_n = max_n
    
    def calculate_bleu(self, candidates, references, weights=None):
        """Calculate BLEU score for multiple candidates against references"""
        if weights is None:
            weights = [1.0/self.max_n] * self.max_n
        
        scores = []
        for candidate, reference_list in zip(candidates, references):
            score = self._sentence_bleu(candidate, reference_list, weights)
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def _sentence_bleu(self, candidate, references, weights):
        """Calculate BLEU score for a single candidate"""
        candidate_tokens = candidate.split()
        reference_tokens_list = [ref.split() for ref in references]
        
        # Calculate precision for each n-gram order
        precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._calculate_ngram_precision(
                candidate_tokens, reference_tokens_list, n
            )
            precisions.append(precision)
        
        # Calculate brevity penalty
        bp = self._brevity_penalty(candidate_tokens, reference_tokens_list)
        
        # Calculate weighted geometric mean
        if any(p == 0 for p in precisions):
            return 0.0
        
        log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
        geometric_mean = math.exp(sum(log_precisions))
        
        return bp * geometric_mean
    
    def _calculate_ngram_precision(self, candidate, references, n):
        """Calculate n-gram precision"""
        candidate_ngrams = self._get_ngrams(candidate, n)
        
        if not candidate_ngrams:
            return 0.0
        
        # Get maximum counts from all references
        max_counts = {}
        for reference in references:
            ref_ngrams = self._get_ngrams(reference, n)
            for ngram, count in ref_ngrams.items():
                max_counts[ngram] = max(max_counts.get(ngram, 0), count)
        
        # Calculate clipped counts
        clipped_counts = {}
        for ngram, count in candidate_ngrams.items():
            clipped_counts[ngram] = min(count, max_counts.get(ngram, 0))
        
        return sum(clipped_counts.values()) / sum(candidate_ngrams.values())
    
    def _get_ngrams(self, tokens, n):
        """Extract n-grams from token list"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _brevity_penalty(self, candidate, references):
        """Calculate brevity penalty"""
        candidate_length = len(candidate)
        
        # Find closest reference length
        ref_lengths = [len(ref) for ref in references]
        closest_ref_length = min(ref_lengths, 
                                key=lambda x: abs(x - candidate_length))
        
        if candidate_length > closest_ref_length:
            return 1.0
        else:
            return math.exp(1 - closest_ref_length / candidate_length)

# Usage
bleu_calculator = BLEUCalculator()
bleu_score = bleu_calculator.calculate_bleu(candidates, references)
print(f"BLEU Score: {bleu_score:.3f}")
```

#### ROUGE Score Implementation
```python
class ROUGECalculator:
    def __init__(self):
        pass
    
    def calculate_rouge_l(self, candidate, reference):
        """Calculate ROUGE-L score"""
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        
        lcs_length = self._lcs_length(candidate_tokens, reference_tokens)
        
        if len(candidate_tokens) == 0:
            recall = 0.0
        else:
            recall = lcs_length / len(reference_tokens)
        
        if len(reference_tokens) == 0:
            precision = 0.0
        else:
            precision = lcs_length / len(candidate_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {"rouge_l_precision": precision, 
                "rouge_l_recall": recall, 
                "rouge_l_f1": f1}
    
    def calculate_rouge_n(self, candidate, reference, n=2):
        """Calculate ROUGE-n score"""
        candidate_ngrams = self._get_ngrams(candidate.split(), n)
        reference_ngrams = self._get_ngrams(reference.split(), n)
        
        if not reference_ngrams:
            return {"rouge_n_precision": 0.0, "rouge_n_recall": 0.0, "rouge_n_f1": 0.0}
        
        # Calculate overlap
        overlap = 0
        for ngram in candidate_ngrams:
            if ngram in reference_ngrams:
                overlap += min(candidate_ngrams[ngram], reference_ngrams[ngram])
        
        recall = overlap / sum(reference_ngrams.values()) if reference_ngrams else 0.0
        precision = overlap / sum(candidate_ngrams.values()) if candidate_ngrams else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {f"rouge_{n}_precision": precision, 
                f"rouge_{n}_recall": recall, 
                f"rouge_{n}_f1": f1}
    
    def _lcs_length(self, seq1, seq2):
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _get_ngrams(self, tokens, n):
        """Extract n-grams with counts"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
```

## 🔄 Multimodal Evaluation

### Vision-Language Model Evaluation
```python
class VisionLanguageEvaluator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def evaluate_image_captioning(self, images, reference_captions):
        """Evaluate image captioning performance"""
        generated_captions = []
        
        for image in images:
            # Generate caption
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            generated_captions.append(caption)
        
        # Calculate metrics
        bleu_calc = BLEUCalculator()
        rouge_calc = ROUGECalculator()
        
        bleu_score = bleu_calc.calculate_bleu(generated_captions, reference_captions)
        
        rouge_scores = []
        for gen, ref in zip(generated_captions, reference_captions):
            rouge_score = rouge_calc.calculate_rouge_l(gen, ref[0])  # Use first reference
            rouge_scores.append(rouge_score['rouge_l_f1'])
        
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        
        return {
            "bleu": bleu_score,
            "rouge_l": avg_rouge,
            "generated_captions": generated_captions
        }
    
    def evaluate_vqa(self, questions, images, answers):
        """Evaluate Visual Question Answering"""
        correct_answers = 0
        total_questions = len(questions)
        
        for question, image, correct_answer in zip(questions, images, answers):
            # Process input
            inputs = self.processor(
                images=image, 
                text=question, 
                return_tensors="pt"
            )
            
            # Generate answer
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=20)
            
            predicted_answer = self.processor.decode(
                generated_ids[0], skip_special_tokens=True
            )
            
            # Check if answer is correct (simple exact match)
            if predicted_answer.strip().lower() == correct_answer.strip().lower():
                correct_answers += 1
        
        accuracy = correct_answers / total_questions
        return {"vqa_accuracy": accuracy}
```

### Cross-modal Retrieval Evaluation
```python
class CrossModalRetrievalEvaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate_image_text_retrieval(self, images, texts, batch_size=32):
        """Evaluate image-text retrieval performance"""
        # Get embeddings
        image_embeddings = self._get_image_embeddings(images, batch_size)
        text_embeddings = self._get_text_embeddings(texts, batch_size)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Image-to-text retrieval
        i2t_ranks = self._calculate_ranks(similarity_matrix)
        i2t_metrics = self._calculate_retrieval_metrics(i2t_ranks)
        
        # Text-to-image retrieval
        t2i_ranks = self._calculate_ranks(similarity_matrix.T)
        t2i_metrics = self._calculate_retrieval_metrics(t2i_ranks)
        
        return {
            "image_to_text": i2t_metrics,
            "text_to_image": t2i_metrics,
            "average_recall_at_1": (i2t_metrics["recall_at_1"] + t2i_metrics["recall_at_1"]) / 2
        }
    
    def _get_image_embeddings(self, images, batch_size):
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(batch)
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)
    
    def _get_text_embeddings(self, texts, batch_size):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode_text(batch)
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)
    
    def _calculate_ranks(self, similarity_matrix):
        """Calculate rank of correct matches"""
        ranks = []
        for i in range(similarity_matrix.size(0)):
            # Get similarities for query i
            similarities = similarity_matrix[i]
            
            # Rank by similarity (descending order)
            _, indices = torch.sort(similarities, descending=True)
            
            # Find rank of correct match (assuming diagonal is correct)
            rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        return ranks
    
    def _calculate_retrieval_metrics(self, ranks):
        """Calculate retrieval metrics from ranks"""
        ranks = torch.tensor(ranks, dtype=torch.float32)
        
        recall_at_1 = (ranks <= 1).float().mean().item()
        recall_at_5 = (ranks <= 5).float().mean().item()
        recall_at_10 = (ranks <= 10).float().mean().item()
        median_rank = ranks.median().item()
        mean_rank = ranks.mean().item()
        
        return {
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "median_rank": median_rank,
            "mean_rank": mean_rank
        }
```

## 📈 Production Monitoring Metrics

### Real-time Model Performance Monitoring
```python
class ProductionMetricsCollector:
    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.metrics_buffer = []
        
    def log_inference(self, input_data, output_data, latency, metadata=None):
        """Log a single inference for monitoring"""
        metric_entry = {
            "timestamp": time.time(),
            "model_name": self.model_name,
            "model_version": self.version,
            "latency_ms": latency * 1000,
            "input_tokens": len(input_data.get("input_ids", [])),
            "output_tokens": len(output_data.get("generated_ids", [])),
            "metadata": metadata or {}
        }
        
        # Add quality metrics if available
        if "confidence_scores" in output_data:
            metric_entry["avg_confidence"] = np.mean(output_data["confidence_scores"])
            metric_entry["min_confidence"] = np.min(output_data["confidence_scores"])
        
        self.metrics_buffer.append(metric_entry)
        
    def calculate_performance_metrics(self, window_size_minutes=60):
        """Calculate performance metrics over a time window"""
        current_time = time.time()
        window_start = current_time - (window_size_minutes * 60)
        
        # Filter metrics in time window
        window_metrics = [
            m for m in self.metrics_buffer 
            if m["timestamp"] >= window_start
        ]
        
        if not window_metrics:
            return {}
        
        latencies = [m["latency_ms"] for m in window_metrics]
        input_lengths = [m["input_tokens"] for m in window_metrics]
        output_lengths = [m["output_tokens"] for m in window_metrics]
        
        metrics = {
            "requests_per_minute": len(window_metrics) / window_size_minutes,
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_input_length": np.mean(input_lengths),
            "avg_output_length": np.mean(output_lengths),
            "total_tokens_processed": sum(input_lengths) + sum(output_lengths)
        }
        
        # Add confidence metrics if available
        confidences = [m.get("avg_confidence") for m in window_metrics if "avg_confidence" in m]
        if confidences:
            metrics["avg_confidence"] = np.mean(confidences)
            metrics["low_confidence_rate"] = sum(1 for c in confidences if c < 0.5) / len(confidences)
        
        return metrics
    
    def detect_anomalies(self, current_metrics, baseline_metrics, threshold=2.0):
        """Detect performance anomalies compared to baseline"""
        anomalies = []
        
        key_metrics = ["avg_latency_ms", "p95_latency_ms", "avg_confidence"]
        
        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                if baseline_val > 0:
                    ratio = current_val / baseline_val
                    
                    if ratio > threshold:
                        anomalies.append({
                            "metric": metric,
                            "current_value": current_val,
                            "baseline_value": baseline_val,
                            "ratio": ratio,
                            "type": "performance_degradation"
                        })
                    elif ratio < (1 / threshold):
                        anomalies.append({
                            "metric": metric,
                            "current_value": current_val,
                            "baseline_value": baseline_val,
                            "ratio": ratio,
                            "type": "unexpected_improvement"
                        })
        
        return anomalies
```

### A/B Testing Framework
```python
class ABTestingFramework:
    def __init__(self, test_name, control_model, treatment_model):
        self.test_name = test_name
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.results = {"control": [], "treatment": []}
        
    def run_comparison(self, test_data, evaluation_metric, traffic_split=0.5):
        """Run A/B test comparison"""
        np.random.shuffle(test_data)
        split_point = int(len(test_data) * traffic_split)
        
        control_data = test_data[:split_point]
        treatment_data = test_data[split_point:]
        
        # Evaluate control model
        control_results = []
        for item in control_data:
            result = self._evaluate_single_item(self.control_model, item, evaluation_metric)
            control_results.append(result)
            self.results["control"].append(result)
        
        # Evaluate treatment model
        treatment_results = []
        for item in treatment_data:
            result = self._evaluate_single_item(self.treatment_model, item, evaluation_metric)
            treatment_results.append(result)
            self.results["treatment"].append(result)
        
        # Calculate statistical significance
        significance_result = self._calculate_significance(control_results, treatment_results)
        
        return {
            "control_mean": np.mean(control_results),
            "treatment_mean": np.mean(treatment_results),
            "improvement": (np.mean(treatment_results) - np.mean(control_results)) / np.mean(control_results),
            "statistical_significance": significance_result,
            "sample_sizes": {"control": len(control_results), "treatment": len(treatment_results)}
        }
    
    def _evaluate_single_item(self, model, item, metric_function):
        """Evaluate a single item with the given model"""
        prediction = model.generate(item["input"])
        return metric_function(prediction, item["target"])
    
    def _calculate_significance(self, control_results, treatment_results, alpha=0.05):
        """Calculate statistical significance using t-test"""
        from scipy import stats
        
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_results, control_results)
        
        is_significant = p_value < alpha
        
        # Calculate confidence interval for the difference
        diff_mean = np.mean(treatment_results) - np.mean(control_results)
        diff_se = np.sqrt(np.var(treatment_results)/len(treatment_results) + 
                         np.var(control_results)/len(control_results))
        
        ci_lower = diff_mean - 1.96 * diff_se
        ci_upper = diff_mean + 1.96 * diff_se
        
        return {
            "p_value": p_value,
            "is_significant": is_significant,
            "t_statistic": t_stat,
            "confidence_interval": (ci_lower, ci_upper),
            "effect_size": diff_mean / np.sqrt((np.var(treatment_results) + np.var(control_results)) / 2)
        }
```

## 🎯 Interview Questions & Answers

### Q1: How would you design an evaluation pipeline for a new multimodal AI system?
**Answer**:
1. **Task-specific metrics**: BLEU/ROUGE for captioning, accuracy for VQA, recall@k for retrieval
2. **Human evaluation**: Collect human ratings for relevance, coherence, and factual accuracy
3. **Robustness testing**: Evaluate on out-of-distribution data, adversarial examples
4. **Efficiency metrics**: Latency, memory usage, throughput measurements
5. **Fairness evaluation**: Test performance across different demographic groups
6. **A/B testing**: Compare against baseline models in production-like settings

### Q2: What are the limitations of automatic evaluation metrics for generative models?
**Answer**:
- **BLEU/ROUGE**: Focus on n-gram overlap, miss semantic similarity
- **Perplexity**: Domain-dependent, doesn't correlate with human judgment
- **FID**: Requires large sample sizes, sensitive to image preprocessing
- **Human evaluation**: Expensive, subjective, doesn't scale
- **Task-specific bias**: Metrics optimized for specific tasks may not generalize

### Q3: How do you handle evaluation at scale in production?
**Answer**:
1. **Sampling strategies**: Statistical sampling to reduce evaluation cost
2. **Automated monitoring**: Real-time metrics collection and anomaly detection
3. **Continuous evaluation**: Regular benchmarking on held-out test sets
4. **Human-in-the-loop**: Strategic human evaluation for critical cases
5. **Multi-tier evaluation**: Fast automatic metrics + slower comprehensive evaluation

### Q4: Explain the trade-offs between different similarity metrics for text generation.
**Answer**:
- **Exact match**: High precision, low recall, ignores semantic equivalence
- **BLEU**: Good for translation, poor for dialogue or creative tasks
- **Embedding similarity**: Captures semantics, but may be too permissive
- **Learned metrics**: Task-specific but require training data
- **Human evaluation**: Most reliable but expensive and slow

### Q5: How would you evaluate a model's robustness and safety?
**Answer**:
1. **Adversarial testing**: Prompt injection, jailbreaking attempts
2. **Bias evaluation**: Test outputs across demographic groups
3. **Factual accuracy**: Fact-checking against reliable sources
4. **Toxicity detection**: Automated toxicity scoring and human review
5. **Edge case testing**: Unusual inputs, corner cases, out-of-distribution data
6. **Red team evaluation**: Structured attempts to find model failures

## 📊 Evaluation Best Practices

### 1. **Comprehensive Evaluation Strategy**
- Multiple complementary metrics
- Both automatic and human evaluation
- Regular benchmark updates
- Cross-validation across domains

### 2. **Statistical Rigor**
- Proper train/validation/test splits
- Statistical significance testing
- Confidence intervals for metrics
- Multiple random seeds for reproducibility

### 3. **Production Considerations**
- Real-time monitoring dashboards
- Automated alert systems
- Performance degradation detection
- User feedback integration

### 4. **Fairness and Ethics**
- Bias detection across groups
- Inclusive evaluation datasets
- Transparency in metric limitations
- Regular audit procedures

## 🔗 Additional Resources

- **Papers**: "Beyond Accuracy: Behavioral Testing of NLP Models", "Evaluation of Text Generation: A Survey"
- **Tools**: NLTK, spaCy, Hugging Face Evaluate, MLflow
- **Benchmarks**: GLUE, SuperGLUE, HELM, BIG-bench
- **Platforms**: Weights & Biases, Neptune, TensorBoard
