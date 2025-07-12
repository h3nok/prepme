# Amazon Applied Scientist On‑Site Interview Preparation Guide

---

## Executive Summary

Amazon’s Applied Scientist loop blends rigorous machine‑learning system design, practical coding drills, and deep alignment with the company’s 16 Leadership Principles (LPs). Expect five one‑hour sessions—two coding, one or two ML system‑design/technical depth, and one or two LP‑heavy rounds that interleave behavioural and technical probes. In most loops behavioural content consumes **40‑50 %** of each interview, with the Bar‑Raiser pushing aggressively on customer impact and metrics ([igotanoffer.com](https://igotanoffer.com/en/advice/amazon-applied-scientist-interview?utm_source=chatgpt.com)) ([reddit.com](https://www.reddit.com/r/leetcode/comments/1jwond7/amazon_applied_scientist_interview_experience/?utm_source=chatgpt.com)).  This guide arms you with end‑to‑end preparation material: detailed loop anatomy, exemplar LP stories, fully‑worked coding solutions (PyTorch‑autograd allowed only), ML breadth flash cards with answers, and two ready‑to‑speak system‑design templates (Fraud Detection and Computer Vision).  A 14‑day power study plan and curated resource list round things out.

---

## 1  Interview Loop Anatomy

| Round                        | Typical Focus                                                       | Success Signals                                                      |
| ---------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Coding #1**                | Classic DS&A plus clean Python                                      | O(1) space, clear narration, edge‑cases covered                      |
| **Coding #2 (ML coding)**    | Implement an optimizer, loss, or small model from scratch           | Correct math, unit‑tested code, time‑boxed < 25 min                  |
| **ML System Design / Depth** | End‑to‑end design for prod‑scale model (fraud, recommender, CV/NLP) | Clear KPI framing, multiple candidate architectures, monitoring plan |
| **Research / ML Breadth**    | Rapid‑fire fundamentals + practical trade‑offs                      | Concise derivations, knowing *when* to apply which model             |
| **Bar‑Raiser (LP focus)**    | Aggressive LP probing woven into tech discussion                    | Quantified impact, reflection, learnings                             |

Interviews are calibrated against Amazon’s LPs ([amazon.jobs](https://www.amazon.jobs/content/our-workplace/leadership-principles?utm_source=chatgpt.com)), and candidates who cannot anchor technical decisions in customer or business metrics rarely clear the bar‑raiser ([igotanoffer.com](https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview?utm_source=chatgpt.com)).

---

## 2  Leadership Principles Cheat‑Sheet

Below are compact STAR prompts you can rehearse.  For each LP prepare **three** distinct stories; rotate fresh examples if an interviewer re‑asks.

| LP                       | STAR Hook                                                                        | Metric                                                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Customer Obsession       | Detected silent data‑drift that inflated false positives; shipped hot‑fix in 4 h | −20 % FP, +\$1.2 M revenue protect                                                                                                                                                  |
| Invent & Simplify        | Re‑engineered rules engine into single GNN micro‑service                         | 10× latency cut                                                                                                                                                                     |
| Dive Deep                | Latency spike traced to cold start in SageMaker; implemented warm pool           | 55 ms → 22 ms p95 ([aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/rad-ai-reduces-real-time-inference-latency-by-50-using-amazon-sagemaker/?utm_source=chatgpt.com)) |
| Are Right, A Lot         | Predicted recall drop due to holiday skew; A/B confirmed                         | +8 pp recall                                                                                                                                                                        |
| Insist on High Standards | Re‑wrote feature pipeline tests; 0 failed deploys in Q4                          | 15 → 0 regression bugs                                                                                                                                                              |
| Earn Trust               | Published explanation dashboard for Risk team                                    | NPS ↑ +28                                                                                                                                                                           |
| Learn & Be Curious       | Completed AWS Certified MLOps in 6 weeks                                         | new infra PoC                                                                                                                                                                       |

Use precise numbers; bar‑raisers value quantified outcomes ([igotanoffer.com](https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview?utm_source=chatgpt.com)).

---

## 3  ML Coding—Fully‑Worked Solutions

### 3.0  Coding #1 — DS&A Classics

Amazon’s first coding round typically focuses on **general data‑structures & algorithms problems** (arrays, linked lists, hash maps, trees/graphs, intervals, DP).  Solutions are expected to be *space‑optimal* (often O(1)) and narrated clearly.  Review the patterns and sample implementations below before moving to the ML‑heavy round.

#### 3.0.1  Recurring Patterns  ([Medium](https://medium.com/swlh/how-to-study-for-data-structures-and-algorithms-interviews-at-faang-65043e00b5df))

- Two‑Pointers / Sliding‑Window — arrays & strings.
- Hash Map for O(1) look‑ups.
- Stack for balanced‑parentheses / monotonic‑stack.
- Binary Search on sorted or answer‑space.
- DFS/BFS for trees, graphs, matrices.
- Dynamic Programming (1‑D / 2‑D table, rolling O(1) space).

#### 3.0.2  High‑Yield Questions  (curated from Glassdoor/LeetCode top Amazon list ([LeetCode](https://leetcode.com/problem-list/7p5x763/)) ([DesignGurus](https://www.designgurus.io/blog/amazon-14-question)))

| # | Problem                        | Pattern           | Edge Cases                       | Goal Space                                                                         | Python Sketch          |
| - | ------------------------------ | ----------------- | -------------------------------- | ---------------------------------------------------------------------------------- | ---------------------- |
| 1 | Two Sum                        | Hash Map          | negative nums; duplicate indices | O(N) time / O(N) space.O(1) space variant: sort + two‑pointer, but destroys order. | See `two_sum.py` below |
| 2 | Reverse Linked List            | Pointers          | 1‑node list; empty list          | O(N) / **O(1)**                                                                    | `reverse_list.py`      |
| 3 | Detect Cycle in Directed Graph | DFS, in‑stack set | self‑loop; disconnected graph    | O(V+E) / O(V)                                                                      | `has_cycle.py`         |
| 4 | LRU Cache (Design)             | Hash Map + DLL    | capacity = 1                     | O(1) per op / O(C)                                                                 | `lru_cache.py`         |
| 5 | Longest Substring w/o Repeat   | Sliding Window    | unicode; all same char           | O(N) / O(min(N,Σ))                                                                 | `longest_sub.py`       |
| 6 | Search Rotated Array           | Binary Search     | duplicates; no rotation          | O(log N) / **O(1)**                                                                | `search_rot.py`        |
| 7 | Number of Islands              | BFS               | diagonal adjacency?              | O(NM) / O(min(N,M))                                                                | `num_islands.py`       |
| 8 | Merge Intervals                | Sort + sweep      | nested intervals                 | O(N log N) / **O(1)**                                                              | `merge_int.py`         |

---

##### two\_sum.py (hash map, narrated)

```python
# ===== Two Sum =====
# Return indices i,j such that nums[i] + nums[j] = target.
# Hash‑map gives O(N) time, O(N) extra space.
# Interview twist: can you trade space for O(N log N) sort?

def two_sum(nums, target):
    seen = {}                      # val -> index
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:     # found pair
            return [seen[complement], i]
        seen[x] = i                # store after check to avoid i==j
    return []                      # no solution ↦ empty list
```

##### reverse\_list.py (iterative O(1) space)

```python
class ListNode:
    def __init__(self, val=0, nxt=None):
        self.val, self.next = val, nxt

def reverse_list(head):
    """Reverse singly linked list in‑place."""
    prev, cur = None, head
    while cur:
        nxt = cur.next      # 1. save next
        cur.next = prev     # 2. reverse pointer
        prev, cur = cur, nxt  # 3. advance both ptrs
    return prev
# Edge cases: head is None, single node → loop runs 0 / 1 times.
```

##### search\_rot.py (binary search w/ rotation awareness)

```python
def search_rot(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        # Left half sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1  # not found
# Handles no rotation (fully sorted) because first branch holds.
```

Focus on narrating **invariants**, checking **boundary indices** (0, len‑1), and talking through **empty inputs** and **duplicated values** during the interview.

---

### 3.1  Linear Regression (batch GD, PyTorch autograd allowed)

```python
# ===== Batch Gradient Descent Linear Regression =====
# Goal: learn weight vector w that minimises MSE ‖Xw − y‖²
# Key interview checks:
#   • Can you derive the gradient?  ∇ = 2 Xᵀ(Xw − y) / N
#   • Why not use closed‑form?  (XᵀX)⁻¹ is numerically unstable for ill‑conditioned X.
#   • How do you pick learning‑rate?  Try lr=1e‑2 → diverges if features un‑standardised.

import torch
import torch.optim as optim

torch.manual_seed(0)            # reproducibility – interviewers love this!
N, d = 1_000, 5                 # N samples, d features
X = torch.randn(N, d)           # design matrix 𝑿
true_w = torch.randn(d, 1)      # ground‑truth weights

# Generate labels:  y = X·w*  + Gaussian noise ε~𝒩(0,0.1)
y = X @ true_w + 0.1 * torch.randn(N, 1)

# Initialise learnable weights with gradient tracking
a = torch.zeros(d, 1, requires_grad=True)

opt = optim.SGD([a], lr=1e-2)   # SGD is fine for small data; Adam more robust on noisy grads

for epoch in range(200):        # fixed epoch budget gives deterministic runtime
    opt.zero_grad()             # 1. clear old gradients ← *easy to forget!*
    y_hat = X @ a               # 2. forward pass
    loss = torch.mean((y_hat - y)**2)  # 3. compute MSE
    loss.backward()             # 4. populate a.grad via autograd DAG
    opt.step()                  # 5. gradient descent update

# At convergence ‖a − true_w‖∞ ≈ 0.05 → sanity check.
```

A closed‑form solution via `(XᵀX)⁻¹Xᵀy` exists but showcases numerical instability for ill‑conditioned matrices; gradient‑descent avoids explicit inversion ([medium.com](https://medium.com/%40sahin.samia/understanding-pytorch-basics-through-building-a-logistic-regression-model-from-scratch-71be33a43a00?utm_source=chatgpt.com)).

### 3.2  Logistic Regression (+ L2)

```python
# ===== Logistic Regression with L2 Regularisation =====
# Binary classification:  p(y=1|x) = σ(x·w + b)
# Gotchas:
#   • For severe class‑imbalance, switch to focal‑loss or weight positive class.
#   • torch.sigmoid overflows for |x|>88 → safe for float32 magnitude.

import torch
import torch.nn as nn
import torch.optim as optim

class Logistic(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        # initialise near‑zero weights; random init would break symmetry irrelevantly here
        self.w = nn.Parameter(torch.zeros(d, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = X @ self.w + self.b
        return torch.sigmoid(logits)

def train_lr(X: torch.Tensor, y: torch.Tensor, lam: float = 1e-2,
             lr: float = 1e-2, epochs: int = 300) -> Logistic:
    """lam – L2 penalty; lr – learning rate. y must be float 0/1 of shape (N,1)."""
    model = Logistic(X.size(1))
    opt   = optim.Adam(model.parameters(), lr=lr)  # Adam tackles poorly scaled features
    for epoch in range(epochs):
        opt.zero_grad()
        probs = model(X)
        ce_loss = nn.functional.binary_cross_entropy(probs, y)
        l2_pen  = lam * (model.w ** 2).sum()
        loss = ce_loss + l2_pen
        loss.backward()
        opt.step()
    return model
```

Add quadratic terms (`x₁², x₂², x₁x₂`) or use a polynomial kernel to fit quadratically separable data ([xavierbourretsicotte.github.io](https://xavierbourretsicotte.github.io/Kernel_feature_map.html?utm_source=chatgpt.com)).

### 3.3  k‑Nearest Neighbors

Training cost is **O(1)**—just store the dataset; inference is **O(N d)** for Euclidean scan ([medium.com](https://medium.com/nerd-for-tech/why-the-time-complexity-for-training-k-nearest-neighbors-is-o-1-5b8f417104cf?utm_source=chatgpt.com)) ([en.wikipedia.org](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm?utm_source=chatgpt.com)).  Accelerate with FAISS or Approximate NN.

**Pseudo‑code:**

```python
# ===== Brute‑force k‑Nearest Neighbour (classification) =====
# Complexity:
#   Train  – O(1) (store dataset)
#   Query  – O(N·d) per point → poor scalability.  Use FAISS/HNSW for ANN.
# Edge cases:
#   • Tie‑breaking: choose smallest distance or lower label id.
#   • Missing values: impute or distance masking.

import torch


def knn_predict(X_train: torch.Tensor,           # shape (N, d)
                y_train: torch.Tensor,           # shape (N,)
                x_query: torch.Tensor,           # shape (d,)
                k: int = 5) -> int:
    """Return majority label among k nearest Euclidean neighbours."""
    # 1. Squared distance (no sqrt) – monotonic for ranking
    dists2 = ((X_train - x_query) ** 2).sum(dim=1)

    # 2. Indices of k smallest distances; topk with largest=False is efficient
    knn_idx = torch.topk(dists2, k, largest=False).indices

    # 3. Mode of labels (classification). For regression: y_train[knn_idx].mean()
    return torch.mode(y_train[knn_idx]).values.item()
```

### 3.4  1‑Hidden‑Layer Neural Network (+ Dropout)

```python
# ===== Minimal 1‑Hidden‑Layer Multilayer Perceptron =====
# Design choices:
#   • ReLU for fast convergence; swap with GELU for better performance on large models.
#   • Dropout combats co‑adaptation; disable during eval.
#   • Weight initialisation: default Kaiming works for ReLU.

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),            # avoids vanishing gradients vs tanh/sigmoid
            nn.Dropout(p_drop),   # active only when model.train() is True
            nn.Linear(d_h, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Gotcha:
#   Call model.eval() before validation/testing to freeze dropout & batch‑norm stats.
```

ReLU preserves gradient magnitude for positive inputs, easing deep training compared with tanh/sigmoid ([telnyx.com](https://telnyx.com/learn-ai/rectified-linear-unit?utm_source=chatgpt.com)) ([machinelearningmastery.com](https://www.machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/?utm_source=chatgpt.com)).

### 3.5  Loss & Activation Functions — Deep Dive

Loss (objective) functions guide gradient descent by mapping model outputs to a scalar “cost.” Picking the wrong loss can **slow convergence, mask bugs, or misalign with business KPIs.** Below is a concise but thorough cheat‑sheet.

#### 3.5.1  Core Activations

| Activation     | Formula           | Gradient       | Typical Use     | Gotchas                                             |   |                      |
| -------------- | ----------------- | -------------- | --------------- | --------------------------------------------------- | - | -------------------- |
| **Sigmoid**    | σ(z)=1/(1+e^{−z}) | σ(z)(1−σ(z))   | Binary logits   | Gradient vanishes when                              | z | ≫0; clip inputs <88. |
| **Tanh**       | tanh(z)=2σ(2z)−1  | 1−tanh²(z)     | RNNs pre‑LSTM   | Saturates on±1; use GRU/LSTM instead.               |   |                      |
| **ReLU**       | max(0,z)          | 1[z>0]         | CNN/MLP default | “Dead neurons” if lr too high.                      |   |                      |
| **Leaky ReLU** | max(αz,z)         | 1[z>0]+α1[z≤0] | GANs            | Pick α≈0.01.                                        |   |                      |
| **Softmax**    | e^{zᵢ}/Σ e^{zⱼ}   | pᵢ(δᵢⱼ−pⱼ)     | Multiclass      | Compute on *logits*; subtract z\_max for stability. |   |                      |

#### 3.5.2  Supervised Losses & When to Use Them

| Loss                           | Formula (per‑sample)                 | Task Fit             | Pros                              | Cons / Pitfalls                                         |                    |                                        |                |                         |
| ------------------------------ | ------------------------------------ | -------------------- | --------------------------------- | ------------------------------------------------------- | ------------------ | -------------------------------------- | -------------- | ----------------------- |
| **MSE (L2)**                   | (ŷ−y)²                              | Regression           | Smooth gradient; closed‑form soln | Sensitive to outliers; mismatch for heavy‑tailed noise. |                    |                                        |                |                         |
| **MAE (L1)**                   |                                      | ŷ−y                 |                                   | Regression median                                       | Robust to outliers | Non‑differentiable at 0; slower optim. |                |                         |
| **Huber**                      | ½(ŷ−y)² if                          | Δ                    | <δ else δ(                        | Δ                                                       | −½δ)               | Regression, RL                         | Combines L1+L2 | Need δ hyper‑parameter. |
| **Binary Cross‑Entropy (BCE)** | −[y log p +(1−y) log (1−p)]          | Binary class         | Probabilistic, differentiable     | Must avoid log(0): add ε or use `BCEWithLogitsLoss`.    |                    |                                        |                |                         |
| **BCE with Logits**            | BCE on logits z with `sigmoid` fused | Binary               | Numerical‑stable; single op       | Forgetting reduction='sum' vs 'mean' changes scale.     |                    |                                        |                |                         |
| **Categorical Cross‑Entropy**  | −Σ yᵢ log pᵢ                         | Multiclass softmax   | Works with one‑hot labels         | Use `CrossEntropyLoss` which fuses `log_softmax`.       |                    |                                        |                |                         |
| **Focal Loss**                 | −α(1−pᵧ)^γ log pᵧ                    | Imbalanced detection | Down‑weights easy negatives       | Tune α, γ; heavier compute.                             |                    |                                        |                |                         |
| **Hinge (SVM)**                | max(0,1−y f(x))                      | Large‑margin binary  | Sparse gradients                  | Non‑probabilistic; requires y∈{±1}.                     |                    |                                        |                |                         |
| **Triplet / Contrastive**      | max(d(a,p)−d(a,n)+m,0)               | Metric learning      | Learns embedding                  | Needs hard‑example mining.                              |                    |                                        |                |                         |

> **Rule‑of‑thumb:** *Classification?* → BCE (binary) or CE (multi‑class).\
> *Regression with outliers?* → start with Huber.\
> *Extreme class‑imbalance* (fraud, defect‑detection)? → Focal or class‑weighted BCE.

#### 3.5.3  Numerically Stable Implementations

```python
# ----- Binary Cross‑Entropy w/ logits (preferred) -----
import torch, torch.nn.functional as F

logits = model(x)              # raw, un‑squashed scores
labels = torch.empty_like(logits).bernoulli_(0.1)
loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

# Internally:  BCE = max(z,0) − z*y + log(1+exp(−|z|))
# This avoids overflow when |z| is large.

# ----- Manual Softmax Cross‑Entropy (multi‑class) -----
logits = model(x)              # shape (N, C)
log_prob = logits - logits.logsumexp(dim=1, keepdim=True)  # log‑softmax
nll_loss = -log_prob[torch.arange(N), targets]             # NLL
loss = nll_loss.mean()
```

*Gotchas*

- Using `sigmoid` **followed** by `BCELoss` duplicates the sigmoid when networking in evaluation mode—> probabilities get squashed twice.
- For CE the target must be **class indices**, *not* one‑hot; automatic reduction defaults to **mean**—scales differently from TF default **sum**.
- Label‑smoothing (e.g., ε=0.1) can regularise CE by preventing over‑confident logits; PyTorch `CrossEntropyLoss(label_smoothing=0.1)`.

#### 3.5.4  Connecting Loss to Business Metrics

- Fraud detection usually optimises **expected cost** = FP · OpEx + FN · chargeback.  BCE aligns with log‑likelihood, but you can re‑weight positive class proportional to chargeback cost.
- Recommendation quality often measured by **NDCG**; softmax‑CE on implicit positive vs sampled negatives approximates ranking objective (see *BPR loss*).
- CV quality inspection may care about **Intersection‑over‑Union**; training with BCE on pixel masks but monitoring IoU ensures production alignment.

---

## 4  ML Breadth Flash Cards (with Answers)

  ML Breadth Flash Cards (with Answers)

| Question                                                          | 20‑second Whiteboard Answer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Parameters of a depth‑1 decision tree on two features (x₁,x₂)** | Internal node stores `(feature, threshold)`; two leaves store predictions. Total learnable params = 1 threshold + 2 leaf values ([scikit-learn.org](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html?utm_source=chatgpt.com))                                                                                                                                                                                                                                                                                                      |
| **Parameter vector for logistic regression (2‑D)**                | θ ∈ ℝ³: `[bias, w₁, w₂]`. Decision boundary w₁x₁ + w₂x₂ + bias = 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Adapting logistic regression to quadratic separation**          | Expand features: `[1, x₁, x₂, x₁², x₂², x₁x₂]` or apply polynomial kernel; train standard LR on mapped space ([xavierbourretsicotte.github.io](https://xavierbourretsicotte.github.io/Kernel_feature_map.html?utm_source=chatgpt.com))                                                                                                                                                                                                                                                                                                                               |
| **Precision vs Recall**                                           | Precision = TP/(TP+FP) prioritises false‑positive cost; Recall = TP/(TP+FN) prioritises false‑negatives ([en.wikipedia.org](https://en.wikipedia.org/wiki/Precision_and_recall?utm_source=chatgpt.com)).  In fraud detection you often sweep thresholds for PR curve rather than ROC due to heavy class imbalance ([kount.com](https://kount.com/blog/precision-recall-when-conventional-fraud-metrics-fall-short?utm_source=chatgpt.com)) ([evidentlyai.com](https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall?utm_source=chatgpt.com)). |
| **F1‑score rationale**                                            | Harmonic mean balances precision/recall; useful when uneven class distribution ([developers.google.com](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?utm_source=chatgpt.com))                                                                                                                                                                                                                                                                                                                                |
| **Why ReLU over tanh?**                                           | Non‑saturating for positive z, sparse activations, mitigates vanishing gradient ([telnyx.com](https://telnyx.com/learn-ai/rectified-linear-unit?utm_source=chatgpt.com))                                                                                                                                                                                                                                                                                                                                                                                             |
| **k‑NN Pros/Cons**                                                | + No training, interpretable; – slow at inference, curse of dimensionality ([en.wikipedia.org](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm?utm_source=chatgpt.com))                                                                                                                                                                                                                                                                                                                                                                                  |

---

## 5  System Design Templates

### 5.1  Fraud Detection (Real‑Time)

1. **Problem & KPI** — optimise *cost = FP × \$OPEX + FN × chargebacks*.
2. **Data Ingestion** — Kinesis stream → AWS Feature Store (online) + S3 (offline) ([aws.amazon.com](https://aws.amazon.com/sagemaker-ai/deploy/?utm_source=chatgpt.com)).
3. **Feature Groups** — transaction, user behaviour, device/IP risk, merchant profile.
4. **Model Options**
   | Candidate            | Pros                          | Cons                     |
   | -------------------- | ----------------------------- | ------------------------ |
   | XGBoost              | tabular, low‑latency          | limited temporal context |
   | Deep Sets            | order‑invariant cart features | heavier infra            |
   | GNN (user‑txn graph) | collusion rings               | cold‑start               |
5. **Serving** — SageMaker real‑time endpoint p95 < 50 ms, proven by Rad AI and Cisco case studies ([aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/rad-ai-reduces-real-time-inference-latency-by-50-using-amazon-sagemaker/?utm_source=chatgpt.com)) ([aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/cisco-achieves-50-latency-improvement-using-amazon-sagemaker-inference-faster-autoscaling-feature/?utm_source=chatgpt.com)).
6. **Monitoring** — KS‑stat data drift, population stability index, Shadow mode.
7. **Retraining** — nightly incremental; champion/challenger gating.
8. **Privacy** — optional federated learning via Flower on SageMaker ([aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/fraud-detection-empowered-by-federated-learning-with-the-flower-framework-on-amazon-sagemaker-ai/?utm_source=chatgpt.com)).

### 5.2  Computer Vision Quality Inspection

Same 8‑step skeleton; swap streaming source with S3 Batch + SNS triggers; model candidates: EfficientNet (latency), Vision Transformers (accuracy).  Edge deployment via SageMaker Neo.

---

## 6  14‑Day Power Study Plan

| Day | AM                                  | PM                                                                                                                                                                                          |
| --- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Mock system design (fraud)          | Review LP STARs                                                                                                                                                                             |
| 2   | Code: linear + logistic timed       | ML breadth quiz (Huyen) ([huyenchip.com](https://huyenchip.com/ml-interviews-book/?utm_source=chatgpt.com))                                                                                 |
| 3   | Second system design (CV)           | Implement KNN + benchmark                                                                                                                                                                   |
| 4   | Review decision trees, kernels      | Build NN with dropout                                                                                                                                                                       |
| 5   | End‑to‑end SageMaker lab            | Watch Rad AI latency talk ([aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/rad-ai-reduces-real-time-inference-latency-by-50-using-amazon-sagemaker/?utm_source=chatgpt.com)) |
| 6   | Mock full loop (record)             | Retrospective improvements                                                                                                                                                                  |
| 7   | REST — light flash cards            |                                                                                                                                                                                             |
| 8   | Bar‑raiser style peer interview     | Precision‑Recall threshold tuning                                                                                                                                                           |
| 9   | Live‑coding drills (leetcode‑style) | Re‑write bug‑free quickly                                                                                                                                                                   |
| 10  | CV or NLP domain deep‑dive          | Update STAR metrics                                                                                                                                                                         |
| 11  | Dry‑run presentation of research    |                                                                                                                                                                                             |
| 12  | Second mock loop                    | Sleep hygiene                                                                                                                                                                               |
| 13  | Travel/logistics                    | Skim notes                                                                                                                                                                                  |
| 14  | On‑site — execute                   |                                                                                                                                                                                             |

---

## 7  Curated Resource Stack

- **Huyen Chip’s ML Interview Book** — 200+ breadth questions ([huyenchip.com](https://huyenchip.com/ml-interviews-book/?utm_source=chatgpt.com)).
- **IGotAnOffer Guides** — Loop breakdown, Bar‑raiser tips ([igotanoffer.com](https://igotanoffer.com/en/advice/amazon-applied-scientist-interview?utm_source=chatgpt.com)) ([igotanoffer.com](https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview?utm_source=chatgpt.com)).
- **SageMaker Doc Set** — deployment, latency optimisation ([aws.amazon.com](https://aws.amazon.com/sagemaker-ai/deploy/?utm_source=chatgpt.com)) ([docs.aws.amazon.com](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html?utm_source=chatgpt.com)).
- **AWS Fraud Detection Blog Series** — federated learning, feature engineering ([aws.amazon.com](https://aws.amazon.com/blogs/machine-learning/fraud-detection-empowered-by-federated-learning-with-the-flower-framework-on-amazon-sagemaker-ai/?utm_source=chatgpt.com)).
- **Scikit‑learn tree structure example** — visualise parameters ([scikit-learn.org](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html?utm_source=chatgpt.com)).
- **Kernel & Feature Map Tutorial** — polynomial LR intuition ([xavierbourretsicotte.github.io](https://xavierbourretsicotte.github.io/Kernel_feature_map.html?utm_source=chatgpt.com)).
- **Google ML Crash Course** — metrics refresher ([developers.google.com](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?utm_source=chatgpt.com)).
- **ReLU & Vanishing Gradient Reading** — Telnyx, MachineLearningMastery ([telnyx.com](https://telnyx.com/learn-ai/rectified-linear-unit?utm_source=chatgpt.com)) ([machinelearningmastery.com](https://www.machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/?utm_source=chatgpt.com)).

---

### Final Reminder

1. **Tie every technical decision to a customer metric.** 2. **Narrate trade‑offs out loud.** 3. **Lead with Amazon LP language.**  You now have both theory and code at your fingertips—time to practise until every section flows in under 15 minutes.

