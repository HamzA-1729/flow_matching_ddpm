# Flow Matching vs DDPM: Image Generation from Scratch (DiT Backbone)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A first-principles PyTorch implementation and comparative benchmark of Continuous-Time Optimal Transport Flow Matching against Denoising Diffusion Probabilistic Models (DDPM) using a shared Patch Diffusion Transformer (DiT) backbone.*

---

## 📌 Project Status & Roadmap

> [!NOTE]
> **Status:** Active Research & Implementation. The shared DiT architecture and shape verification tests are complete and passing. Mathematical losses and samplers are currently being implemented.

- [x] **DiT Backbone Architecture** (`src/flow_matching_ddpm/model.py`)
- [x] **AdaLN-Zero Identity Initialization & 2D Positional Encodings**
- [x] **Backbone Unit Tests** (`tests/test_shapes.py`)
- [ ] **Flow Matching Velocity Loss** (`src/flow_matching_ddpm/loss.py`)
- [ ] **DDPM Noise Prediction Loss** (`src/flow_matching_ddpm/loss.py`)
- [ ] **Euler & Heun Numerical ODE Solvers** (`src/flow_matching_ddpm/sampler.py`)
- [ ] **Ancestral & DDIM Deterministic Samplers** (`src/flow_matching_ddpm/sampler.py`)
- [ ] **MNIST Training Loop with EMA & CFG** (`src/flow_matching_ddpm/train.py`)
- [ ] **Deterministic 1-Batch Overfit Test** (`tests/test_overfit.py`)
- [ ] **NFE Step Sweep Benchmark Grid (5, 10, 20, 50, 100)**

---

## 🔬 Core Hypothesis & Overview

Generative models turn pure Gaussian noise into structured images. **DDPM** (Ho et al., 2020) frames this as a reverse Brownian diffusion process, removing noise across hundreds of curved stochastic steps. Truncating this schedule to few steps leads to severe trajectory drift.

**Flow Matching** (Lipman et al., 2022) replaces curved diffusion paths with **straight-line probability paths** connecting noise directly to clean data, generating images by solving an Ordinary Differential Equation (ODE).

This project investigates: **Does Flow Matching produce sharper images at low function evaluations (5–20 NFE) compared to DDPM and DDIM when trained on the exact same architecture and compute budget?**

---

## 📐 Architecture: Diffusion Transformer (DiT)

Both paradigms share the identical neural network engine to ensure a scientifically fair comparison:

```
Input Image (B, C, H, W)
       │
       ▼
[ Patch Embedding: Conv2d(stride=p) ]  ──► Tokens: (B, N, D) + 2D Sin/Cos Pos Embed
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ AdaLN-Zero Transformer Block (Repeated D times)        │
│                                                        │
│ Conditioning: c = Embed_time(t) + Embed_class(y)       │
│ Modulation: (γ1, β1, α1, γ2, β2, α2) = Linear(c)       │
│                                                        │
│ x = x + α1 ⊙ MultiHeadAttention( LayerNorm(x)·(1+γ1) + β1 )
│ x = x + α2 ⊙ MLP( LayerNorm(x)·(1+γ2) + β2 )           │
└────────────────────────────────────────────────────────┘
       │
       ▼
[ Final AdaLN Modulation + Linear Projection ]
       │
       ▼
Unpatchify ──► Output Tensor: (B, C, H, W)  [Noise or Velocity]
```

---

## 📚 Mathematical Formulations

Notation:
* $x_{\text{data}} \in \mathbb{R}^{C \times H \times W}$: Clean data sample.
* $x_{\text{noise}} \sim \mathcal{N}(0, \mathbf{I})$: Standard Gaussian noise.
* $t$: Continuous or discrete time variable.
* $y \in \{0, \dots, K\}$: Class condition label (where $K$ denotes the null token for CFG).

### 1. Flow Matching (Continuous Optimal Transport Flow)
* **Straight Probability Path:**
  $$x_t = (1 - t) \cdot x_{\text{noise}} + t \cdot x_{\text{data}}, \quad t \in [0, 1]$$
* **Target Velocity Field:**
  $$u_t(x_t \mid x_{\text{noise}}, x_{\text{data}}) = \frac{d}{dt} x_t = x_{\text{data}} - x_{\text{noise}}$$
* **Velocity Matching Loss:**
  $$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x_{\text{data}}, x_{\text{noise}}, t, y} \left[ \| v_\theta(x_t, t, y) - (x_{\text{data}} - x_{\text{noise}}) \|^2 \right]$$
* **ODE Sampling:**
  * **1st Order Euler:** $x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, y)$
  * **2nd Order Heun (Predictor-Corrector):**
    $$\tilde{x}_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, y)$$
    $$x_{t + \Delta t} = x_t + \frac{\Delta t}{2} \left[ v_\theta(x_t, t, y) + v_\theta(\tilde{x}_{t + \Delta t}, t + \Delta t, y) \right]$$

### 2. DDPM & DDIM (Discrete Diffusion Process)
* **Forward Schedule:**
  $$x_t = \sqrt{\bar{\alpha}_t} \cdot x_{\text{data}} + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
* **Noise Prediction Loss:**
  $$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{x_{\text{data}}, \epsilon, t, y} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 \right]$$
* **DDIM (Deterministic ODE Counterpart):**
  $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

### 3. Classifier-Free Guidance (CFG)
During inference, conditional and unconditional predictions are linearly combined with guidance weight $w \ge 1.0$:
$$\hat{\phi}_{\text{guided}}(x_t, t, y) = \phi_\theta(x_t, t, \emptyset) + w \cdot \left( \phi_\theta(x_t, t, y) - \phi_\theta(x_t, t, \emptyset) \right)$$

---

## ⚖️ Design Decisions & Justifications

| Decision | Choice Made | Scientific Justification |
|---|---|---|
| **Backbone** | **DiT (Diffusion Transformer)** | Scalable self-attention replacing convolutional U-Nets, matching frontier architectures (SD3, Flux). |
| **Modulation** | **AdaLN-Zero** | Zero-initialized gates ensure the block is an exact mathematical identity at step 0 ($f(x) = x$), preventing initial gradient instability. |
| **Activation** | **GELU (tanh approximation)** | Continuous non-zero curvature prevents dead neurons common in standard ReLU. |
| **Time Scale** | **Scale $t \times 1000$** | Continuous $t \in [0, 1]$ and DDPM $t \in [1, 1000]$ span identical harmonic frequency bands. |
| **Evaluation** | **NFE-aligned** | Normalizes compute: Heun (2 NFE/step) at 10 steps is compared directly against Euler at 20 steps. |

---

## 📁 Repository Structure

```text
flow_matching_ddpm/
├── README.md                      # Mathematical specification & findings
├── pyproject.toml                 # uv package configuration
├── configs/
│   └── mnist_dit.yaml             # Reproducible MNIST benchmark configuration
├── data/                          # Cached datasets (git-ignored)
├── checkpoints/                   # Saved model weights (git-ignored)
├── outputs/                       # Generated grids & evaluation plots (git-ignored)
├── src/flow_matching_ddpm/
│   ├── __init__.py                # Package exports
│   ├── model.py                   # DiT backbone, PatchEmbed, AdaLNZeroBlock
│   ├── loss.py                    # DDPMLoss and FlowMatchingLoss (WIP)
│   ├── sampler.py                 # Euler, Heun, DDIM, Ancestral samplers (WIP)
│   ├── dataset.py                 # Normalization and DataLoader helpers (WIP)
│   ├── train.py                   # Training loop with EMA (WIP)
│   └── utils.py                   # Visualization and metrics (WIP)
└── tests/
    ├── __init__.py
    └── test_shapes.py             # Shape preservation & identity init tests
```

---

## 🚀 Quickstart & Verification

### 1. Installation
Clone the repository and install dependencies with `uv`:
```bash
git clone https://github.com/HamzA-1729/flow_matching_ddpm.git
cd flow_matching_ddpm
uv sync
```

### 2. Run Unit Tests
```bash
# Verify tensor shapes, patch unpatchify, and AdaLN-Zero identity initialization
uv run pytest tests/test_shapes.py -v
```

### 3. Training & Evaluation *(Coming in Stage 3 & 4)*
```bash
# Train Flow Matching on MNIST
uv run python -m flow_matching_ddpm.train --mode flow --config configs/mnist_dit.yaml

# Train DDPM on MNIST
uv run python -m flow_matching_ddpm.train --mode ddpm --config configs/mnist_dit.yaml

# Generate side-by-side comparative NFE sweep grid
uv run python -m flow_matching_ddpm.sampler --config configs/mnist_dit.yaml
```

---

## 📖 References
1. **Flow Matching for Generative Modeling** — Lipman et al., 2022 ([arXiv:2210.02747](https://arxiv.org/abs/2210.02747))
2. **Denoising Diffusion Probabilistic Models (DDPM)** — Ho et al., 2020 ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239))
3. **Denoising Diffusion Implicit Models (DDIM)** — Song et al., 2020 ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502))
4. **Scalable Diffusion Models with Transformers (DiT)** — Peebles & Xie, 2022 ([arXiv:2212.09748](https://arxiv.org/abs/2212.09748))

---

## 📜 License
Distributed under the [MIT License](LICENSE).
