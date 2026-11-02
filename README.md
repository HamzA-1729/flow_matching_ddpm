# Flow Matching vs DDPM: Image Generation from Scratch (DiT Backbone)

[![CI](https://github.com/HamzA-1729/flow_matching_ddpm/actions/workflows/ci.yml/badge.svg)](https://github.com/HamzA-1729/flow_matching_ddpm/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status: WIP](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)

*A clean, first-principles implementation and comparative evaluation of Continuous-Time Optimal Transport Flow Matching against Denoising Diffusion Probabilistic Models (DDPM) using a shared Patch Diffusion Transformer (DiT) backbone.*

> **Status: work in progress.** The DiT backbone and its tests are done. Losses, samplers, training and benchmarks are being built; sections marked *(planned)* describe the target design, not code that runs today.

## Roadmap

- [x] DiT backbone with AdaLN-Zero, 2D sin-cos positions, null-token CFG (`model.py`)
- [x] Shape, identity-init and single-batch overfit tests
- [ ] Flow Matching and DDPM losses (`loss.py`)
- [ ] Euler, Heun, DDIM and respaced ancestral samplers (`sampler.py`)
- [ ] MNIST data pipeline and training loop with EMA (`dataset.py`, `train.py`)
- [ ] NFE benchmark: 5 / 10 / 20 / 50 / 100 NFE comparison grid
- [ ] CIFAR-10 config and results

---

## 1. Problem, Approach, and Core Hypothesis

Generative modeling transforms unstructured random Gaussian noise into coherent, high-dimensional data samples. For years, **Denoising Diffusion Probabilistic Models (DDPM)** have dominated this space by framing image synthesis as a reverse Brownian diffusion process, gradually removing Gaussian noise across hundreds of discrete, stochastic steps. However, because diffusion paths are inherently curved and Brownian, truncating the sampling schedule to few steps leads to severe trajectory drift and blurriness unless complex variance respacing or distillation is employed.

In this project, we implement and evaluate **Continuous-Time Flow Matching (OT-Flow)** against DDPM. Flow Matching replaces curved stochastic diffusion trajectories with straight-line probability paths connecting Gaussian noise directly to clean data via Ordinary Differential Equations (ODEs). 

To ensure a 100% fair scientific comparison (same capacity, data, and compute budget), both paradigms are trained on the exact same architecture built from scratch in PyTorch: a **Patch Diffusion Transformer (DiT)** equipped with **AdaLN-Zero** conditioning and **Classifier-Free Guidance (CFG)**. We evaluate both models under identical initial noise latents across matching **Number of Function Evaluations (NFE)** $\in [5, 10, 20, 50, 100]$ using Euler and Heun ODE solvers, DDIM, and respaced ancestral DDPM.

---

## 2. Mathematical Formulations & Symbol Dictionary

To eliminate ambiguity across paradigms, we strictly adopt the notation:
* $x_{\text{data}} \in \mathbb{R}^{C \times H \times W}$: Clean data sample drawn from empirical data distribution $q(x_{\text{data}})$.
* $x_{\text{noise}} \sim \mathcal{N}(0, \mathbf{I})$: Standard Gaussian noise tensor.
* $t$: Continuous or discrete time conditioning variable.
* $y \in \{0, \dots, K-1\}$: Class condition label, where $y = \emptyset$ (index $K$) denotes the unconditional null token.

---

### 2.1. Flow Matching (Continuous Optimal Transport Flow)

#### Probability Path
We define a time-dependent linear interpolation between pure noise at $t=0$ and clean image data at $t=1$:
$$x_t = (1 - t) \cdot x_{\text{noise}} + t \cdot x_{\text{data}}, \quad t \in [0, 1]$$

#### Target Vector Field
Differentiating $x_t$ with respect to time $t$ yields the constant conditional velocity:
$$u_t(x_t \mid x_{\text{noise}}, x_{\text{data}}) = \frac{d}{dt} x_t = x_{\text{data}} - x_{\text{noise}}$$

> [!NOTE]
> **Marginal vs. Conditional Flow:** While each pairwise path between a specific $x_{\text{noise}}$ and $x_{\text{data}}$ is a straight line, the learned marginal vector field $v_\theta(x, t) = \mathbb{E}[u_t \mid x_t = x]$ is a non-linear mixture over the entire dataset. This is why multi-step numerical ODE integration is still required.

#### Training Objective (Velocity Matching Loss)
The neural network $v_\theta(x_t, t, y)$ learns to predict this velocity vector field:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x_{\text{data}}, x_{\text{noise}}, t \sim \mathcal{U}[0, 1], y} \left[ \| v_\theta(x_t, t, y) - (x_{\text{data}} - x_{\text{noise}}) \|^2 \right]$$

#### ODE Numerical Solvers (Sampling)
Given initial noise $x_0 \sim \mathcal{N}(0, \mathbf{I})$ at $t=0$, we integrate forward to $t=1$ with step size $\Delta t = \frac{1}{N}$:
* **1st Order Euler Integrator (1 NFE per step):**
  $$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, y)$$
* **2nd Order Heun Integrator (Predictor-Corrector, 2 NFE per step):**
  $$\tilde{x}_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, y)$$
  $$x_{t + \Delta t} = x_t + \frac{\Delta t}{2} \left[ v_\theta(x_t, t, y) + v_\theta(\tilde{x}_{t + \Delta t}, t + \Delta t, y) \right]$$

---

### 2.2. DDPM & DDIM (Discrete Diffusion Process)

#### Forward Noising Process
Given a linear variance schedule $\beta_1, \dots, \beta_T$ (where $T=1000, \beta_1=10^{-4}, \beta_T=0.02$):
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_{\text{data}} + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

#### Training Objective (Noise Prediction Loss)
The network $\epsilon_\theta(x_t, t, y)$ predicts the injected noise:
$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{x_{\text{data}}, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t \sim \mathcal{U}\{1, \dots, T\}, y} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 \right]$$

#### Reverse Sampling Algorithms
1. **Ancestral Sampling (Stochastic SDE):**
   $$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, y) \right)$$
   $$x_{t-1} = \tilde{\mu}_t + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I}) \text{ for } t > 1 \text{ (else } z=0)$$
   where posterior variance $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$.
   *To sample with $S < T$ steps, the schedule must be respaced according to Improved DDPM (Nichol & Dhariwal).*
2. **DDIM (Deterministic ODE counterpart):**
   Predict clean data estimate $\hat{x}_{\text{data}} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$:
   $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_{\text{data}} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t, y)$$

---

### 2.3. Classifier-Free Guidance (CFG)
During training, conditioning labels $y$ are randomly dropped to null token $\emptyset$ with probability $p_{\text{drop}} = 0.10$.
At inference time, with guidance scale $w \ge 1.0$, model evaluations are combined:
$$\hat{\phi}_{\text{guided}}(x_t, t, y) = \phi_\theta(x_t, t, \emptyset) + w \cdot \left( \phi_\theta(x_t, t, y) - \phi_\theta(x_t, t, \emptyset) \right)$$
*(where $\phi$ is $\epsilon$ for DDPM and $v$ for Flow Matching).*

---

### 2.4. Shared Backbone: Diffusion Transformer (DiT) with AdaLN-Zero

```
Input Image (B, C, H, W)
       │
       ▼
[ Patch Embedding: Conv2d(stride=p) ]  ──► Tokens: (B, N, D) + 2D Positional Encoding
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
Unpatchify ──► Output Tensor: (B, C, H, W)
```

---

## 3. Architectural Decisions & Justifications

| Decision | Choice Made | Scientific / Mathematical Justification |
|---|---|---|
| **Backbone Family** | **Patch Diffusion Transformer (DiT)** | Replaces legacy U-Net convolutions with scalable multi-head self-attention, matching frontier architectures (SD3, Flux, Sora). |
| **Conditioning Scheme** | **AdaLN-Zero** | Modulates normalized token activations dynamically via scale/shift $(\gamma, \beta)$ and gate $\alpha$. Initializing the modulation layer to exact zero ensures the block is an exact mathematical identity at step 0 ($f(x) = x$), preventing initial gradient shocks. |
| **Activation Function** | **GELU (tanh approximation)** | Provides continuous, non-zero curvature everywhere, avoiding the hard dead-neuron regime of standard ReLU. |
| **Optimizer** | **AdamW ($10^{-2}$ weight decay)** | Decouples $L_2$ weight regularization from gradient moving moments, preventing weight divergence during prolonged training. |
| **Learning Rate Schedule** | **Linear Warmup + Cosine Decay** | 250-step linear warmup builds stable AdamW momentum buffers before ramping to maximum learning rate ($\eta_{\text{max}} = 3 \times 10^{-4}$); cosine annealing smoothly guides parameters to the loss minimum. |
| **Gradient Clipping** | **Max Norm = 1.0** | Tames occasional gradient spikes occurring at boundary time steps ($t \to 0$ or $t \to 1$). |
| **Time Scale Normalization** | **Scale $t \times 1000$** | Ensures Flow Matching continuous $t \in [0, 1]$ and DDPM discrete timesteps $t \in [1, 1000]$ span identical harmonic frequency bands in the sinusoidal embedding. |
| **Evaluation Standard** | **NFE-aligned (Number of Function Evaluations)** | Normalizes compute budget: Heun (2 NFE/step) at 10 steps is evaluated directly against Euler at 20 steps and DDIM at 20 steps. |

---

## 4. Directory Structure

```text
flow_matching_ddpm/
├── README.md                      # Mathematical specification & findings log
├── pyproject.toml                 # uv package configuration
├── .github/workflows/ci.yml       # ruff + pytest on every push/PR
├── configs/
│   └── mnist_dit.yaml             # Reproducible MNIST benchmark configuration
├── src/flow_matching_ddpm/
│   ├── __init__.py
│   ├── model.py                   # DiT backbone, PatchEmbed, AdaLNZeroBlock
│   ├── loss.py                    # (planned) DDPMLoss and FlowMatchingLoss
│   ├── sampler.py                 # (planned) Ancestral, DDIM, Euler, and Heun samplers
│   ├── dataset.py                 # (planned) MNIST/CIFAR-10 normalization [-1, 1]
│   ├── train.py                   # (planned) Training loop, EMA tracking, grad clipping
│   └── utils.py                   # (planned) Seeding, visual grid exporter, classifier evaluator
└── tests/
    ├── __init__.py
    ├── test_shapes.py             # Reversibility, tensor shapes, AdaLN-Zero identity check
    └── test_overfit.py            # Deterministic 1-batch overfit test
```

---

## 5. Quickstart & Reproducibility

### 5.1. Install
```bash
git clone https://github.com/HamzA-1729/flow_matching_ddpm.git && cd flow_matching_ddpm
uv sync
```

### 5.2. Run the tests
```bash
uv run pytest -v          # shapes, AdaLN-Zero identity init, single-batch overfit
uv run ruff check .       # lint
```

### 5.3. Training and benchmarking *(planned)*
```bash
uv run python -m flow_matching_ddpm.train --mode flow --config configs/mnist_dit.yaml
uv run python -m flow_matching_ddpm.train --mode ddpm --config configs/mnist_dit.yaml
uv run python -m flow_matching_ddpm.sampler --config configs/mnist_dit.yaml
```

---

## 6. Results *(planned)*

Sample grids and NFE-vs-quality comparison (Flow Matching vs DDPM/DDIM at 5-100 NFE) will be added here once training is complete.

---

## 7. Known Limitations

1. The straight-line OT path is conditioned per-pair $(x_{\text{noise}}, x_{\text{data}})$; the marginal path remains non-linear, which is why 1-step generation requires distillation.
2. Ancestral DDPM requires timestep respacing to remain mathematically coherent at low step counts ($N < 50$).

---

## 8. References

1. Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
2. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
3. Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
4. Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, ICML 2021.
5. Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)*, ICCV 2023.
6. Ho & Salimans, *Classifier-Free Diffusion Guidance*, 2022.

## 9. License & Citation

Released under the [MIT License](LICENSE). If you use this work, see [CITATION.cff](CITATION.cff). Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
