# Flow Matching vs DDPM: Image Generation from Scratch (DiT Backbone)

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
├── configs/
│   ├── mnist_dit.yaml             # Reproducible MNIST benchmark configuration
│   └── cifar_dit.yaml             # Scaled CIFAR-10 configuration
├── src/flow_matching_ddpm/
│   ├── __init__.py
│   ├── model.py                   # DiT backbone, PatchEmbed, AdaLNZeroBlock
│   ├── loss.py                    # DDPMLoss and FlowMatchingLoss
│   ├── sampler.py                 # Ancestral, DDIM, Euler, and Heun samplers
│   ├── dataset.py                 # MNIST/CIFAR-10 normalization [-1, 1]
│   ├── train.py                   # Training loop, EMA tracking, grad clipping
│   └── utils.py                   # Seeding, visual grid exporter, classifier evaluator
└── tests/
    ├── __init__.py
    ├── test_shapes.py             # Reversibility, tensor shapes, AdaLN-Zero identity check
    └── test_overfit.py            # Deterministic 1-batch overfit test
```

---

## 5. Quickstart & Reproducibility

### 5.1. Run Core Unit Tests
```bash
# Verify tensor shapes, patch unpatchify, and AdaLN-Zero identity initialization
uv run pytest tests/test_shapes.py -v

# Run 1-batch overfit test (loss must drop by >90% within 100 steps on CPU)
uv run pytest tests/test_overfit.py -v
```

### 5.2. Training
```bash
# Train Flow Matching on MNIST
uv run python -m flow_matching_ddpm.train --mode flow --config configs/mnist_dit.yaml

# Train DDPM on MNIST
uv run python -m flow_matching_ddpm.train --mode ddpm --config configs/mnist_dit.yaml
```

### 5.3. Benchmark & Comparative Grid
```bash
# Generate side-by-side NFE comparison grid (5, 10, 20, 50, 100 NFE)
uv run python -m flow_matching_ddpm.sampler --config configs/mnist_dit.yaml
```

---

## 6. Engineering Log & Known Limitations

*(This section is updated dynamically throughout development)*

* **Milestone 1:** Repository scaffolded with `uv`, isolated dependencies, and mathematical specification committed.
* **Limitations:**
  1. The straight-line OT path is conditioned per-pair $(x_{\text{noise}}, x_{\text{data}})$; the marginal path remains non-linear, which is why 1-step generation requires distillation.
  2. Ancestral DDPM requires timestep respacing to remain mathematically coherent at low step counts ($N < 50$).

