# An Empirical Framework for Deep Learning Architectures: Tabular, Vision, and Sequential Landscapes

This repository contains a comprehensive, experimental sandbox designed to systematically implement, optimize, and analyze diverse deep learning architectures from scratch. The codebase spans three primary paradigms: Multilayer Perceptrons (MLPs) for tabular objectives, Convolutional Neural Networks (CNNs) for computer vision tasks, and Recurrent/Attention-driven networks for continuous time-series forecasting.

The primary objective is to isolate and evaluate the empirical impacts of hyperparameter variations, regularization strategies, and topological modifications on model convergence and generalization capacity.

---
### Important note:
For the plots to be displayed properly you have to download HW3.ipynb and run it within vscode. You can't see the interactive plots online.
---

## 1. Core Framework & Environment Selection

All modules are engineered exclusively within **PyTorch**. This unified framework choice is dictated by three primary architectural and engineering constraints:

* **Hardware Acceleration (Apple Silicon & Linux Parallelism):** PyTorch provides native, out-of-the-box support for Apple's Metal Performance Shaders (`mps`) backend alongside standard NVIDIA `cuda` acceleration. This enables high-performance local compilation and execution across disparate Unix environments without platform-specific configuration overhead or dependency version-locking.
* **Dynamic Computation Graphs:** The framework utilizes an eager execution paradigm (Autograd). This dynamic execution setup is critical for running procedural sweeps, injecting hooks, inspecting intermediate layer tensors, and parameterizing network topologies on the fly.
* **Pythonic Design Execution:** Native compatibility with standard Python control flows simplifies debugging and allows for direct integration with scientific computing abstractions like NumPy, Scikit-Learn, and Plotly.

Target devices are assigned programmatically using the following uniform runtime initialization pattern:

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

```

---

## 2. Repository Data Management Strategy

To maintain a lightweight version-control footprint and adhere to software engineering best practices, this repository enforces a zero-binary tracking policy. No raw datasets, compiled databases, or localized telemetry outputs are tracked via source control.

Data acquisition pipelines are completely automated. Upon initial execution, localized initialization scripts query local directories for target sub-folders. If missing, the environment dynamically orchestrates the downstream download, integrity verification, extraction, and tensor preprocessing protocols:

* **Tabular Pipeline:** Dynamically imported via `sklearn.datasets.fetch_california_housing`.
* **Vision Pipeline:** Managed directly through `torchvision.datasets`.
* **Temporal Pipeline:** Automatically fetched via secure remote zip endpoints and extracted locally.

### Source Control Configuration

```text
# Local data folders and diagnostic outputs isolated from source control
data/
*.json
*.html

```

---

## 3. Module I: Multilayer Perceptrons (MLP) & Hyperparameter Tuning

This module establishes an experimental sandbox over continuous and discrete optimization landscapes using a single source dataset.

### Dataset and Task Engineering

The pipeline utilizes the **California Housing Dataset** (8 continuous demographic and geographic features). To evaluate how identical architectural choices adapt to fundamentally distinct objective landscapes, the dataset is engineered to support two simultaneous tasks:

1. **Regression (Primary):** Predicting the continuous variable `MedHouseVal` via Mean Squared Error (MSE) loss.
2. **Binary Classification (Derived):** Classifying records as either `Above Median` (1) or `Below Median` (0) relative to global property valuations, optimized via Binary Cross-Entropy (BCE) loss.

### Baseline Topology

A parameterized helper function (`build_and_train_mlp`) programmatically constructs a standard anchor architecture:

* **Input Layer:** 8 Nodes (matching the input features).
* **Hidden Layer 1:** 64 Neurons + ReLU activation.
* **Hidden Layer 2:** 32 Neurons + ReLU activation.
* **Output Layer:** 1 Neuron (Linear for Regression; Sigmoid activation for Classification).

### Experimental Vectors Evaluated

The model topology reshapes itself dynamically to systematically benchmark the following configurations:

* **Optimization Variants:** Stochastic Gradient Descent (Standard vs. Classical Momentum) vs. Adam.
* **Hyperparameters:** Learning rate boundaries, Step Decay scheduling, and Batch Size distributions.
* **Topology Geometry:** Depth adjustments (scaling from 1 to 4 dense layers) and Width variations (narrow bottlenecks vs. wide representations).
* **Activation Dynamics:** Comparative evaluation of ReLU, LeakyReLU, Tanh, and Sigmoid functions alongside initialization configurations (Random, Xavier/Glorot, and He/Kaiming).
* **Regularization & Stability Control:** Structural integration of Batch Normalization, Dropout layers, Weight Decay ($L_1 / L_2$ penalties executed via forward hooks), Gradient Clipping thresholds, and Early Stopping patience.

---

## 4. Module II: Convolutional Neural Networks (CNN) & Transfer Learning

This module targets localized, translation-invariant feature extraction, expanding from baseline custom layers up to deep pretrained residual networks.

### Dataset and Pipeline Execution

The framework uses the **CIFAR-10** benchmark dataset (60,000 low-resolution 32x32 pixel RGB images spanning 10 mutually exclusive natural classes). The execution is bifurcated into two discrete preprocessing pipelines:

* **Native Pipeline (Bespoke Processing):** Implements channel-wise standardization. Advanced subsets introduce online stochastic geometric and photometric transformations (horizontal flips, variable rotations, reflective padding crops, and color jitter) to mitigate overfitting.
* **Upscaled Pipeline (Transfer Learning):** Dynamically interpolates and upscales spatial dimensions from 32x32 to 224x224 pixels while adjusting normalization constants to match ImageNet feature constraints.

### Evaluated Architectures

| Architecture Variant | Layer Stack & Spatial Operations | Dimensionality Transitions |
| --- | --- | --- |
| **Part A: Custom Baseline CNN** | 3 x Convolutional Blocks (Conv $\rightarrow$ BatchNorm $\rightarrow$ ReLU) punctuated by Max-Pooling (2x2, stride 2). | Input ($3 \times 32 \times 32$) $\rightarrow$ Channels: 32 $\rightarrow$ 64 $\rightarrow$ 128. Flattened to an 8,192-dimensional vector. |
| **Part B: Hyperparameter Sweep** | Procedural class (`HyperparameterCNN`) mutating kernel structures, strides, pooling variations, and depth scales. | Parameterized kernel filters (3x3, 5x5, 7x7); variable strides (1 to 3); Max vs. Average pooling selection. |
| **Part D: Pretrained ResNet18** | Frozen ResNet18 convolutional backbone (`param.requires_grad = False`). Linear projection head substituted. | Input upscaled to $3 \times 224 \times 224$. Original 1,000-class head stripped and re-initialized to a 10-way classifier. |

---

## 5. Module III: Sequence Modeling & Attention-Based Architectures

This module evaluates multi-step climate sequence forecasting, isolating performance limits between classic Recurrent Neural Networks and parallelized Multi-Head Self-Attention mechanisms.

### Dataset and Sequence Engineering

The module utilizes the **Jena Climate Dataset** (14 meteorological features sampled at 10-minute intervals). The raw temporal tracking matrices undergo standard engineering to enforce stability:

* **Subsampling:** The data is downsampled to 1-hour intervals to alleviate high-frequency noise and accelerate training velocity.
* **Temporal Isolation:** Splitting is strictly chronological (70% Train, 15% Validation, 15% Test) to prevent future-lookahead data leakage. Feature scaling transforms are fit strictly on the training subset.
* **Sliding Window Dimensions:** Input tensors $X$ are formatted to dimensions to forecast the future temperature point.

### Architectural Paradigms Compared

#### Paradigm A: Recurrent Architectures (`DynamicSeqModel`)

A unified class wraps alternative recurrent units (Vanilla RNN, LSTM, or GRU), mapping the final hidden state vector across historical sequence boundaries into a linear regression output. Hyperparameter evaluation tracks:

* **Context Horizon Boundaries:** Short-context tracking (6 hours) vs. broad macro-horizons (168 hours).
* **Representational Footprints:** Hidden capacities swept from narrow (8 units) to wide (128 units).
* **Hierarchical Layer Stacking:** Linear depth comparisons (1 layer vs. 3 stacked layers).
* **Directional Flow Analysis:** Unidirectional sequence processing vs. bidirectional feature extraction.
* **Regularization:** Inter-layer Dropout optimization ($p = 0.3$).

#### Paradigm B: Attention-Based Architectures (`TimeSeriesTransformer`)

Replaces recurrent loops completely with parallelized self-attention modules. The pipeline projects raw signals onto a distributed $d_{\text{model}}$ embedding space. Because attention matrices are inherently permutation-invariant, sequential chronology is explicitly restored by injecting deterministic, frequency-based sinusoidal positional vectors before the encoder phase:

$$\text{PE}_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

The final sequence step representation is sliced directly from the output attention maps and routed to a linear regression head.

---

## 6. Instrumentation, Analytics, and Core Insights

### Monitoring & Visualization

Every framework module integrates an analytical suite for logging performance metrics:

* **Wall-Clock Telemetry:** Precise tracking of computational runtimes per epoch using execution timers.
* **Consolidated Reporting:** Summary tables printed to standard output tracking minimized test loss, peak validation accuracy, and total training duration.
* **Interactive Dashboards:** Programmatic generation of responsive Plotly charts tracking optimization paths (Loss/Accuracy loops) featuring isolated dropdown filters to selectively evaluate model iterations.

### Summary of Empirical Insights

* **Vision Landscape:** Transfer learning pipelines leveraging frozen ResNet18 structures demonstrated superior generalization convergence rates compared to custom CNN architectures built from scratch, validating the utility of cross-domain feature initialization even when input fields require synthetic upscaling.
* **Sequence Landscape:** Gated Recurrent Units (specifically LSTMs) consistently out-performed the Transformer Encoder structures across mid-scale continuous climate horizons. This gap highlights the advantage of the strong sequential inductive biases native to recurrent tracking blocks when modeling smoothly varying continuous physical systems, contrasted against the optimization fragility of Post-Layer Normalization Transformers operating without complex learning rate warmups.
