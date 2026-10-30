# Comprehensive Literature Survey & Benchmark Comparison on the Cholec80 Laparoscopic De-Smoking Dataset

## 1. Overview of the Cholec80 Surgical Smoke Benchmark
In minimally invasive surgery (MIS), intraoperative paired image acquisition (capturing simultaneous smoky and smoke-free frames of the exact same surgical moment in vivo) is physically impossible. To train and evaluate supervised medical image restoration networks, the surgical computer vision community established the **Cholec80 Surgical Smoke Benchmark** (pioneered by Bolkar et al., ICIP 2018; standardized by Zhang et al., PRCV 2024; and expanded in LSD3K, Chang & Chen, 2024):
* **Source Dataset**: *Cholec80* (Twinanda et al., IEEE TMI 2017), consisting of 80 laparoscopic cholecystectomy videos performed by 13 clinical surgeons.
* **Ground-Truth Selection**: 660 pristine, smoke-free frames extracted across varying surgical phases, hepatic pathologies, electrosurgical glare, and organ orientations.
* **Physics-Based Smoke Synthesis**: Volumetric particle simulation in Blender rendering non-homogeneous smoke plumes, convective turbulence, light diffusion, and variable optical density across light, medium, and heavy smoke levels.
* **Standard Test Protocol**: Unseen test frames isolated at the surgical case level to benchmark restoration fidelity (PSNR, SSIM, CIEDE2000, and MSE).

---

## 2. Key Published Papers Using the Cholec80 Benchmark

| Paper & Citation | Authors & Venue | Core Architecture & Methodology | Key Datasets Used |
| :--- | :--- | :--- | :--- |
| **Progressive Frequency-Aware Network (PFAN)** [1] | J. Zhang, W. Huang, X. Liao, Q. Wang (*PRCV 2024 / arXiv:2312.12023*) | Dual-stream hybrid CNN-Transformer: CNN Multi-scale Bottleneck-Inverting (MBI) for high-frequency + Axial Attention Transformer (LAT) for low-frequency. | Cholec80 Synthetic Benchmark (660 frames) |
| **LSD3K: A Benchmark for Smoke Removal from Laparoscopic Surgery Images** [2] | W. Chang, H. Chen (*arXiv:2407.13132*) | Benchmark paper establishing LSD3K using 660 Cholec80 ground-truth frames + Blender smoke. Benchmarks GridDehazeNet, DehazeFormer, Restormer, Dehamer, AOD-Net, and DCP. | Cholec80 / C80 + CholecT50 |
| **Deep Smoke Removal from MIS Videos** [3] | S. Bolkar, C. Wang, F. A. Cheikh, S. Yildirim (*IEEE ICIP 2018*) | Pioneer of deep synthetic laparoscopic smoke synthesis using Cholec80 + Blender. Encoder-decoder CNN for surgical haze extraction. | Cholec80 + Blender Simulation |
| **DeSmoke-LAP** [4] | K. Sidorov et al. / O. Alirezaee et al. (*IEEE ISBI / MedIA*) | Unpaired image-to-image translation based on CycleGAN with dedicated dark-channel and inter-channel gradient loss functions. | Cholec80 / Robotic Laparoscopy |
| **Dual-Pathway Feature Fusion & Attention Network** [5] | Z. Chen, R. Liu, Y. Ding (*Comput. Biol. Med. 2022*) | Dual-branch CNN with spatial and channel attention mechanisms to decouple high/low frequency smoke components. | Synthetic Laparoscopic Pairs |
| **Mamba-PFAN (Proposed Work)** [6] | G. Daniel P, A. Kumaran, S. Srishreyan (*IEEE 2026*) | First Visual State Space Model (SSM) for surgical desmoking. Replaces quadratic/axial attention with 2D Selective SSM and Four-Way Cross-Scanning (4-Way CSS). | Cholec80 Test Set (103 unseen pairs) |

---

## 3. Master Multi-Paper Quantitative Comparison Table

The following table compiles reported quantitative results on Cholec80-derived surgical smoke benchmarks alongside our local results. Published values are not strictly head-to-head evidence because protocols, splits, and preprocessing may differ. on the Cholec80-derived surgical smoke dataset across published papers and our proposed Mamba-PFAN:

| Architecture / Model | Category | Parameters (#Params) | Complexity | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | Reference Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DCP (Dark Channel Prior)** | Physics Prior | — | Handcrafted | 18.46 – 27.63 | 0.5528 – 0.8637 | 35.99 | Zhang et al. [1], Chang et al. [2] |
| **AOD-Net** | Lightweight CNN | 0.02M (20K) | $\mathcal{O}(N)$ | 16.81 | 0.8249 | — | Chang & Chen [2] |
| **CycleGAN (U-Net)** | Unpaired GAN | 54.41M | $\mathcal{O}(N)$ | 28.74 | 0.7621 | 10.33 | Zhang et al. [1] |
| **CycleGAN (ResNet-6)** | Unpaired GAN | 7.84M | $\mathcal{O}(N)$ | 29.03 | 0.7826 | 9.58 | Zhang et al. [1] |
| **CycleGAN (ResNet-9)** | Unpaired GAN | 11.38M | $\mathcal{O}(N)$ | 29.09 | 0.7802 | 9.29 | Zhang et al. [1] |
| **Pix2Pix (U-Net)** | Paired cGAN | 54.41M | $\mathcal{O}(N)$ | 29.30 | 0.7073 | 8.81 | Zhang et al. [1] |
| **Pix2Pix (ResNet-6)** | Paired cGAN | 7.84M | $\mathcal{O}(N)$ | 29.82 | 0.8358 | 6.94 | Zhang et al. [1] |
| **Pix2Pix (ResNet-9)** | Paired cGAN | 11.38M | $\mathcal{O}(N)$ | 29.87 | 0.8417 | 6.70 | Zhang et al. [1] |
| **Pix2Pix (Uformer)** | Transformer cGAN | 85.61M | $\mathcal{O}(N^2)$ | 29.70 | 0.8026 | 8.06 | Zhang et al. [1] |
| **Restormer** | Transformer | 26.13M | Transposed Attn | 25.10 | 0.9344 | — | Chang & Chen [2] |
| **Dehamer** | Transformer | 132.45M | $\mathcal{O}(N^2)$ | 26.73 | 0.9263 | — | Chang & Chen [2] |
| **GridDehazeNet** | Multi-Scale CNN | 0.96M (960K) | $\mathcal{O}(N)$ | 32.45 | 0.9780 | — | Chang & Chen [2] |
| **DehazeFormer** | Transformer | 4.64M | Window Attn | 29.72 | 0.9665 | — | Chang & Chen [2] |
| **PFAN (w/o LAT Module)** | Ablated Hybrid | 0.09M (90K) | $\mathcal{O}(N)$ | 28.89 | 0.7857 | 10.13 | Zhang et al. [1] |
| **PFAN (Zhang et al. Published)** | Hybrid CNN-ViT | 0.629M (629K) | $\mathcal{O}(N^{1.5})$ | 30.49 | 0.9061 | 5.50 | Zhang et al. [1] |
| **PFAN (Standardized Test Split)** | Hybrid CNN-ViT | 0.629M (629K) | $\mathcal{O}(N^{1.5})$ | 27.39 | 0.9159 | 3.74 | Our Benchmark (Local Eval) |
| **Mamba-PFAN (1-Way SSM)** | Visual SSM | **0.388M (388K)** | $\mathcal{O}(N)$ | 22.30 | 0.8508 | 6.76 | Our Ablation Matrix |
| **Mamba-PFAN (2-Way Horiz SSM)**| Visual SSM | **0.388M (388K)** | $\mathcal{O}(N)$ | 21.89 | 0.8469 | 7.17 | Our Ablation Matrix |
| **Mamba-PFAN (2-Way Vert SSM)** | Visual SSM | **0.388M (388K)** | $\mathcal{O}(N)$ | 22.90 | 0.8469 | 6.45 | Our Ablation Matrix |
| **Mamba-PFAN (4-Way CSS, Ours, 200 ep)** | **Visual SSM** | **0.388M (388K)** | **Linear $\mathcal{O}(N)$** | **25.71** | **0.8829** | **4.93** | **Our standardized local evaluation** |

---

## 4. Key Comparative Findings & Insights

### 1. The Breakdown of Traditional Physics Priors (DCP)
Across both Zhang et al. and Chang et al., Dark Channel Prior (DCP) exhibits severe failure modes (CIEDE2000 of 35.99 and PSNR down to 18.46 dB). DCP relies on the outdoor atmospheric scattering model:
$$I(x) = J(x)t(x) + A(1 - t(x))$$
which assumes uniform infinite-distance atmospheric haze and homogenous ambient daylight. In an enclosed laparoscopic cavity, lighting emanates from point-source fiber optic cables, tissue surfaces are glossy and moist (causing specular reflections), and smoke density fluctuates locally due to cautery plume discharge. DCP misinterprets high-frequency specular reflections as smoke and dark red liver parenchyma as shadow, causing severe darkening and unnatural chrominance distortion.

### 2. Parameter Explosion of General Restoration Models
General image restoration and dehazing models evaluated in the literature carry enormous parameter overheads:
* **Dehamer**: 132.45M parameters
* **Pix2Pix-Uformer**: 85.61M parameters
* **CycleGAN / Pix2Pix (U-Net)**: 54.41M parameters
* **Restormer**: 26.13M parameters

In a clinical operating theater, endoscopy video towers run at 1080p / 4K resolution at 30 to 60 frames per second. Models with $>25\text{M}$ parameters exceed the memory bandwidth and compute budgets of embedded medical edge hardware (e.g., NVIDIA Jetson AGX Orin or integrated endoscopy image processing units), incurring latencies $>50\text{–}100\text{ ms}$ that induce surgical disorientation and hand-eye coordination lag.

---

## 5. Interpreting the Accuracy-Efficiency Trade-off

Mamba-PFAN should not be described as the highest-quality model across all metrics. In the matched local evaluation, PFAN achieves higher PSNR and SSIM, while Mamba-PFAN is substantially smaller and has linear theoretical scan complexity:

- PFAN: 629,421 generator parameters, 27.39 dB PSNR, 0.9159 SSIM.
- Mamba-PFAN: 387,917 generator parameters, 25.71 dB PSNR, 0.8829 SSIM.
- Mamba-PFAN is 38.4% smaller, with a PSNR gap of 1.68 dB and an SSIM gap of 0.0330.

The evidence supports an accuracy-efficiency trade-off. The lower SSIM should not be attributed solely to computational cost without controlled capacity and hardware experiments. It is more accurate to state that replacing the axial-attention pathway reduces parameters and theoretical complexity while retaining useful structural restoration quality.

## 6. Limitations and Reproducibility

The multi-paper values above are drawn from their cited publications and should be treated as contextual literature comparisons, not direct re-rankings. A publication-ready study should reproduce the strongest baselines with the same split, preprocessing, resolution, training budget, random seeds, and evaluation code. In addition, latency, FPS, FLOPs, and memory must be measured on the same hardware before making real-time or edge-deployment claims.

The 200-epoch Mamba-PFAN result was evaluated on 103 test pairs and improved over the earlier 50-epoch result from 24.58 to 25.71 dB PSNR and from 0.8685 to 0.8829 SSIM.
