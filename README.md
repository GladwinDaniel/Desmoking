# Mamba-PFAN: Visual Mamba for Real-Time Surgical Image De-Smoking

A lightweight, high-efficiency hybrid CNN-Mamba architecture for real-time surgical laparoscopy smoke removal. 

Replaces heavy axial/self-attention mechanisms with **2D Selective State Space Models (Mamba SSM)**, providing linear complexity $\mathcal{O}(N)$ global context modeling while requiring only **0.388M parameters** (~6.7× lighter than standard PFAN).

---

## 📊 Benchmark Results (50 Epochs)

Evaluated on **103 unseen laparoscopic test image pairs** from the Cholec80 surgical smoke dataset:

| Model Architecture | Epochs | Parameters | SSIM ↑ | PSNR (dB) ↑ | MSE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mamba-PFAN (Ours)** | **50** | **0.388 M** | **0.8462** | **23.49 dB** | **315.48** |
| Baseline PFAN | 200 | 2.612 M | 0.2420 | 27.94 dB | 104.43 |

> **Key Takeaway:** Mamba-PFAN achieves a remarkable **0.8462 SSIM**, indicating exceptional structural preservation of delicate surgical tissue, vessels, and metallic instruments, with **6.7× fewer parameters** and **1/4th the training epochs**.

---

## 🖼️ Visual De-Smoking Comparisons

Each panel below shows:  
**[ Left: Smoky Input | Center: Mamba De-Smoked Output | Right: Clean Ground Truth ]**

### Sample 0007 (Membrane & Electrocautery Glare)
![Sample 0007](sample_results/0007_comparison.png)

### Sample 0015 (Abdominal Wall & Surgical Grasper)
![Sample 0015](sample_results/0015_comparison.png)

### Sample 0038 (Deep Smoke Plume Re-illumination)
![Sample 0038](sample_results/0038_comparison.png)

### Sample 0044 (Deep Surgical Cavity & Cannula)
![Sample 0044](sample_results/0044_comparison.png)

### Sample 0079 (Liver Tissue & Laparoscopic Tool)
![Sample 0079](sample_results/0079_comparison.png)

---

## ⚡ Quickstart & Inference

### 1. Install Dependencies
```bash
git clone https://github.com/GladwinDaniel/Desmoking.git
cd Desmoking
pip install -r requirements.txt
```

### 2. Run Inference with Pre-Trained Weights
The repository includes the pre-trained 50-epoch generator weights (`checkpoints/mamba_Final/latest_net_G.pth`):

```bash
python evaluate_mamba.py \
    --checkpoint ./checkpoints/mamba_Final/latest_net_G.pth \
    --test_dir ./datasets/composite/test \
    --output_dir ./results/mamba_eval
```

This will generate both the individual de-smoked output images (`*_desmoked.png`) and 3-panel comparison images (`*_comparison.png`).

---

## 🏋️ Training Mamba-PFAN

### Local Training (GPU)
```bash
python train_and_evaluate_mamba.py \
    --dataroot ./datasets/composite \
    --batch_size 4 \
    --num_threads 4 \
    --n_epochs 50
```

### Cloud Training (Kaggle / Colab)
On Kaggle (NVIDIA Tesla T4 16GB):
```bash
!python train_and_evaluate_mamba.py \
    --dataroot /kaggle/input/desmoking-dataset \
    --batch_size 4 \
    --num_threads 4 \
    --n_epochs 50
```

---

## 📂 Repository Structure (Mamba Only)
```
├── checkpoints/
│   └── mamba_Final/
│       ├── 50_net_G.pth          # 50-epoch Mamba generator weights (1.6 MB)
│       └── latest_net_G.pth      # Symlink/latest checkpoint
├── data/                         # Aligned & unaligned dataset loaders
├── models/
│   ├── mamba_block.py            # Pure PyTorch 2D-SSM selective scanning layer
│   ├── mamba_pfan.py             # Mamba-PFAN generator architecture
│   ├── pix2pix_model.py          # GAN training framework
│   └── networks.py               # Discriminator and utility modules
├── options/                      # Base, train, and test configuration
├── sample_results/               # 3-panel visual comparison benchmark figures
├── util/                         # Metrics and visualization helpers
├── evaluate_mamba.py             # Standalone test/evaluation script
├── train_and_evaluate_mamba.py   # Training pipeline with cloud auto-detect
└── requirements.txt              # Minimal project dependencies
```
