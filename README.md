# Mamba-PFAN: Surgical Image De-Smoking

A lightweight, hybrid CNN-Mamba architecture for real-time surgical laparoscopy image de-smoking.

## Architecture
- **Generator**: `MambaPFAN` (Directional 2D Selective State Space Model replacing standard attention)
- **Discriminator**: 70×70 PatchGAN
- **Loss**: Vanilla GAN + L1 Reconstruction Loss ($\lambda_{L1} = 100$)

---

## 🚀 Running on Kaggle (Free 16GB GPU)

### 1. In your Kaggle Notebook:
```bash
# Clone the repository
!git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
%cd <REPO_NAME>

# Install dependencies
!pip install -q -r requirements.txt
```

### 2. Train and Evaluate:
```bash
# Attach your uploaded dataset in Kaggle (e.g. /kaggle/input/desmoking-dataset/composite)
python train_and_evaluate_mamba.py \
    --dataroot /kaggle/input/desmoking-dataset/composite \
    --checkpoints_dir /kaggle/working/checkpoints \
    --batch_size 8 \
    --num_threads 4 \
    --n_epochs 30
```

### 3. Download Model Weights & Images:
```bash
!zip -r /kaggle/working/mamba_results.zip /kaggle/working/checkpoints/mamba_Final
```

---

## 💻 Running Locally (Windows / Linux)
```bash
pip install -r requirements.txt
python train_and_evaluate_mamba.py --batch_size 4 --n_epochs 30
```
