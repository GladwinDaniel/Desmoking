`# 🎓 PROJECT REVIEW SHOWCASE & VIVA GUIDE

Welcome to the **Mamba-PFAN Project Review Showcase**. This directory is structured to give you immediate, one-click access to everything you need to show examiners, evaluators, and the review committee during your project presentation.

---

## 📂 Showcase Directory Map

```
PROJECT_REVIEW_SHOWCASE/
├── 01_IEEE_Paper_and_Slides/
│   ├── Mamba_PFAN_IEEE_Conference_Paper.pdf     <-- 6-Page IEEE Camera-Ready Conference Paper
│   ├── Project_Review_Presentation.pptx         <-- Official Presentation Slide Deck
│   ├── Cholec80_Desmoking_Literature_Comparison.md <-- Multi-Paper Benchmark Comparison
│   └── PAPER_REPORT.md                          <-- Comprehensive Technical & Formula Report
│
├── 02_Architecture_and_System_Diagrams/
│   ├── Mamba_PFAN_Generator_Architecture.png    <-- High-Res Generator & 4-Way CSS Diagram
│   ├── Mamba_PFAN_Class_Diagram.png             <-- Software Architecture Class Diagram
│   ├── Mamba_PFAN_UseCase_Diagram.png           <-- End-to-End System Use Case Diagram
│   ├── Mamba_PFAN_Sequence_Diagram.png          <-- Frame Processing Sequence Diagram
│   ├── Mamba_PFAN_State_Diagram.png             <-- Training & Inference State Transition Diagram
│   └── Mamba_PFAN_Activity_Diagram.png          <-- Frequency Decoupling Activity Diagram
│
├── 03_Real_Surgical_Desmoking_Outputs/
│   ├── Multi_Sample_Desmoking_Overview.png      <-- 4-Sample Panel [Smoky | Restored | Clean GT]
│   ├── 0001_correct_comparison.png              <-- Sample #0001 (Tool Shadow & Deep Mucosa)
│   ├── 0007_correct_comparison.png              <-- Sample #0007 (Heavy Electrosurgical Plume)
│   ├── 0015_correct_comparison.png              <-- Sample #0015 (Dense Cautery Smoke)
│   ├── 0038_correct_comparison.png              <-- Sample #0038 (Liver Capsule Boundary)
│   ├── 0044_correct_comparison.png              <-- Sample #0044 (High Glare & Smoke)
│   └── 0079_correct_comparison.png              <-- Sample #0079 (Fine Hepatic Microvasculature)
│
├── 04_Ablation_Study_Visual_Comparisons/
│   ├── Ablation_6Column_Strip_0007.png          <-- 6-Column Strip [Smoky | 1W | 2WH | 2WV | 4W | GT]
│   ├── ablation_comparison_0001.png             <-- Frame #0001 6-Way Directional Strip
│   ├── ablation_comparison_0007.png             <-- Frame #0007 6-Way Directional Strip
│   ├── ablation_comparison_0015.png             <-- Frame #0015 6-Way Directional Strip
│   └── ablation_comparison_0020.png             <-- Frame #0020 6-Way Directional Strip
│
└── 05_Benchmark_Metrics_and_Data/
    └── ablation_summary.csv                     <-- Standardized Metrics across all 103 Test Pairs
```

---

## 🎯 4-Step Review Presentation Flow

### Step 1: Introduce the Problem & Your Paper
1. Open [`01_IEEE_Paper_and_Slides/Project_Review_Presentation.pptx`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/01_IEEE_Paper_and_Slides/Project_Review_Presentation.pptx).
2. Show the IEEE Conference Paper: [`01_IEEE_Paper_and_Slides/Mamba_PFAN_IEEE_Conference_Paper.pdf`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/01_IEEE_Paper_and_Slides/Mamba_PFAN_IEEE_Conference_Paper.pdf).
3. **Key Talking Point**:
   > *"Existing Vision Transformers for laparoscopic smoke removal carry 26M to 132M parameters and quadratic complexity $\mathcal{O}(N^2)$, causing severe latency (>100 ms) that prevents live surgical video deployment. We propose Mamba-PFAN, which introduces 2D Selective State Space Models with linear $\mathcal{O}(N)$ complexity with only 387.9K parameters; real-time FPS remains to be measured on target hardware."*

### Step 2: Show the Architecture & Software Design
1. Open [`02_Architecture_and_System_Diagrams/Mamba_PFAN_Generator_Architecture.png`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/02_Architecture_and_System_Diagrams/Mamba_PFAN_Generator_Architecture.png).
2. Explain the **Dual-Branch Frequency Decoupling**:
   - **High-Frequency Stream**: ConvNeXt depthwise convolution ($7 \times 7$) capturing microvascular edges and metallic instrument boundaries.
   - **Global Low-Frequency Stream**: 2D Selective SSM (Mamba S6) with 4-Way Cross-Scanning to eliminate directional blind spots.
3. Show [`Mamba_PFAN_Class_Diagram.png`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/02_Architecture_and_System_Diagrams/Mamba_PFAN_Class_Diagram.png) and [`Mamba_PFAN_UseCase_Diagram.png`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/02_Architecture_and_System_Diagrams/Mamba_PFAN_UseCase_Diagram.png) for software engineering rigor.

### Step 3: Present Real Laparoscopic Output Results
1. Open [`03_Real_Surgical_Desmoking_Outputs/Multi_Sample_Desmoking_Overview.png`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/03_Real_Surgical_Desmoking_Outputs/Multi_Sample_Desmoking_Overview.png) (or individual frame comparisons `0007_correct_comparison.png`).
2. Point out:
   - **Left**: Input degraded by dense electrocautery smoke plume.
   - **Center**: Mamba-PFAN restoration recovering liver parenchyma texture and metallic tool contours.
   - **Right**: Clean Ground Truth reference.

### Step 4: Prove the Innovation via the Ablation Study
1. Open [`04_Ablation_Study_Visual_Comparisons/Ablation_6Column_Strip_0007.png`](file:///e:/Projects/DeSmoking/PROJECT_REVIEW_SHOWCASE/04_Ablation_Study_Visual_Comparisons/Ablation_6Column_Strip_0007.png).
2. Explain the **Scanning Direction Matrix**:
   - *1-Way Forward (22.30 dB)* leaves vertical streaking artifacts.
   - *Vertical 2-Way (22.90 dB)* outperforms Horizontal 2-Way (21.89 dB) because surgical cautery smoke naturally billows **upward** due to thermal buoyancy in $CO_2$ pneumoperitoneum.
   - *4-Way Cross-Scan (24.58 dB, 0.8685 SSIM at 50 epochs)* achieves a **+2.28 dB PSNR boost and 42.3% MSE reduction** over the one-way ablation. The final 200-epoch model reaches **25.71 dB PSNR and 0.8829 SSIM**.

---

## 🛡️ Quick Viva Defense Cheat-Sheet

* **Q: "Why does baseline PFAN have higher PSNR (27.39 dB) than Mamba-PFAN (25.71 dB)?"**
  * **Answer**: *"At a matched 200-epoch budget, PFAN reaches 27.39 dB PSNR and 0.9159 SSIM, while Mamba-PFAN reaches 25.71 dB PSNR and 0.8829 SSIM. Mamba-PFAN is therefore lower in absolute reconstruction quality, but it uses 38.4% fewer generator parameters and has linear theoretical scan complexity. We present this as an accuracy-efficiency trade-off, not as proof that lower SSIM is caused only by computational cost. Hardware FPS and latency still require measurement."*

* **Q: "Why didn't you use a bigger Transformer like Restormer or Dehamer?"**
  * **Answer**: *"Restormer has 26M parameters and Dehamer has 132M parameters, while Mamba-PFAN has 387,917. This makes Mamba-PFAN a promising efficiency-oriented alternative, but latency, memory, and FPS must be measured on identical target hardware before making clinical deployment claims."*
