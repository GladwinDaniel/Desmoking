import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

paper_dir = Path("e:/Projects/DeSmoking/ieee_conference_paper")
fig_dir = paper_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate Mamba-PFAN Architecture Diagram (Figure 1)
print("Generating Mamba-PFAN Architecture Diagram...")
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# Styles
box_style = dict(boxstyle="round,pad=0.3", edgecolor="#2B3A42", linewidth=1.5)
arrow_props = dict(arrowstyle="->", color="#1F2937", lw=1.8, mutation_scale=14)

# Background panels
ax.add_patch(patches.Rectangle((2.8, 0.4), 8.4, 6.2, fill=True, color="#F8FAFC", ec="#CBD5E1", lw=1.5, ls="--", zorder=0))
ax.text(7.0, 6.3, "Mamba-PFAN Generator Architecture (387.9K Parameters, O(N) Complexity)", 
        ha="center", va="center", fontsize=11, fontweight="bold", color="#1E293B", zorder=1)

# Input
ax.text(1.2, 3.5, "Input Image\n$I_{smoke}$\n$(3 \\times 256 \\times 256)$", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.4", fc="#E2E8F0", ec="#64748B", lw=1.5))

# Stem Conv
ax.text(2.6, 3.5, "Stem\n$1 \\times 1$ Conv\n$(C=64)$", 
        ha="center", va="center", fontsize=8.5, color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#D97706", lw=1.5))

ax.annotate("", xy=(2.1, 3.5), xytext=(1.9, 3.5), arrowprops=arrow_props)
ax.annotate("", xy=(3.3, 3.5), xytext=(3.1, 3.5), arrowprops=arrow_props)

# Split into two branches
ax.annotate("", xy=(3.7, 4.8), xytext=(3.4, 3.5), arrowprops=arrow_props)
ax.annotate("", xy=(3.7, 2.2), xytext=(3.4, 3.5), arrowprops=arrow_props)

# Branch 1: High Frequency ConvNeXt Block
ax.add_patch(patches.Rectangle((3.7, 4.0), 3.8, 1.6, fill=True, color="#EFF6FF", ec="#3B82F6", lw=1.5, zorder=1))
ax.text(5.6, 5.3, "High-Frequency Stream: ConvNeXt-MBI", ha="center", va="center", fontsize=9, fontweight="bold", color="#1D4ED8")
ax.text(5.6, 4.55, "• $7 \\times 7$ Depthwise Conv (Local Edges)\n• LayerNorm + Pointwise MLP ($4 \\times C$)\n• Residual Connection $\\mathbf{X}_{HF} = \\mathbf{X} + \\text{Block}(\\mathbf{X})$", 
        ha="center", va="center", fontsize=7.8, color="#1E3A8A")

# Branch 2: Low Frequency Mamba-SSM Block
ax.add_patch(patches.Rectangle((3.7, 1.0), 3.8, 2.4, fill=True, color="#F0FDF4", ec="#16A34A", lw=1.5, zorder=1))
ax.text(5.6, 3.1, "Global Low-Freq Stream: 2D-SSM (Ours)", ha="center", va="center", fontsize=9, fontweight="bold", color="#15803D")
ax.text(5.6, 2.1, "• Dual-branch Linear Projection\n• Directional Depthwise State Dynamics\n• Input-dependent modulation $(\\mathbf{B}, \\mathbf{C}, \\Delta)$\n• Continuous decay $\\sigma(\\mathbf{A}_{decay})$ gating\n• Feedthrough skip connection $\\mathbf{D}$", 
        ha="center", va="center", fontsize=7.8, color="#14532D")

# 4-Way Scanning Sub-box
ax.add_patch(patches.Rectangle((7.8, 1.2), 3.1, 4.4, fill=True, color="#FAF5FF", ec="#9333EA", lw=1.5, zorder=1))
ax.text(9.35, 5.3, "4-Way Cross-Scanning (4-Way CSS)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7E22CE")
ax.text(9.35, 4.7, "Horizontal Fwd ($L \\rightarrow R$)", ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.2", fc="#F3E8FF", ec="#C084FC"))
ax.text(9.35, 4.05, "Horizontal Bwd ($R \\rightarrow L$)", ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.2", fc="#F3E8FF", ec="#C084FC"))
ax.text(9.35, 3.4, "Vertical Fwd ($T \\rightarrow B$)", ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.2", fc="#F3E8FF", ec="#C084FC"))
ax.text(9.35, 2.75, "Vertical Bwd ($B \\rightarrow T$)", ha="center", va="center", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.2", fc="#F3E8FF", ec="#C084FC"))
ax.text(9.35, 1.8, "$\\mathbf{Y}_{SSM} = \\sum_{d=1}^4 \\mathbf{S}_d \\odot \\mathbf{C} + \\mathbf{D} \\mathbf{X}$", ha="center", va="center", fontsize=7.5, color="#581C87")

# Arrows from branches to fusion
ax.annotate("", xy=(7.8, 4.8), xytext=(7.5, 4.8), arrowprops=arrow_props)
ax.annotate("", xy=(7.8, 2.2), xytext=(7.5, 2.2), arrowprops=arrow_props)

# Fusion Block
ax.text(11.4, 3.5, "Frequency Fusion\n& Channel Gating\n$(C=64)$", 
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#D97706", lw=1.5))
ax.annotate("", xy=(11.0, 3.5), xytext=(10.9, 3.5), arrowprops=arrow_props)

# Final Output Conv
ax.text(12.7, 3.5, "Restored Image\n$I_{clean}$\n$(3 \\times 256 \\times 256)$", 
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.4", fc="#D1FAE5", ec="#059669", lw=1.5))
ax.annotate("", xy=(12.2, 3.5), xytext=(11.9, 3.5), arrowprops=arrow_props)

# PatchGAN Discriminator (below)
ax.text(12.7, 1.1, "PatchGAN\nDiscriminator\n(Local Realism)", 
        ha="center", va="center", fontsize=8, color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEE2E2", ec="#DC2626", lw=1.2))
ax.annotate("", xy=(12.7, 1.8), xytext=(12.7, 2.8), arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5, ls=":"))

plt.tight_layout()
arch_path = fig_dir / "mamba_pfan_architecture.png"
plt.savefig(arch_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {arch_path}")

# 2. Copy Real Visual Comparison Figure from Ablation Results (Figure 2)
print("Copying Real Ablation Visual Comparison Figure...")
ablation_src = Path("e:/Projects/DeSmoking/ablation_results/visual_comparisons/ablation_comparison_0007.png")
if ablation_src.exists():
    im = Image.open(ablation_src)
    im.save(fig_dir / "ablation_visual_comparison.png")
    print(f"Saved: {fig_dir / 'ablation_visual_comparison.png'}")

# 3. Create Multi-Sample Laparoscopic De-Smoking Results Figure (Figure 3)
print("Creating Multi-Sample De-Smoking Figure (Cholec80 Real Results)...")
samples = ["0001", "0007", "0015", "0038"]
sample_rows = []

for sid in samples:
    comp_file = Path(f"e:/Projects/DeSmoking/real_test_visuals/{sid}_correct_comparison.png")
    if not comp_file.exists():
        comp_file = Path(f"e:/Projects/DeSmoking/mamba_repo/sample_results/{sid}_comparison.png")
    if comp_file.exists():
        sample_rows.append((sid, Image.open(comp_file)))

if sample_rows:
    fig, axes = plt.subplots(len(sample_rows), 1, figsize=(10, 2.8 * len(sample_rows)), dpi=250)
    if len(sample_rows) == 1:
        axes = [axes]
    for idx, (sid, img) in enumerate(sample_rows):
        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(f"Sample #{sid}: [Left: Smoky Input | Center: Mamba-PFAN Restored | Right: Ground Truth]", 
                            fontsize=9, fontweight="bold", pad=4)
    plt.tight_layout()
    results_path = fig_dir / "desmoking_visual_results.png"
    plt.savefig(results_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved: {results_path}")

print("All figures successfully created and organized in ieee_conference_paper/figures/!")
