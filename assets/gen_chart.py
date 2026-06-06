#!/usr/bin/env python3
"""Generate hero chart for inference-kernel-cookbook README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

recipes = [
    ("Naive Attention\n(materialized N×N)", 996.18, "#e74c3c"),
    ("Flash Attention\n(tiled, online softmax)", 12.28, "#27ae60"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [2, 1.2]})

# Left: latency comparison
ax = axes[0]
labels = [r[0] for r in recipes]
values = [r[1] for r in recipes]
colors = [r[2] for r in recipes]
bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor='white')
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
            f'{val:.1f} ms', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Latency (ms)', fontsize=11, fontweight='bold')
ax.set_title('Flash Attention: 81x Speedup\nN=2048, d=64', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1200)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: memory comparison
ax2 = axes[1]
mem_labels = ['Standard\n(N×N matrix)', 'Flash\n(O(N) only)']
mem_values = [16.0, 0.016]
mem_colors = ['#e74c3c', '#27ae60']
bars2 = ax2.barh(mem_labels, mem_values, color=mem_colors, height=0.5, edgecolor='white')
ax2.text(mem_values[0] + 0.3, 0, f'{mem_values[0]:.0f} MB', va='center', fontsize=11, fontweight='bold')
ax2.text(mem_values[1] + 0.3, 1, f'{mem_values[1]*1000:.0f} KB', va='center', fontsize=11, fontweight='bold')
ax2.set_xlabel('Memory (MB)', fontsize=11, fontweight='bold')
ax2.set_title('Memory: 1000x Reduction', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 22)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('assets/performance.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved assets/performance.png")
