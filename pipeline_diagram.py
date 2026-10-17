"""Draws the 5-step Master Engineering Pipeline for this project -> outputs/pipeline_flow.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

steps = [
    ("1. DEFINE", "Input: 14 sensor\nreadings x 30 cycles\n\nOutput: RUL\n(cycles left)"),
    ("2. DATA", "NASA C-MAPSS\nFD001\n100 run-to-failure\nengines"),
    ("3. PROCESS", "RUL label (cap 125)\nDrop 7 flat sensors\nSplit by engine 80/20\nMin-Max scale (train)\n30-cycle windows"),
    ("4. BUILD", "LSTM (64 units)\n+ Dense head\n\nBaselines: Mean,\nRandom Forest"),
    ("5. EVALUATE", "RMSE, MAE,\nNASA score,\nerror distribution,\nmaintenance alerts"),
]
colors = ["#1f5aa6", "#2471a3", "#2e86c1", "#3498db", "#5dade2"]

fig, ax = plt.subplots(figsize=(15, 4.6))
w, h, tip = 2.8, 3.0, 0.45
for i, ((title, body), c) in enumerate(zip(steps, colors)):
    x = i * (w + 0.1)
    pts = [(x, 0), (x + w, 0), (x + w + tip, h / 2), (x + w, h), (x, h)]
    if i > 0:
        pts.append((x + tip, h / 2))
    ax.add_patch(Polygon(pts, closed=True, fc=c, ec="white", lw=2))
    cx = x + w / 2 + (tip / 2 if i > 0 else 0)
    ax.text(cx, h - 0.4, title, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.text(cx, h / 2 - 0.25, body, ha="center", va="center", color="white", fontsize=10.5, family="monospace")

ax.text(7.25, -0.45, "Result: LSTM RMSE 13.5 cycles  vs  Random Forest 16.4  vs  Mean baseline 40.6",
        ha="center", fontsize=12, fontweight="bold", color="#1f2d3d")
ax.set_xlim(-0.1, 5 * (w + 0.1) + 0.4)
ax.set_ylim(-0.8, h + 0.6)
ax.set_title("Rolls-Royce RUL Prediction - Master Engineering Pipeline", fontsize=16, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/pipeline_flow.png", dpi=170, bbox_inches="tight")
print("saved outputs/pipeline_flow.png")
