
import matplotlib.pyplot as plt

def plot_single(ax, x, cpu, nmi, title, xlabel):
    ax2 = ax.twinx()

    ax.plot(x, cpu, color="crimson", marker="o", label="CPU time")
    ax2.plot(x, nmi, color="seagreen", marker="s", label="NMI")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CPU time [s]", color="crimson")
    ax2.set_ylabel("NMI", color="seagreen")
    ax2.set_ylim(0, 1.05)

    # Sweet spot heuristic
    idx = max(range(1, len(x) - 1),
              key=lambda i: (nmi[i+1] - nmi[i-1]) /
                            (cpu[i+1] - cpu[i-1] + 1e-9))
    ax.axvline(x[idx], linestyle="--", color="black", alpha=0.6)

    ax.set_title(title, fontsize=11)

# --------------------------------------------------
# Our Inputs (we need to compute the real values beforehand)
# --------------------------------------------------

moore_x, moore_cpu, moore_nmi = [500,1000,2000,5000,10000], [0.2,0.3,0.5,1.2,2.5], [0.40,0.62,0.78,0.81,0.82]
lpa_x, lpa_cpu, lpa_nmi       = [1,2,5,10,20,50],           [0.05,0.08,0.12,0.3,0.8,2.0], [0.30,0.55,0.76,0.80,0.81,0.81]
spec_x, spec_cpu, spec_nmi    = [2,3,4,5,6,8],             [0.05,0.08,0.12,0.25,0.6,1.5], [0.45,0.68,0.82,0.83,0.83,0.83]
ward_x, ward_cpu, ward_nmi    = [2,3,4,6,8,10],            [0.01,0.015,0.02,0.04,0.07,0.1], [1.0,1.0,1.0,1.0,1.0,1.0]

# --------------------------------------------------
# Plot grid
# --------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot_single(
    axes[0,0], moore_x, moore_cpu, moore_nmi,
    "Moore Streaming", "Processed edges"
)

plot_single(
    axes[0,1], lpa_x, lpa_cpu, lpa_nmi,
    "Label Propagation (LPA)", "Iterations"
)

plot_single(
    axes[1,0], spec_x, spec_cpu, spec_nmi,
    "Spectral Laplacian Clustering", "Number of clusters (k)"
)

plot_single(
    axes[1,1], ward_x, ward_cpu, ward_nmi,
    "Ward Hierarchical Clustering", "Dendrogram cut (k)"
)

fig.suptitle(
    "Sweet‑Spot Analysis: CPU–Stability Trade‑offs across Community Detection Algorithms",
    fontsize=14, y=0.98
)

plt.tight_layout()
plt.savefig("sweet_spot_4algorithms.png", dpi=150)
plt.show()
