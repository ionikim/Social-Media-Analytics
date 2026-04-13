"""
EEG Network Transition: Interictal → Ictal
===========================================
Dataset : CHB-MIT · chb01_03 · 2980-3010 s
Onset   : +16 s into file (sample 4096)
 
Run with:  python3 eeg_transition_analysis.py
Plots are saved as PNG files in the same directory.
"""
 
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
 
# ── configuration ──────────────────────────────────────────────────────────
NPZ_PATH     = '/Users/antoniaspoerk/Desktop/Digital Neuroscience /Social Media Analytics/epilepsy_pediatrics_EEG/data/graphs/adjacency_sparse/inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz'
N_CHANNELS   = 23
N_TIMEPOINTS = 7680    # 30 s x 256 Hz
FS           = 256
WINDOW_SEC   = 5
STEP_SEC     = 1
ONSET_SEC    = 16      # seizure onset offset within this file
 
CHANNEL_NAMES = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8-1"
]
 
BG             = "#0f1117"
PANEL_BG       = "#1a1d27"
TEXT           = "#e0e0e0"
ONSET_COLOR    = "#ff4f4f"
INTERICTAL_CLR = "#7c6af7"
ICTAL_CLR      = "#f7936a"
 
 
# ── step 1: load & precompute ──────────────────────────────────────────────
print("Loading adjacency matrix ...")
mat = sp.load_npz(NPZ_PATH)
print(f"  Shape: {mat.shape}  |  Non-zeros: {mat.nnz:,}")
 
window_samples = WINDOW_SEC * FS
step_samples   = STEP_SEC * FS
t_starts       = list(range(0, N_TIMEPOINTS - window_samples + 1, step_samples))
t_centers_sec  = [(t + window_samples / 2) / FS for t in t_starts]
print(f"  Windows: {len(t_starts)}  ({WINDOW_SEC}s window, {STEP_SEC}s step)")
 
print("Precomputing correlation matrices ...")
all_corrs = []
for t_start in t_starts:
    ch_temporal = np.zeros((N_CHANNELS, window_samples))
    for ch in range(N_CHANNELS):
        block = mat[
            ch * N_TIMEPOINTS + t_start : ch * N_TIMEPOINTS + t_start + window_samples,
            ch * N_TIMEPOINTS + t_start : ch * N_TIMEPOINTS + t_start + window_samples
        ]
        ch_temporal[ch] = np.array(block.sum(axis=1)).flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(ch_temporal)
    all_corrs.append(np.nan_to_num(corr, nan=0.0))
 
all_corrs = np.array(all_corrs)   # (n_windows, 23, 23)
idx       = np.triu_indices(N_CHANNELS, k=1)
mean_corr = [c[idx].mean() for c in all_corrs]
 
# fixed channel order from average interictal correlation
interictal_idx = [i for i, t in enumerate(t_centers_sec) if t <= ONSET_SEC]
ictal_idx      = [i for i, t in enumerate(t_centers_sec) if t >  ONSET_SEC]
corr_inter     = all_corrs[interictal_idx].mean(axis=0)
dist_inter     = squareform(1.0 - np.clip(corr_inter, -1, 1), checks=False)
order          = sch.dendrogram(sch.linkage(dist_inter, method="ward"), no_plot=True)["leaves"]
labels         = [CHANNEL_NAMES[i] for i in order]
 
def reorder(m):
    return m[np.ix_(order, order)]
 
print("Done.\n")
 
 
# ── plot 1: global synchrony ───────────────────────────────────────────────
print("Saving plot 1: global synchrony ...")
fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
 
ax.axvspan(0, ONSET_SEC, alpha=0.08, color=INTERICTAL_CLR, label="Interictal")
ax.axvspan(ONSET_SEC, t_centers_sec[-1] + WINDOW_SEC / 2,
           alpha=0.08, color=ICTAL_CLR, label="Ictal")
ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2,
           linestyle="--", label=f"Seizure onset (+{ONSET_SEC}s)")
ax.plot(t_centers_sec, mean_corr, color="white", linewidth=2, zorder=3)
ax.fill_between(t_centers_sec, mean_corr, alpha=0.25, color="white")
ax.set_xlabel("Time within recording window (s)", color=TEXT, fontsize=10)
ax.set_ylabel("Mean Pearson r", color=TEXT, fontsize=10)
ax.set_title("Global Network Synchrony  .  CHB-01  .  chb01_03  .  2980-3010 s",
             color=TEXT, fontsize=11, fontweight="bold")
ax.tick_params(colors=TEXT)
ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=9)
plt.tight_layout()
fig.savefig("plot1_synchrony.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> plot1_synchrony.png")
 
 
# ── plot 2: interictal vs ictal heatmaps ──────────────────────────────────
print("Saving plot 2: interictal vs ictal heatmaps ...")
corr_interictal = all_corrs[interictal_idx].mean(axis=0)
corr_ictal      = all_corrs[ictal_idx].mean(axis=0)
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle("Average Network Correlation  .  Interictal vs Ictal  .  CHB-01 chb01_03",
             color=TEXT, fontsize=12, fontweight="bold", y=1.01)
 
for ax, corr_m, title, accent in zip(
    axes,
    [reorder(corr_interictal), reorder(corr_ictal)],
    ["Interictal  (0-16 s)", "Ictal  (16-30 s)"],
    [INTERICTAL_CLR, ICTAL_CLR]
):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    im = ax.imshow(corr_m, cmap="RdYlBu_r", vmin=-0.2, vmax=0.5,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(N_CHANNELS))
    ax.set_yticks(range(N_CHANNELS))
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=TEXT)
    ax.set_yticklabels(labels, fontsize=7, color=TEXT)
    ax.set_title(title, color=accent, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson r", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
    cbar.outline.set_edgecolor(PANEL_BG)
 
plt.tight_layout()
fig.savefig("plot2_interictal_vs_ictal.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> plot2_interictal_vs_ictal.png")
 
 
# ── plot 3: animated sliding window ───────────────────────────────────────
print("Saving plot 3: animated sliding window (takes ~20s) ...")
n_frames = len(all_corrs)
 
fig = plt.figure(figsize=(13, 6), facecolor=BG)
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.35,
                        left=0.05, right=0.97, top=0.88, bottom=0.12)
ax_heat = fig.add_subplot(gs[0])
ax_sync = fig.add_subplot(gs[1])
fig.suptitle("Sliding Window Network Analysis  .  CHB-01 chb01_03  .  2980-3010 s",
             color=TEXT, fontsize=11, fontweight="bold")
 
ax_heat.set_facecolor(PANEL_BG)
for sp_ in ax_heat.spines.values():
    sp_.set_visible(False)
im = ax_heat.imshow(reorder(all_corrs[0]), cmap="RdYlBu_r",
                    vmin=-0.2, vmax=0.5, aspect="auto", interpolation="nearest")
ax_heat.set_xticks(range(N_CHANNELS))
ax_heat.set_yticks(range(N_CHANNELS))
ax_heat.set_xticklabels(labels, rotation=90, fontsize=6.5, color=TEXT)
ax_heat.set_yticklabels(labels, fontsize=6.5, color=TEXT)
ax_heat.tick_params(colors=TEXT)
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
cbar.set_label("Pearson r", color=TEXT, fontsize=8)
cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
cbar.outline.set_edgecolor(PANEL_BG)
heat_title = ax_heat.set_title("", color=TEXT, fontsize=10, fontweight="bold")
 
ax_sync.set_facecolor(PANEL_BG)
for sp_ in ax_sync.spines.values():
    sp_.set_edgecolor("#333")
ax_sync.axvspan(0, ONSET_SEC, alpha=0.10, color=INTERICTAL_CLR)
ax_sync.axvspan(ONSET_SEC, t_centers_sec[-1] + WINDOW_SEC / 2, alpha=0.10, color=ICTAL_CLR)
ax_sync.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=1.5, linestyle="--")
ax_sync.plot(t_centers_sec, mean_corr, color="white", linewidth=1.5, alpha=0.4)
ax_sync.fill_between(t_centers_sec, mean_corr, alpha=0.1, color="white")
cursor_dot,  = ax_sync.plot([], [], "o", color=ONSET_COLOR, markersize=8, zorder=5)
cursor_line  = ax_sync.axvline(t_centers_sec[0], color=ONSET_COLOR,
                               linewidth=1.2, linestyle=":", alpha=0.8)
ax_sync.set_xlabel("Window centre (s)", color=TEXT, fontsize=8)
ax_sync.set_ylabel("Mean Pearson r", color=TEXT, fontsize=8)
ax_sync.set_title("Global Synchrony", color=TEXT, fontsize=9)
ax_sync.tick_params(colors=TEXT, labelsize=7)
 
def animate(frame):
    t_c   = t_centers_sec[frame]
    t_s   = t_starts[frame] / FS
    t_e   = t_s + WINDOW_SEC
    phase = "ICTAL" if t_c > ONSET_SEC else "interictal"
    color = ICTAL_CLR if phase == "ICTAL" else INTERICTAL_CLR
    im.set_data(reorder(all_corrs[frame]))
    heat_title.set_text(f"{phase}  |  window {t_s:.0f}-{t_e:.0f} s")
    heat_title.set_color(color)
    cursor_dot.set_data([t_c], [mean_corr[frame]])
    cursor_line.set_xdata([t_c, t_c])
    return im, heat_title, cursor_dot, cursor_line
 
anim = FuncAnimation(fig, animate, frames=n_frames, interval=600, blit=True, repeat=True)
anim.save("plot3_animation.gif", writer="pillow", fps=2, dpi=120)
plt.close(fig)
print("  Saved -> plot3_animation.gif")
 
 
# ── plot 4: dendrogram grid ────────────────────────────────────────────────
print("Saving plot 4: dendrogram grid ...")
n_cols = 6
n_rows = int(np.ceil(n_frames / n_cols))
 
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 2.8, n_rows * 2.4), facecolor=BG)
fig.suptitle("Dendrogram per Window  .  CHB-01 chb01_03  .  2980-3010 s",
             color=TEXT, fontsize=12, fontweight="bold", y=1.01)
sch.set_link_color_palette(["#7c6af7", "#f7936a", "#5ecfb1", "#e05c97", "#f5d547"])
 
for i, ax in enumerate(axes.flat):
    ax.set_facecolor(PANEL_BG)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    if i >= n_frames:
        ax.axis("off")
        continue
    t_s   = t_starts[i] / FS
    t_e   = t_s + WINDOW_SEC
    t_c   = t_centers_sec[i]
    phase = "ICTAL" if t_c > ONSET_SEC else "interictal"
    color = ICTAL_CLR if phase == "ICTAL" else INTERICTAL_CLR
    dist_w = squareform(1.0 - np.clip(all_corrs[i], -1, 1), checks=False)
    link_w = sch.linkage(dist_w, method="ward")
    sch.dendrogram(link_w, labels=CHANNEL_NAMES, orientation="left", ax=ax,
                   color_threshold=0.55 * link_w[-1, 2],
                   above_threshold_color=TEXT, leaf_font_size=4.5)
    ax.tick_params(colors=TEXT, labelsize=4)
    onset_tag = "  <- ONSET" if abs(t_s - ONSET_SEC) < 0.5 else ""
    ax.set_title(f"{t_s:.0f}-{t_e:.0f}s  {phase}{onset_tag}",
                 color=color, fontsize=6, pad=3)
 
plt.tight_layout()
fig.savefig("plot4_dendrogram_grid.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> plot4_dendrogram_grid.png")
 
print("\nAll done. Output files:")
print("  plot1_synchrony.png")
print("  plot2_interictal_vs_ictal.png")
print("  plot3_animation.gif")
print("  plot4_dendrogram_grid.png")
 