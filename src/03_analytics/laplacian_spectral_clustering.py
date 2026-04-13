"""
Spectral Clustering from Scratch — Graph Laplacian on EEG Adjacency Matrix
===========================================================================
Dataset : CHB-MIT · chb01_03 · 2980-3010 s
Onset   : +16 s into file (sample 4096)
 
No sklearn. No scipy clustering. Pure numpy throughout.
 
Pipeline:
  1. Load sparse adjacency matrix
  2. Extract 23x23 channel correlation matrix per sliding window
  3. Threshold correlation -> weighted adjacency graph
  4. Compute normalized graph Laplacian from scratch
  5. Eigendecompose Laplacian -> embed channels in eigenspace
  6. Assign clusters via simple k-means on eigenvectors (also from scratch)
  7. Track how cluster assignments change across the seizure transition
  8. Plot results
 
Run with:  python3 eeg_spectral_clustering.py
"""
 
import warnings
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
# ── configuration ──────────────────────────────────────────────────────────
NPZ_PATH     = '/Users/antoniaspoerk/Desktop/Digital Neuroscience /Social Media Analytics/epilepsy_pediatrics_EEG/data/graphs/adjacency_sparse/inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz'
N_CHANNELS   = 23
N_TIMEPOINTS = 7680
FS           = 256
WINDOW_SEC   = 5
STEP_SEC     = 1
ONSET_SEC    = 16
K_CLUSTERS   = 4      # number of clusters for spectral clustering
THRESHOLD    = 0.3    # correlation threshold to build the graph
                      # edges with r < threshold are set to 0
 
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
CLUSTER_COLORS = ["#7c6af7", "#f7936a", "#5ecfb1", "#e05c97", "#f5d547"]
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — load & precompute correlation matrices (same as before)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading adjacency matrix ...")
mat = sp.load_npz(NPZ_PATH)
print(f"  Shape: {mat.shape}  |  Non-zeros: {mat.nnz:,}")
 
window_samples = WINDOW_SEC * FS
step_samples   = STEP_SEC * FS
t_starts       = list(range(0, N_TIMEPOINTS - window_samples + 1, step_samples))
t_centers_sec  = [(t + window_samples / 2) / FS for t in t_starts]
n_windows      = len(t_starts)
 
print(f"Precomputing {n_windows} correlation matrices ...")
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
 
all_corrs = np.array(all_corrs)
print("Done.\n")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS — implemented from scratch in numpy
# ══════════════════════════════════════════════════════════════════════════════
 
def build_adjacency(corr_matrix, threshold):
    """
    Convert a correlation matrix into a weighted adjacency matrix.
 
    Edges only exist where correlation > threshold.
    Self-loops are removed (diagonal = 0).
    Negative correlations are clipped to 0 (we want similarity, not dissimilarity).
 
    Parameters
    ----------
    corr_matrix : (N, N) array
    threshold   : float, minimum correlation to keep an edge
 
    Returns
    -------
    A : (N, N) weighted adjacency matrix
    """
    A = corr_matrix.copy()
    A[A < threshold] = 0.0      # remove weak edges
    A[A < 0]         = 0.0      # remove negative correlations
    np.fill_diagonal(A, 0.0)    # no self-loops
    return A
 
 
def compute_normalized_laplacian(A):
    """
    Compute the symmetric normalized graph Laplacian from scratch.
 
        L_sym = I - D^(-1/2) A D^(-1/2)
 
    where D is the diagonal degree matrix: D[i,i] = sum of row i of A.
 
    The normalized version is preferred over the unnormalized L = D - A
    because it handles nodes with very different degrees (which is common
    in EEG graphs where some channels are highly connected and others are not).
 
    Parameters
    ----------
    A : (N, N) weighted adjacency matrix (no self-loops)
 
    Returns
    -------
    L : (N, N) normalized Laplacian matrix
    """
    N      = A.shape[0]
    degree = A.sum(axis=1)                  # degree of each node
 
    # D^(-1/2): inverse square root of degree, handle degree=0 safely
    d_inv_sqrt = np.zeros(N)
    nonzero    = degree > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
 
    # Build D^(-1/2) as a diagonal matrix
    D_inv_sqrt = np.diag(d_inv_sqrt)
 
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
 
    return L
 
 
def spectral_embed(L, k):
    """
    Eigendecompose the Laplacian and return the k smallest eigenvectors.
 
    The smallest eigenvalue of L is always 0 (the constant vector).
    The next k-1 eigenvectors encode the cluster structure of the graph —
    this is the core insight of spectral clustering.
 
    We use numpy's eigh (symmetric eigenvalue solver) which is more
    numerically stable than eig for symmetric matrices.
 
    Parameters
    ----------
    L : (N, N) normalized Laplacian
    k : int, number of clusters = number of eigenvectors to keep
 
    Returns
    -------
    embedding : (N, k) matrix — each row is a channel in eigenspace
    eigenvalues : (N,) sorted eigenvalues (for inspection)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(L)
 
    # eigh returns eigenvalues in ascending order
    # take the k smallest (they carry the cluster structure)
    embedding = eigenvectors[:, :k]
 
    return embedding, eigenvalues
 
 
def normalize_rows(X):
    """
    L2-normalize each row of the embedding matrix.
    Standard practice before k-means on spectral embeddings —
    projects all points onto the unit sphere so distance reflects
    angle rather than magnitude.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0     # avoid division by zero for isolated nodes
    return X / norms
 
 
def kmeans_scratch(X, k, n_init=10, max_iter=300, tol=1e-6, seed=42):
    """
    K-means clustering implemented from scratch.
 
    Runs n_init times with different random seeds and returns
    the result with the lowest inertia (sum of squared distances).
 
    Parameters
    ----------
    X        : (N, d) data matrix — rows are points, cols are features
    k        : int, number of clusters
    n_init   : int, number of random restarts
    max_iter : int, maximum iterations per run
    tol      : float, convergence threshold on centroid movement
    seed     : int, base random seed
 
    Returns
    -------
    best_labels    : (N,) cluster assignment for each point
    best_centroids : (k, d) final centroids
    best_inertia   : float, sum of squared distances to nearest centroid
    """
    rng          = np.random.default_rng(seed)
    best_labels  = None
    best_inertia = np.inf
 
    for init_run in range(n_init):
 
        # ── initialise centroids (k-means++ style) ──────────────────────────
        # Pick first centroid randomly, then pick each subsequent centroid
        # with probability proportional to distance from nearest existing centroid
        centroids = []
        first_idx = rng.integers(0, len(X))
        centroids.append(X[first_idx].copy())
 
        for _ in range(k - 1):
            # distance from each point to its nearest centroid so far
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs    = dists / dists.sum()
            next_idx = rng.choice(len(X), p=probs)
            centroids.append(X[next_idx].copy())
 
        centroids = np.array(centroids)   # (k, d)
 
        # ── iterate ─────────────────────────────────────────────────────────
        labels = np.zeros(len(X), dtype=int)
 
        for iteration in range(max_iter):
 
            # Assignment step: each point -> nearest centroid
            dists_all = np.array([
                np.sum((X - c) ** 2, axis=1) for c in centroids
            ])                                  # (k, N)
            new_labels = np.argmin(dists_all, axis=0)   # (N,)
 
            # Update step: recompute centroids as cluster means
            new_centroids = np.zeros_like(centroids)
            for cluster_id in range(k):
                mask = new_labels == cluster_id
                if mask.sum() > 0:
                    new_centroids[cluster_id] = X[mask].mean(axis=0)
                else:
                    # empty cluster — reinitialise to a random point
                    new_centroids[cluster_id] = X[rng.integers(0, len(X))]
 
            # Convergence check
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels    = new_labels
 
            if shift < tol:
                break
 
        # Inertia: sum of squared distances to assigned centroid
        inertia = sum(
            np.sum((X[labels == c] - centroids[c]) ** 2)
            for c in range(k)
            if (labels == c).sum() > 0
        )
 
        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels.copy()
            best_centroids = centroids.copy()
 
    return best_labels, best_centroids, best_inertia
 
 
def spectral_clustering(corr_matrix, k, threshold):
    """
    Full spectral clustering pipeline for one correlation matrix.
 
    Steps:
      1. Build adjacency graph (threshold weak correlations)
      2. Compute normalized Laplacian
      3. Embed channels in k-dimensional eigenspace
      4. Row-normalize the embedding
      5. Run k-means from scratch on the embedding
 
    Parameters
    ----------
    corr_matrix : (N, N) correlation matrix
    k           : int, number of clusters
    threshold   : float, minimum correlation to keep edge
 
    Returns
    -------
    labels      : (N,) cluster assignment per channel
    eigenvalues : (N,) Laplacian eigenvalues (for eigengap inspection)
    embedding   : (N, k) spectral embedding
    """
    A          = build_adjacency(corr_matrix, threshold)
    L          = compute_normalized_laplacian(A)
    embedding, eigenvalues = spectral_embed(L, k)
    embedding  = normalize_rows(embedding)
    labels, _, _ = kmeans_scratch(embedding, k)
    return labels, eigenvalues, embedding
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — run spectral clustering on every window
# ══════════════════════════════════════════════════════════════════════════════
print(f"Running spectral clustering (k={K_CLUSTERS}, threshold={THRESHOLD}) ...")
all_labels      = []   # cluster assignments per window
all_eigenvalues = []   # eigenvalue spectra per window
 
for i, corr in enumerate(all_corrs):
    labels, eigenvalues, _ = spectral_clustering(corr, K_CLUSTERS, THRESHOLD)
    all_labels.append(labels)
    all_eigenvalues.append(eigenvalues)
    t_s = t_starts[i] / FS
    print(f"  window {t_s:4.0f}s  clusters: {[int((labels==c).sum()) for c in range(K_CLUSTERS)]}")
 
all_labels      = np.array(all_labels)       # (n_windows, N_CHANNELS)
all_eigenvalues = np.array(all_eigenvalues)  # (n_windows, N_CHANNELS)
print("Done.\n")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — align cluster labels across windows
#  (k-means labels are arbitrary per run — align greedily to window 0)
# ══════════════════════════════════════════════════════════════════════════════
def align_labels(ref_labels, new_labels, k):
    """
    Permute new_labels so they best match ref_labels.
    Uses greedy maximum overlap matching.
    """
    perm = {}
    used = set()
    for ref_c in range(k):
        ref_mask = ref_labels == ref_c
        best_overlap = -1
        best_new_c   = -1
        for new_c in range(k):
            if new_c in used:
                continue
            overlap = (new_labels[ref_mask] == new_c).sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_new_c   = new_c
        perm[best_new_c] = ref_c
        used.add(best_new_c)
    aligned = np.array([perm.get(l, l) for l in new_labels])
    return aligned
 
aligned_labels = [all_labels[0].copy()]
for i in range(1, n_windows):
    aligned = align_labels(aligned_labels[0], all_labels[i], K_CLUSTERS)
    aligned_labels.append(aligned)
aligned_labels = np.array(aligned_labels)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — cluster assignments over time (raster / carpet plot)
# ══════════════════════════════════════════════════════════════════════════════
print("Saving plot 1: cluster assignment raster ...")
fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_visible(False)
 
cmap_discrete = plt.cm.colors if hasattr(plt.cm, 'colors') else None
colors_4 = CLUSTER_COLORS[:K_CLUSTERS]
 
from matplotlib.colors import ListedColormap
cmap_disc = ListedColormap(colors_4)
 
img = ax.imshow(
    aligned_labels.T,           # (N_CHANNELS, n_windows)
    aspect="auto",
    cmap=cmap_disc,
    vmin=0, vmax=K_CLUSTERS - 1,
    interpolation="nearest"
)
 
# onset line
onset_frame = next(i for i, t in enumerate(t_centers_sec) if t > ONSET_SEC) - 0.5
ax.axvline(onset_frame, color=ONSET_COLOR, linewidth=2,
           linestyle="--", label="Seizure onset")
 
ax.set_yticks(range(N_CHANNELS))
ax.set_yticklabels(CHANNEL_NAMES, fontsize=7, color=TEXT)
ax.set_xticks(range(n_windows))
ax.set_xticklabels([f"{t:.0f}s" for t in t_centers_sec],
                   rotation=90, fontsize=7, color=TEXT)
ax.tick_params(colors=TEXT)
ax.set_xlabel("Window centre", color=TEXT, fontsize=10)
ax.set_title(
    f"Spectral Cluster Assignment per Channel over Time  .  k={K_CLUSTERS}  .  CHB-01 chb01_03",
    color=TEXT, fontsize=11, fontweight="bold"
)
 
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_4[c], label=f"Cluster {c+1}")
                   for c in range(K_CLUSTERS)]
legend_elements.append(plt.Line2D([0], [0], color=ONSET_COLOR,
                                  linestyle="--", label="Seizure onset"))
ax.legend(handles=legend_elements, facecolor=PANEL_BG, edgecolor="#444",
          labelcolor=TEXT, fontsize=8, loc="upper left")
 
plt.tight_layout()
fig.savefig("spectral_plot1_raster.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> spectral_plot1_raster.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — eigengap over time
#  The eigengap (gap between k-th and (k+1)-th eigenvalue) tells you
#  how well-separated the clusters are. A large gap = strong cluster structure.
# ══════════════════════════════════════════════════════════════════════════════
print("Saving plot 2: eigengap over time ...")
eigengap = all_eigenvalues[:, K_CLUSTERS] - all_eigenvalues[:, K_CLUSTERS - 1]
 
fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
 
ax.axvspan(0, ONSET_SEC, alpha=0.08, color=INTERICTAL_CLR, label="Interictal")
ax.axvspan(ONSET_SEC, t_centers_sec[-1] + WINDOW_SEC / 2,
           alpha=0.08, color=ICTAL_CLR, label="Ictal")
ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2,
           linestyle="--", label="Seizure onset")
 
ax.plot(t_centers_sec, eigengap, color="white", linewidth=2, zorder=3)
ax.fill_between(t_centers_sec, eigengap, alpha=0.25, color="white")
 
ax.set_xlabel("Window centre (s)", color=TEXT, fontsize=10)
ax.set_ylabel(f"Eigengap (λ{K_CLUSTERS+1} - λ{K_CLUSTERS})", color=TEXT, fontsize=10)
ax.set_title(
    f"Laplacian Eigengap over Time  .  larger = stronger cluster structure",
    color=TEXT, fontsize=11, fontweight="bold"
)
ax.tick_params(colors=TEXT)
ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=9)
 
plt.tight_layout()
fig.savefig("spectral_plot2_eigengap.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> spectral_plot2_eigengap.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — interictal vs ictal cluster maps (which channels in which cluster)
# ══════════════════════════════════════════════════════════════════════════════
print("Saving plot 3: interictal vs ictal cluster comparison ...")
 
interictal_idx  = [i for i, t in enumerate(t_centers_sec) if t <= ONSET_SEC]
ictal_idx       = [i for i, t in enumerate(t_centers_sec) if t >  ONSET_SEC]
 
# majority vote cluster per channel in each phase
def majority_cluster(labels_subset):
    """For each channel, return the most common cluster across windows."""
    result = np.zeros(N_CHANNELS, dtype=int)
    for ch in range(N_CHANNELS):
        counts = np.bincount(labels_subset[:, ch], minlength=K_CLUSTERS)
        result[ch] = np.argmax(counts)
    return result
 
inter_majority = majority_cluster(aligned_labels[interictal_idx])
ictal_majority = majority_cluster(aligned_labels[ictal_idx])
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "Dominant Cluster per Channel  .  Interictal vs Ictal  .  CHB-01 chb01_03",
    color=TEXT, fontsize=12, fontweight="bold", y=1.01
)
 
for ax, majority, title, accent in zip(
    axes,
    [inter_majority, ictal_majority],
    ["Interictal  (0-16 s)", "Ictal  (16-30 s)"],
    [INTERICTAL_CLR, ICTAL_CLR]
):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    # horizontal bar, one per channel, coloured by cluster
    for ch_idx, (ch_name, cluster) in enumerate(zip(CHANNEL_NAMES, majority)):
        ax.barh(ch_idx, 1, color=colors_4[cluster], alpha=0.85, height=0.8)
        ax.text(1.05, ch_idx, f"cluster {cluster + 1}",
                va="center", fontsize=7, color=colors_4[cluster])
 
    ax.set_yticks(range(N_CHANNELS))
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=7.5, color=TEXT)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_title(title, color=accent, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT)
 
plt.tight_layout()
fig.savefig("spectral_plot3_cluster_map.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> spectral_plot3_cluster_map.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 4 — eigenvalue spectrum for one interictal vs one ictal window
# ══════════════════════════════════════════════════════════════════════════════
print("Saving plot 4: eigenvalue spectrum comparison ...")
 
mid_inter = interictal_idx[len(interictal_idx) // 2]
mid_ictal = ictal_idx[len(ictal_idx) // 2]
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
fig.suptitle(
    "Laplacian Eigenvalue Spectrum  .  Interictal vs Ictal",
    color=TEXT, fontsize=12, fontweight="bold", y=1.01
)
 
for ax, win_idx, title, accent in zip(
    axes,
    [mid_inter, mid_ictal],
    [f"Interictal window  ({t_centers_sec[mid_inter]:.0f}s)",
     f"Ictal window  ({t_centers_sec[mid_ictal]:.0f}s)"],
    [INTERICTAL_CLR, ICTAL_CLR]
):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
 
    evals = all_eigenvalues[win_idx]
    ax.bar(range(len(evals)), evals, color=accent, alpha=0.7, width=0.7)
 
    # mark the k-th eigengap
    ax.axvline(K_CLUSTERS - 0.5, color=ONSET_COLOR, linewidth=1.5,
               linestyle="--", label=f"k={K_CLUSTERS} cut")
 
    ax.set_xlabel("Eigenvalue index", color=TEXT, fontsize=9)
    ax.set_ylabel("Eigenvalue", color=TEXT, fontsize=9)
    ax.set_title(title, color=accent, fontsize=10, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=8)
 
plt.tight_layout()
fig.savefig("spectral_plot4_eigenspectrum.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved -> spectral_plot4_eigenspectrum.png")
 
print("\nAll done. Output files:")
print("  spectral_plot1_raster.png        — cluster assignments over time")
print("  spectral_plot2_eigengap.png      — cluster separation strength over time")
print("  spectral_plot3_cluster_map.png   — interictal vs ictal cluster membership")
print("  spectral_plot4_eigenspectrum.png — Laplacian eigenvalue spectrum")
 