
# benchmarks/sweet_spots/spectral_sweet_spot.py
import numpy as np
from sweet_spot_framework import sweet_spot_experiment, plot_sweet_spot
from your_spectral_module import spectral_clustering  # eure Implementierung

def run_spectral_wrapped(k, seed):
    np.random.seed(seed)
    labels, _, _ = spectral_clustering(
        corr_matrix=CORR,   # euer vorberechnetes Fenster
        k=k,
        threshold=THRESHOLD
    )
    return labels

ks = [2, 3, 4, 5, 6, 8, 10]

cpu, nmi = sweet_spot_experiment(
    run_fn=run_spectral_wrapped,
    terminations=ks,
    n_runs=5
)

plot_sweet_spot(
    ks,
    cpu,
    nmi,
    title="Sweet Spot — Spectral Laplacian Clustering",
    xlabel="Number of Clusters (k)",
    out="spectral_sweet_spot.png"
)
