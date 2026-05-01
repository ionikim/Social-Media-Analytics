
# benchmarks/sweet_spots/ward_sweet_spot.py
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sweet_spot_framework import sweet_spot_experiment, plot_sweet_spot

Z = linkage(DIST_MATRIX, method="ward")

def run_ward_wrapped(k, seed):
    # seed irrelevant, deterministisch
    labels = fcluster(Z, k, criterion="maxclust")
    return labels

ks = [2, 3, 4, 6, 8, 10, 15]

cpu, nmi = sweet_spot_experiment(
    run_fn=run_ward_wrapped,
    terminations=ks,
    n_runs=2
)

plot_sweet_spot(
    ks,
    cpu,
    nmi,
    title="Sweet Spot — Ward Hierarchical Clustering",
    xlabel="Number of Clusters (Dendrogram Cut)",
    out="ward_sweet_spot.png"
)
