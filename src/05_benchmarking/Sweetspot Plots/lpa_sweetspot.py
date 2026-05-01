
# benchmarks/sweet_spots/lpa_sweet_spot.py
import numpy as np
from sweet_spot_framework import sweet_spot_experiment, plot_sweet_spot
from your_lpa_module import run_lpa   # ← eure LPA-Funktion

def run_lpa_wrapped(max_iter, seed):
    np.random.seed(seed)
    labels = run_lpa(max_iter=max_iter)
    return np.array(list(labels.values()))

terminations = [1, 2, 5, 10, 20, 50, 100]

cpu, nmi = sweet_spot_experiment(
    run_fn=run_lpa_wrapped,
    terminations=terminations,
    n_runs=10
)

plot_sweet_spot(
    terminations, cpu, nmi,
    title="Sweet Spot — Label Propagation",
    xlabel="Max Iterations",
    out="lpa_sweet_spot.png"
)
