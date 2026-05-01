
# benchmarks/sweet_spots/moore_sweet_spot.py
import numpy as np
from sweet_spot_framework import sweet_spot_experiment, plot_sweet_spot
from your_moore_module import stream_moore_partial

def run_moore_wrapped(max_edges, seed):
    np.random.seed(seed)
    return stream_moore_partial(max_edges=max_edges)

terminations = [500, 1000, 2000, 5000, 10000, 20000]

cpu, nmi = sweet_spot_experiment(
    run_fn=run_moore_wrapped,
    terminations=terminations,
    n_runs=5
)

plot_sweet_spot(
    terminations, cpu, nmi,
    title="Sweet Spot — Moore Streaming",
    xlabel="Processed Edges",
    out="moore_streaming_sweet_spot.png"
)
