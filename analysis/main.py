import numpy as np
import networkx as nx
from scipy import sparse
from collections import Counter, defaultdict
import random
from pathlib import Path

# ===============================
# 1. Pfade robust definieren
# ===============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "graphs" / "adjacency_sparse"

# ===============================
# 2. NPZ-Datei automatisch finden
# ===============================
npz_files = list(DATA_DIR.glob("*.npz"))

print("Gefundene NPZ-Dateien:")
for f in npz_files:
    print(" -", f)

if len(npz_files) == 0:
    raise FileNotFoundError(
        "Keine .npz Dateien im data/graphs/adjacency_sparse Ordner gefunden!"
    )

FILE = npz_files[0]
print("Verwende Datei:", FILE)

# ===============================
# 3. Sparse Adjazenzmatrix laden
# ===============================
A = sparse.load_npz(FILE)

print("Adjacency shape:", A.shape)
print("Non-zero edges (roh):", A.nnz)

# ===============================
# 4. Thresholding (EEG-Noise reduzieren)
# ===============================
threshold = 0.2  # Startwert, später variieren

A = A.tocsr()
A.data[A.data < threshold] = 0
A.eliminate_zeros()

print("Non-zero edges (nach Thresholding):", A.nnz)

# ===============================
# 5. Graph erzeugen
# ===============================
G = nx.from_scipy_sparse_array(A)

print(
    f"Graph: {G.number_of_nodes()} Knoten, "
    f"{G.number_of_edges()} Kanten"
)

# ===============================
# 6. Label Propagation Algorithm
# ===============================
def label_propagation(G, initial_labels=None, max_iter=100):
    if initial_labels is None:
        initial_labels = {}

    labels = initial_labels.copy()

    for it in range(max_iter):
        changes = 0
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for node in nodes:
            # feste Labels nicht überschreiben
            if node in initial_labels:
                continue

            neighbor_labels = [
                labels[n]
                for n in G.neighbors(node)
                if n in labels
            ]

            if not neighbor_labels:
                continue

            new_label = Counter(neighbor_labels).most_common(1)[0][0]

            if labels.get(node) != new_label:
                labels[node] = new_label
                changes += 1

        if changes == 0:
            print(f"LPA konvergiert nach {it + 1} Iterationen")
            break

    return labels

# ===============================
# 7. Unsupervised LPA ausführen
# ===============================
labels = label_propagation(G)

# ===============================
# 8. Communities aus Labels bauen
# ===============================
communities = defaultdict(list)
for node, label in labels.items():
    communities[label].append(node)

print(f"\nGefundene Communities: {len(communities)}")
for cid, nodes in communities.items():
    print(f"Community {cid}: {len(nodes)} Knoten")