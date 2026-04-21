# Participants Information
| First Name | Last Name | GitHub Username |
| :--- | :--- | :--- |
| Antonia | SPOERK | @antoniaspoerk |
| Jione | KIM | @ionikim |
| Marina | KOEHLI | @marinakoe |
| Simon | KRUMMENACHER | @simonkrummenacher |

# Temporal and Spatial Network Evolution in Pediatric EEG during Ictal and Interictal Periods
## Research Questions
1. How does the Visibility Graph structure (degree, edge density) change between interictal and ictal periods?
2. Do specific electrodes show consistently higher VG activity during seizure periods?
3. Can community structure in the functional connectivity network derived from VG degree sequences distinguish ictal from interictal brain states?
---
## Project Description
The goal of this project is to analyze the temporal evolution of brain connectivity networks during and between epileptic seizures using pediatric EEG recordings. By constructing Visibility Graphs (VG) from EEG signals and applying network analysis methods, we aim to identify how network structure changes between normal brain activity (interictal) and seizure (ictal) periods.
We analyze an EEG dataset with epileptic seizures from the CHB-MIT database. The input of our data pipeline consists of electrode recordings from one patient (CHB01), representing the electrical activity of brain states 15 seconds before and during a seizure as a time series. We construct a Visibility Graph directly from the raw EEG time series — each timepoint becomes a node, and two timepoints are connected if no taller amplitude value blocks the line of sight between them. This graph-based representation captures the regularity and periodicity of the EEG signal without requiring correlation computation. We analyze how the VG structure evolves second by second across 30 windows, enabling us to identify seizure onset and quantify the transition from complex irregular brain activity to rhythmic ictal activity.
---
### 1) Network Loading & Data Management (Ji-one Kim, Marina Köhli): 

We use a dataset from the CHB-MIT Scalp EEG Database, which contains EEG recordings from 5 male (ages 3–22) and 17 female (ages 1.5–19) pediatric patients with intractable seizures. Recordings consist of 23 EEG channels sampled at 256 Hz with 16-bit resolution. Data is available in EDF (European Data Format). For feasibility, we focus on one patient (CHB01), analyzing a 30-second segment centered on a seizure onset (15s pre-ictal + 15s ictal).
Rather than computing pairwise correlations between electrodes, we construct a Visibility Graph (VG) independently for each electrode's time series. For each 1-second window (256 timepoints at 256 Hz), two timepoints tit_i
ti​ and tjt_j
tj​ are connected by an edge if and only if:
x(tk)<x(ti)+x(tj)−x(ti)tj−ti(tk−ti)∀tk∈(ti,tj)x(t_k) < x(t_i) + \frac{x(t_j) - x(t_i)}{t_j - t_i}(t_k - t_i) \quad \forall t_k \in (t_i, t_j)x(tk​)<x(ti​)+tj​−ti​x(tj​)−x(ti​)​(tk​−ti​)∀tk​∈(ti​,tj​)
This means no intermediate value blocks the line of sight between them. The resulting graph encodes the temporal structure and regularity of the EEG signal: periodic signals (typical of seizures) produce many long-range visibility connections and high node degree, while complex irregular signals (typical of normal brain activity) produce sparse, short-range connections.
The pre-computed adjacency matrix is provided as a sparse matrix of shape 176,640 × 176,640, encoding all VG connections across 23 electrodes × 7,680 timepoints. Node index kk
k maps to electrode ⌊k/7680⌋\lfloor k / 7680 \rfloor
⌊k/7680⌋ at timepoint k mod 7680k \bmod 7680
kmod7680.

### 2) Network Exploration & Analytics (Simon Krummenacher, Antonia Spörk): 

We will compute network measures to characterize brain connectivity. To identify the most influential electrodes, we will apply centrality measures: degree centrality and eigenvector centrality. These metrics help us detect nodes that play a key role in signal propagation during seizures. To describe the overall network organization, we will also evaluate structural network properties: network density and average node degree. These metrics capture how strongly connected the network is across different brain states.  

To identify groups of electrodes that interact strongly, we apply community detection algorithms such as: Louvain and Leiden algorithm. These algorithms partition the network into communities of highly connected nodes and allow to analyze whether a specific group of electrodes consistently forms communities during seizures. Network modularity will be used to quantify the strength of community structure. Graph metrics will be implemented from scratch and benchmarked to ensure correctness and computational efficiency. If we have time, we could analyze how the communities change over time by connecting the (single time point) graphs together. 

### 3) Network Visualization (Marina Köhli, Ji-One Kim): 

We visualize the evolving VG network using two complementary approaches:
1. Cosmograph (GPU-accelerated browser visualization):

cosmograph_nodes_CZ.csv + cosmograph_edges_CZ.csv: CZ electrode VG with timeline (7,680 nodes, 30 windows), enabling second-by-second animation in Cosmograph.
cosmo_freq_nodes_preictal/ictal.csv + cosmo_freq_edges_preictal/ictal.csv: All 23 electrodes, frequency-weighted static comparison (5,888 nodes per file). With electrode-based clustering in Cosmograph, the ictal graph shows tight, well-separated electrode communities while the pre-ictal graph is diffuse — directly reflecting VG edge density differences.

2. Html based interactive visaulization
eeg_vg_comparison_static.html — Compares pre-ictal and ictal circular visibility graphs for a selected electrode, highlighting sparse vs dense connectivity.
eeg_visibility_graph_online_mini_timepoint.html — Animates all 23 electrodes as circular visibility graphs to show seizure evolution over time.
eeg_vg_network_inter.html — Visualizes changing functional connectivity across 23 electrodes during seizure onset.

### 4) Analysis and Interpretation (Antonia Spörk, Simon Krummenacher): 

We will benchmark our solution against existing algorithms and evaluate the strengths and limitations of our project. If capacity allows, we might additionally predict other seizure states within/between subjects and/or conduct an influencer analysis of seizure source electrodes. 


---
## Documentation of work
| Step                                           | Explanation/Sub-steps                                            | Name              |
| :--------------------------------------------- | :--------------------------------------------------------------- | :---------------- |
| Selection of topic and database                |                                                                  | Everyone          |
| Preprocessing                                  | Re-reference and bandpass filter                                 | Marina            |
|                                                | Automate the process for other files, create DataFrame           | Marina and Ji-One |
| Multiplex horizontal visibility graph creation | Select windows (one ictal, one interictal)                       | Ji-One and Marina |
|                                                | Segment DataFrame into windows and averaging                     | Marina            |
|                                                | Create node list, inter- and intra-layer edges, adjacency matrix | Ji-One            |
|                                                | Visualize it in Gephi, compute first exploratory metrics         | Ji-One and Marina |
|                                                | Visualisations and analyses (heatmap, group-level averages and difference analyses) | Ji-One |
| Ordinal Pattern Transition Network creation    | Remove re-referencing, create OPTN, identify electrodes with stronger ictal-interictal differences, visualize it in Gephi                                                                                               | Ji-One            |
| Continuous Multiplex HVG creation              | Apply code from categorical HVG graphs to one continuous window (interictal to ictal) | Marina |
|                                                | Storage in LFS                                                   | Marina            |
| Applied Stream More Algorithm                  | Applied the Stream-Moore community detection algorithm from scratch to HVG adjacency matrix. Algorithm failed to converge due to uniform node degree (~22) and negative modularity (Q ≈ -0.0001), identifying the algorithm as unsuitable for this graph                                                                 | Antonia   
| Applied Hierarchical Clustering                | Designed a sliding window pipeline extracting 23×23 channel correlation matrices from intra-channel temporal connectivity profiles. Applied Ward linkage hierarchical clustering to reveal functional channel groupings                                                                 | Antonia   
| Build a Streamlit Live Demo                    | Built an interactive live demonstration of the sliding window network analysis using Streamlit, with auto-playing animation showing how the EEG functional connectivity network evolves second by second across the interictal-to-ictal transition                                                                 | Antonia   
| Applied Laplacian Spectral Clustering          | Implemented spectral clustering from scratch: constructed weighted adjacency graphs from thresholded correlation matrices, computed the normalised graph Laplacian (L = I - D^(-1/2) A D^(-1/2)), eigendecomposed it to embed channels in eigenspace, and clustered using a custom k-means++ implementation. Applied across all sliding windows to track network topology changes across the seizure transition, revealing a sharp eigengap peak immediately before onset followed by collapse of cluster structure during the ictal phase                                                                 | Antonia  
| Apllied Label Propagation Algorithm (LPA)      | Applied the LPA from scratch to HVG adjacency matrix.  | Simon            |
| checked for benchmarking approaches  1   | Community Detection in Social Networks: An In-depth Benchmarking Study with a Procedure-Oriented Framework -> LPA works best but only with groud truth! Thus we need to evaluate without ground thruth  | Simon            |
| checked for benchmarking approaches  2   | work in progress: find a solution without groundtruth -> Evaluation using Clustering Quality Measures: Clustering quality measures, e.g., SSE (sum of squared errors) or inter- cluster distance  Quality measures used to evaluate community detection should be different from the ones used to find communities.  | Simon            |
