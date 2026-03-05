# Participants Information
| First Name | Last Name | GitHub Username |
| :--- | :--- | :--- |
| Antonia | SPOERK | @antoniaspoerk |
| Jione | KIM | @ionikim |
| Marina | KOEHLI | @marinakoe |
| Simon | KRUMMENACHER | @simonkrummenacher |

# Spatial Network Evolution in Pediatric EEG during Ictal and Interictal Periods
## Research Questions
1. How does modularity change between interictal (between seizures) and ictal (during seizures)?
2. Do specific electrode groups consistently form communities during seizure periods?
---
## Project Description
The goal of this project is to analyze the temporal evolution of brain connectivity networks during and between epileptic seizures using pediatric EEG recordings. By constructing functional brain networks from EEG signals and applying community detection algorithms, we aim to identify how network structure changes between normal brain activity (interictal) and seizure (ictal) periods.   
We analyze an EEG dataset with epileptic seizures from the CHB-MIT database. The input of our data pipeline consists of electrode recordings from one patient, representing the electrical activity of brain states between and during seizures as a time series. After applying filtering techniques, we segment the EEG signals into 5-second windows to enhance robustness. Next, we compute Pearson correlations across all electrodes within each window to construct a functional connectivity matrix. Here, each weighted connection between electrodes. Repeating this process across all windows produces a sequence of time-resolved functional brain networks. 
Finally, we apply a community detection algorithm (Leiden and/or Louvain) to analyze both the temporal brain states and the spatial recruitment patterns, enabling us to identify which electrodes first join seizure-related communities (the output). 
---
### 1) Network Loading & Data Management (Ji-one Kim, Marina Köhli): 

We use a data set from the CHB-MIT Scalp EEG Database, which contains EEG recordings from 5 male (ages 3–22) and 17 (ages 1.5–19) female pediatric patients with intractable seizures. Recordings typically consist of 23 EEG channels, which were all sampled at 256 Hz with 16-bit resolution. Data, including EEG files and associated demographic data, can be downloaded in EDF (European Data Format) format. Each channel records electrical brain activity over time. For feasibility, we reduced ourselves to analyzing the data of one patient. 

For analysis, we will divide the continuous data into 5-second time windows to enhance signal robustness. This allows us to construct a sequence of networks that represent evolving brain connectivity.  We will select a number of time windows for each of the two states to ensure feasibility, since the whole dataset contains more than seven hours of recordings. 

The EEG recordings will be processed using the MNE-Python library. Processing includes bandpass filtering (1–45 Hz), which excludes low-frequency drift and high-frequency noise. In Python we will manually split the time series into 5-second windows and label them into interictal (between seizure) and ictal (during seizure) with the help of a file containing seizure on/off markers. For each 5-second window, we will construct a functional connectivity matrix by computing pairwise Pearson correlation between all electrodes. 

For each window, a 23 × 23 connectivity matrix will be constructed. To avoid a tense network, we will keep only the strongest correlations. The matrix can then be interpreted as a weighted, undirected graph, where the nodes are electrodes. The edges represent the functional connectivity between electrodes, and the edge weights show the strength of the correlations.  

### 2) Network Exploration & Analytics (Simon Krummenacher, Antonia Spörk): 

We will compute network measures to characterize brain connectivity. To identify the most influential electrodes, we will apply centrality measures: degree centrality and eigenvector centrality. These metrics help us detect nodes that play a key role in signal propagation during seizures. To describe the overall network organization, we will also evaluate structural network properties: network density and average node degree. These metrics capture how strongly connected the network is across different brain states.  

To identify groups of electrodes that interact strongly, we apply community detection algorithms such as: Louvain and Leiden algorithm. These algorithms partition the network into communities of highly connected nodes and allow to analyze whether a specific group of electrodes consistently forms communities during seizures. Network modularity will be used to quantify the strength of community structure. Graph metrics will be implemented from scratch and benchmarked to ensure correctness and computational efficiency. If we have time, we could analyze how the communities change over time by connecting the (single time point) graphs together. 

### 3) Network Visualization (Marina Köhli, Ji-One Kim): 

Our network will be visualized with Gephi to have an interactive interface and matplotlib in Python. In our visualizations we want to 1) compare ictal and interictal networks, 2) show communities in different colors (after applying community detection), 3) analyze the modularity between states and 4) analyze the centrality of each electrode in the network.  

### 4) Analysis and Interpretation (Antonia Spörk, Simon Krummenacher): 

We will benchmark our solution against existing algorithms and evaluate the strengths and limitations of our project. If capacity allows, we might additionally predict other seizure states within/between subjects and/or conduct an influencer analysis of seizure source electrodes. 
