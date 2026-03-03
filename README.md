# Participants Information
| First Name | Last Name | GitHub Username |
| :--- | :--- | :--- |
| Antonia | SPOERK | @antoniaspoerk |
| Jione | KIM | @ionikim |
| Marina | KOEHLI | @marinakoe |
| Simon | KRUMMENACHER | @simonkrummenacher |

# Temporal and Spatial Network Evolution in Pediatric EEG during Ictal and Interictal Periods
## Research Questions
1. How does modularity change between interictal (between seizures) and ictal (during seizures)?
2. Do specific electrode groups consistently form communities during seizure periods?
---
## Project Description
The goal of this project is to analyze the temporal evolution of brain connectivity networks during and between epileptic seizures using pediatric EEG recordings. By constructing functional brain networks from EEG signals and applying community detection algorithms, we aim to identify how network structure changes between normal brain activity (interictal) and seizure (ictal) periods.   
We analyze an EEG dataset with epileptic seizures from the CHB-MIT database. The input of our data pipeline consists of electrode recordings from one patient, representing the electrical activity of brain states between and during seizures as a time series. After applying filtering techniques, we segment the EEG signals into 5-second windows to enhance robustness. Next, we compute Pearson correlations across all electrodes within each window to construct a functional connectivity matrix. Here, each weighted connection between electrodes. Repeating this process across all windows produces a sequence of time-resolved functional brain networks. 
Finally, we apply a community detection algorithm (Leiden and/or Louvain) to analyze both the temporal brain states and the spatial recruitment patterns, enabling us to identify which electrodes first join seizure-related communities (the output). 
