# Participants Information
| First Name | Last Name | GitHub Username |
| :--- | :--- | :--- |
| Antonia | SPOERK | @antoniaspoerk |
| Jione | KIM | @ionikim |
| Marina | KOEHLI | @marinakoe |
| Simon | KRUMMENACHER | @simonkrummenacher |

# Temporal Network Evolution in Pediatric EEG During Intractable Seizures
## 👥 Research Questions
1. How does modularity change before and during seizures?  
2. Do specific electrode groups consistently form communities during seizure periods?
---
## 📖 Project Description
This project analyzes how functional brain network organization changes before and during epileptic seizures using the **CHB-MIT EEG dataset**.  
EEG recordings are segmented into **5-second time windows**, and functional connectivity networks are constructed using pairwise correlations between electrodes.  
Graph-theoretical measures such as **degree centrality, eigenvector centrality, density, and modularity** are computed to characterize network structure.  
Community detection algorithms are applied to identify modular organization across preictal and ictal states.  
Key network metrics are partially implemented **from scratch** and benchmarked against standard Python libraries to evaluate correctness and performance.  
The project aims to reveal structural reorganization patterns in brain networks during seizure events.
