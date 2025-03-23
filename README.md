# Overleaf link: https://www.overleaf.com/4387621662zhtvthmxfyyv#9776e7
# DBLP Co-authorship Network Analysis

This project analyzes the DBLP co-authorship network using unsupervised learning techniques to detect community structures and patterns. The analysis is based on the DBLP XML dataset, which contains information about computer science publications and their authors.

## Project Overview

The project implements several techniques for network analysis:
- Graph construction from XML data
- Feature extraction from network structure
- Community detection using the Louvain algorithm
- Spectral clustering for network partitioning
- Gaussian Mixture Models (GMM) for node clustering
- Dimensionality reduction for visualization (PCA, t-SNE)
- Identification of influential authors in each community

## Requirements

The project requires the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- networkx
- scikit-learn
- scipy
- python-louvain (community detection)
- tqdm (progress bars)

You can install these dependencies by running:
```python
!pip install numpy pandas matplotlib seaborn networkx scikit-learn scipy python-louvain tqdm
```

## How to Run the Code

1. **Upload the DBLP XML File to Kaggle**:
   - Go to the "Data" tab in your Kaggle notebook
   - Click "Add Data" → "Upload"
   - Upload your `dblp.xml` file

2. **Import Required Libraries**:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import community as community_louvain
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set(style="whitegrid")
```

3. **Parse the DBLP XML File** (with memory efficiency for large files):
```python
# Specify the path to your uploaded XML file
xml_path = "/kaggle/input/your-dataset-name/dblp.xml"  # Update with your path

# Parse XML to create co-authorship network
G, author_papers = parse_dblp_xml(xml_path, max_publications=1000)
```

4. **Apply Network Analysis Step by Step**:

```python
# Filter the graph to remove isolated nodes
G_filtered = filter_graph(G, min_collaborations=2)

# Extract node features
node_features = extract_features(G_filtered)

# Apply Louvain community detection
louvain_communities = apply_louvain_community_detection(G_filtered)

# Apply spectral clustering
spectral_clusters, spectral_labels = apply_spectral_clustering(G_filtered, n_clusters=10)

# Apply GMM clustering
gmm_clusters, gmm_labels = apply_gmm_clustering(node_features, n_clusters=10)

# Evaluate clustering methods
print("Spectral Clustering:")
evaluate_clustering(node_features, list(spectral_clusters.values()))

print("GMM Clustering:")
evaluate_clustering(node_features, gmm_labels)

# Dimensionality reduction with PCA
X_pca, explained_variance = apply_pca(node_features, n_components=2)

# Visualize the clusters
visualize_clusters(X_pca, list(spectral_clusters.values()), method='PCA + Spectral Clustering')
visualize_clusters(X_pca, gmm_labels, method='PCA + GMM Clustering')

# Apply t-SNE if graph is not too large
if G_filtered.number_of_nodes() <= 5000:
    X_tsne = apply_tsne(node_features)
    visualize_clusters(X_tsne, list(spectral_clusters.values()), method='t-SNE + Spectral Clustering')

# Visualize the network
visualize_network(G_filtered, louvain_communities)

# Find influential authors
influential_authors = find_influential_authors(G_filtered, louvain_communities, top_n=5)

# Print influential authors
for community, authors in sorted(influential_authors.items())[:5]:
    print(f"Community {community}:")
    for author in authors:
        print(f"  - {author}")
```

## Memory Management for Large Datasets

If you're working with a large DBLP XML file, you can adjust the parameters to reduce memory usage:

```python
# Reduce the number of publications to process
G, author_papers = parse_dblp_xml(xml_path, max_publications=500)

# Reduce the number of clusters
spectral_clusters, spectral_labels = apply_spectral_clustering(G_filtered, n_clusters=5)

# Limit visualization to a smaller subgraph
visualize_network(G_filtered, louvain_communities, max_nodes=200)
```

## Interpreting the Results

- **Louvain Communities**: Natural communities based on modularity optimization
- **Spectral Clusters**: Communities detected by treating clustering as a graph-cutting problem
- **GMM Clusters**: Groups based on node feature similarity
- **Silhouette Score**: Measure of clustering quality (higher is better)
- **Influential Authors**: Key researchers in each community based on centrality metrics

## Troubleshooting

1. **Memory Errors**: If you encounter memory issues, reduce the number of publications processed and the chunk size:
   ```python
   G, author_papers = parse_dblp_xml(xml_path, max_publications=500, chunk_size=50000)
   ```

2. **XML Parsing Errors**: If you encounter XML parsing errors related to special characters, the function includes fallback mechanisms to handle these cases.

3. **PCA Error** - If you see `TypeError: 'tuple' object is not callable` when running PCA:
   ```python
   # Ensure PCA is correctly imported before running the function
   from sklearn.decomposition import PCA
   ```
