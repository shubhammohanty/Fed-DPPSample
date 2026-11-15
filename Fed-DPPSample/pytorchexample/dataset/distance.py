import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

# 1. Load Station Info to get the canonical order [0..227]
station_info = pd.read_csv("PeMSD7_M_Station_Info_228.csv")

# Identify the ID column (commonly 'ID' or 'station_id')
id_col = "ID" if "ID" in station_info.columns else station_info.columns[1]
canonical_ids = station_info[id_col].astype(int).tolist()

# 2. Load the unordered adjacency matrix
# index_col=0 sets the first column (IDs) as index; headers are the column IDs
topo_df = pd.read_csv("PeMSD7_Final_Topology.csv", index_col=0)

# Ensure ID types match (cast indices and columns to int)
topo_df.index = topo_df.index.astype(int)
topo_df.columns = topo_df.columns.astype(int)

# 3. Reorder rows and columns to match canonical_ids [0 to 227]
ordered_adj_df = topo_df.reindex(index=canonical_ids, columns=canonical_ids)

# Convert to NumPy array
adj_matrix = ordered_adj_df.to_numpy(dtype=np.float64)

# 4. Handle edge weights for graph traversal
# In PeMS datasets:
# - If Final_Topology contains edge distances (weight > 0 means connected, 0 means no edge):
#   scipy interprets 0 as "no edge" unless directed=False/unweighted.
# - If 0 represents self-loops, keep diagonal 0.
dist_matrix = shortest_path(
    csgraph=adj_matrix,
    method="auto",  # Uses Dijkstra for sparse or Floyd-Warshall for dense
    directed=False,  # Set to True if PeMSD7_Final_Topology is directed
    unweighted=False,  # Set to True if the matrix is binary (0/1 hops)
)

# 5. Save the resulting 228x228 index-based distance matrix
# Here dist_matrix[i, j] corresponds to d(i, j) where i, j in [0, 227]

# Save as raw CSV (no headers or index, purely row i and column j)
np.savetxt("PeMSD7_Shortest_Paths_Indexed.csv", dist_matrix, delimiter=",", fmt="%.4f")

# Or save as a DataFrame with explicit 0..227 indices
output_df = pd.DataFrame(dist_matrix, index=range(228), columns=range(228))
output_df.to_csv("PeMSD7_Shortest_Paths_228x228.csv")

print("Shortest path matrix shape:", dist_matrix.shape)
print("Sample d(0, 1):", dist_matrix[0, 1])
