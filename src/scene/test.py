import numpy as np
from sklearn.neighbors import NearestNeighbors


X = np.random.normal(0, 1, (100, 100))
sample = X[np.random.randint(0, 100, 20)]
nn_searcher = NearestNeighbors(n_neighbors=3)
# nn_searcher.fit(X)


# distances, _ = nn_searcher.kneighbors(X)

distances = distances.min(axis=-1)
print(distances)