import numpy as np
import sklearn.cluster
import sklearn.metrics.pairwise
import nirt.clustering


if __name__ == "__main__":
    x = np.random.rand(100, 1000)
    y = x / np.linalg.norm(x, axis=1)[:, None]
    d = nirt.clustering.abs_cos_dist(y)
    clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="average").fit(d)
    print(clustering)
    print(clustering.labels_)
