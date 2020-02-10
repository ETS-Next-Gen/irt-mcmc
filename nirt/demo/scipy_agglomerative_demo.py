import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample


# a custom function that just computes Euclidean distance
def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5

if __name__ == "__main__":
    # Load list of points for cluster analysis.
    sample = np.array(read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS))
    print(sample.shape)

    # Prepare initial centers using K-Means++ method.
    initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()
    fclust1 = fclusterdata(sample, 1.0, metric=mydist)
    fclust2 = fclusterdata(sample, 1.0, metric='euclidean')
    print(np.allclose(fclust1, fclust2))

    clusters = [np.where(fclust1 == c)[0] for c in range(np.max(fclust1)+1)]
    final_centers = [np.mean(cluster, axis=0) for cluster in clusters]

    print(fclust1)
    print(sample)
    print(clusters)
    print(final_centers)

    # Visualize obtained results
    kmeans_visualizer.show_clusters(sample, clusters, final_centers)
