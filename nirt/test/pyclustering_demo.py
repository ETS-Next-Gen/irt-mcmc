import numpy as np
import sklearn.metrics.pairwise
import time
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
from pyclustering.utils.metric import type_metric, distance_metric

def abs_cos_distance(x, y):
    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]
    #return sklearn.metrics.pairwise.euclidean_distances(x, y)
    return np.abs(sklearn.metrics.pairwise.cosine_distances(x, y))

if __name__ == "__main__":
    # Load list of points for cluster analysis.
    sample = np.array(read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS))
    print(sample.shape)

    metric = distance_metric(type_metric.USER_DEFINED, func=abs_cos_distance)

    # Prepare initial centers using K-Means++ method.
    initial_centers = kmeans_plusplus_initializer(sample, 2).initialize()

    # Create instance of K-Means algorithm with prepared centers.
    kmeans_instance = kmeans(sample, initial_centers, metric=metric)

    # Run cluster analysis and obtain results.
    start = time.time()
    kmeans_instance.process()
    print("Time", time.time() - start)
    clusters = kmeans_instance.get_clusters()
    final_centers = kmeans_instance.get_centers()

    # Visualize obtained results
    #kmeans_visualizer.show_clusters(sample, clusters, final_centers)
