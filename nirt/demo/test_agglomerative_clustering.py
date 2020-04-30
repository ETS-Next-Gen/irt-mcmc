import nirt.clustering
import nirt.simulate.simulate_data
import sklearn.cluster
import sklearn.metrics.pairwise
import unittest


class TestAgglomerativeClustering(unittest.TestCase):

    def test_cluster_members(self):
        c = 5
        x, _, _, _ = nirt.simulate.simulate_data.generate_dichotomous_responses(1000, 20, c, permute_items=False)
        print(x.shape)
        d = nirt.clustering.abs_cos_dist(x)
        print(d)
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=c, affinity="precomputed",
                                                             linkage="average").fit(d)
        print(clustering)
        print(clustering.labels_)
        #assert cluster_set.num_clusters == 4
        #assert_array_equal(cluster_set.cluster_members(0), [2, 5])
        #assert_array_equal(cluster_set.cluster_members(1), [0, 4, 7])
        #assert_array_equal(cluster_set.cluster_members(2), [1, 3, 8])
        #assert_array_equal(cluster_set.cluster_members(3), [6])
