from _vivid import pwdist_cuda, pwdot_cuda, pwabsdot_cuda
from _vivid import argmax_cuda, argmin_cuda, min_cuda, max_cuda
from _vivid import DeviceMatrix
import numpy as np

#TODO In the future determine the size of the split automatically
class Kmeans:
    """
    K-means implementations using Lloyd's algorithm
    """
    def __init__(self, init, K, split_size=10000, dist_fun='euclidean', centering='mean'):
        self.num_clust = K
        self.data = init
        self.dist_fun = dist_fun
        self.centering = centering

        #if the cosine distance is used, normalize the data
        #throw out the zero magnitude data items
        if (dist_fun =='cosine') or (self.dist_fun == 'abscosine'):
            self.data = self.normalize_data(self.data)

        #prepare the data splits
        self.num_data = self.data.shape[0]
        self.data_dim = self.data.shape[1]
        self.split_inds = np.arange(0,self.num_data, split_size)
        if self.split_inds[-1] != self.num_data:
            self.split_inds = np.resize(self.split_inds, self.split_inds.size+1)
            self.split_inds[-1] = self.num_data

        self.num_splits = len(self.split_inds) - 1
        self.split_data = []

        self.cluster_ids = np.zeros(self.data.shape[0],dtype='int')

        #select the distance functions and argmin or argmax will be used
        if self.dist_fun == 'euclidean':
            self.dist_measure_cuda = pwdist_cuda
            self.arg_measure_cuda = argmin_cuda
            self.minmax_measure_cuda = min_cuda
            self.minmax_measure= np.min
            self.arg_measure = np.argmin

        elif self.dist_fun == 'cosine':
            self.dist_measure_cuda = pwdot_cuda
            self.arg_measure_cuda = argmax_cuda
            self.minmax_measure_cuda = max_cuda
            self.minmax_measure= np.max
            self.arg_measure = np.argmax

        elif self.dist_fun == 'abscosine':
            self.dist_measure_cuda = pwabsdot_cuda
            self.arg_measure_cuda = argmax_cuda
            self.minmax_measure_cuda = max_cuda
            self.minmax_measure= np.max
            self.arg_measure = np.argmax

        #Initialize the means by randomly sampling the points
        center_init = self._plusplus_init()
        self.centers = self.data[center_init,:]
        #self.centers = self.data[np.random.permutation(self.num_data)[:self.num_clust],:]
       
        self.centers_DM = DeviceMatrix(self.centers.astype('f'))

    def _plusplus_init(self):
        """
        Pseudo-random initialization based on kmeans++
        http://en.wikipedia.org/wiki/K-means%2B%2B
        """
        #select a random index
        init_ind = np.random.randint(self.num_data)
        
        center_inds = [init_ind]
        non_center_inds = range(self.num_data)
        non_center_inds.pop(non_center_inds.index(init_ind))
        
        for i in range(self.num_clust-1):
            print "Init center %d"%i
            cur_sim = np.abs(np.dot(self.data, self.data[center_inds[-1],:].T))
            if i > 0:
                max_sim = np.maximum(cur_sim, max_sim)
            else:
                max_sim = cur_sim

            furthest_ind = non_center_inds[np.argmin(max_sim[non_center_inds])]
            center_inds.append(furthest_ind)

            index_to_pop = non_center_inds.index(furthest_ind)
            non_center_inds.pop(index_to_pop)

        return center_inds

    def normalize_data(self, data):
        data_mags = np.sqrt((data * data).sum(axis=1))
        valid = data_mags > 0
        data_mags = data_mags[valid]
        data = data[valid] / data_mags[:, np.newaxis]
        return data

    def iterate(self):
        self.accumulate()
        self.update()

    def accumulate(self):
        cur_num_clust = len(self.centers)
        self.fitness = 0

        for si in range(self.num_splits):
            lo = self.split_inds[si]
            hi = self.split_inds[si+1]

            dst_DM = self.dist_measure_cuda(DeviceMatrix(self.data[lo:hi]), self.centers_DM)
            clust_id = self.arg_measure_cuda(dst_DM).mat().ravel()
            best_distances = self.minmax_measure_cuda(dst_DM).mat().ravel()
            self.fitness += best_distances.sum()

            self.cluster_ids[lo:hi] = clust_id

    def update(self):
        counts = [np.sum(self.cluster_ids == i) for i in range(len(self.centers))]
        self.count = np.array(counts)
        #throw out the zero member centers
        non_zeros = self.count > 0
        self.centers = self.centers[non_zeros]
        cur_num_clust = len(self.centers)
        self.count = self.count[non_zeros]
        
        if self.centering == 'mean':
            self.total = np.zeros((cur_num_clust,self.data_dim),dtype='f')
            for i in range(cur_num_clust):
                mask = self.cluster_ids == i
                if mask.any():
                    #update the cumulative numbers
                    self.total[i] = self.data[mask].sum(axis=0)
                    self.count[i] = mask.sum()

            self.centers = self.total / self.count[:, np.newaxis]

        elif self.centering == 'medoid':
            for i in range(cur_num_clust):
                mask = self.cluster_ids == i
                sub_data = self.data[mask]
                sub_data_DM = DeviceMatrix(sub_data)
                n_part = 100
                center_similarity = -1e8
                for si in np.arange(0,sub_data.shape[0], n_part):
                    subsub_data = sub_data[si:min(si+n_part,sub_data.shape[0])]
                    subsub_data_DM = DeviceMatrix(subsub_data)

                    pw_dist = self.dist_measure_cuda(subsub_data_DM, sub_data_DM).mat()
                    sum_similarity = pw_dist.sum(axis=1)
                    best_sample = self.arg_measure(sum_similarity)
                    best_similarity = sum_similarity[best_sample]
                    
                    if (self.dist_fun == 'cosine') or (self.dist_fun == 'abscosine'):
                        if best_similarity >= center_similarity:
                            self.centers[i] = subsub_data[best_sample]
                            center_similarity = best_similarity
                    else:
                        if best_similarity <= center_similarity:
                            self.centers[i] = subsub_data[best_sample]
                            center_similarity = best_similarity

        else:
            raise Exception("Vivid - Kmeans", "unknown centering method")

        if self.dist_fun == 'cosine' or self.dist_fun == 'abscosine':
            self.centers = self.normalize_data(self.centers)

        if np.isnan(self.centers).any():
            raise Exception("Vivid - Kmeans", "nans in data")

        self.centers_DM = DeviceMatrix(self.centers.astype('f'))	

