import numpy as np
import sys

import pathfinder
import vivid

split_size = 1e4
n_clusters = 300

max_iter = 1000
break_thresh = 1e-1

source_file = 'media/image_patches.npy'

target_file = 'media/clusterings.npy'

print "source file: %s"%source_file
print "target file: %s"%target_file

all_patches = np.load(source_file)

#center the data
all_patches -= all_patches.mean(axis=1)[:,np.newaxis]

#filter small magnitude
patch_mags = np.sqrt(np.sum(all_patches * all_patches, axis=1))
valid_mags = patch_mags >= 1e-1
patch_mags = patch_mags[valid_mags]

#set magnitude = 1
all_patches = all_patches[valid_mags] / patch_mags[:,np.newaxis]

k = vivid.Kmeans(all_patches, n_clusters, split_size)
fitness = np.Inf

print 'starting iterations'
for iteration in range(max_iter):
    k.iterate()
    cur_fitness = k.fitness
    fit_diff = fitness - cur_fitness
    print "iter: %d\tvalue: %.2f\tdiff:%.3f"%(iteration, cur_fitness, fit_diff)
    if (iteration > 0)  and (np.abs(fit_diff) < break_thresh):
        break
    fitness = cur_fitness
    sys.stdout.flush()

if iteration == max_iter - 1:
    print "Terminated because the maximum number of iterations is reached."

res_dict_size = len(k.centers)

print "Resulting dictionary size %d"%res_dict_size

valid_ind = np.empty(res_dict_size,dtype='bool')
valid_ind.fill(True)

for i,di in enumerate(k.centers):
    if np.allclose(di,0):
        valid_ind[i] = False

dictionary = k.centers[valid_ind]

np.save(target_file, dictionary)


