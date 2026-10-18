import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import mset
from mset import plot as mset_plot

#read data
i = 0
gazedata = []
with open('../data/gazedata.csv') as f:
    reader = csv.reader(f)
    gaze_user = []
    for row in reader:
        i += 1
        seq = [int(r) for r in row]
        gaze_user.append(seq)
        if i == 5:
            gazedata.append(np.array(gaze_user))
            gaze_user = []
            i = 0

#u = 0
#att = 0
#dseq = gazedata[u][att,:]
#print('mean = ', np.mean(dseq))
#print('median = ', np.median(dseq))
#
#h, _ = np.histogram(dseq, bins=np.arange(1, 5))
#print(h)
#plt.hist(dseq, bins=np.arange(1,5))

## param setting
neu_prob = mset.obs_to_probs(np.ones(3))

## make p values table
win_max = 10

u = 16
mcat = mset.MSET_cat(neu_prob, win_max)
for att in range(5):
    dseq = gazedata[u][att,:] - 1  # values 1..3 -> 0..2
    pvalues = mcat.calc_pvalues_allseq(dseq)
    mplt = mset_plot.MSET_plot_cat(mcat, dseq, pvalues)
    mplt.plot_pvalues()
    mplt.plot_dperiods()
    ind = np.where(np.sum(mplt.dpval, axis=0) > 0)
    
