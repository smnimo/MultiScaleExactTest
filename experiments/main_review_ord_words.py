import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import glob
import mset
from mset import plot as mset_plot
import re
from itertools import chain
from collections import Counter

dataflag = 'amzn'
#dataflag = 'yelp'

if dataflag == 'amzn':
    #fname = "../data/amazon_reviews_us_Camera_v1_00_small.tsv"
    fname = "../data/amazon_reviews_us_Camera_06-08.tsv"
    # former main_amazon_rev.py: fname = "../data/amazon_reviews_us_Camera_more100.tsv" with thr_num = 3000

    ## read data
    df_tmp = pd.read_csv(fname, index_col=3, header=0, delimiter='\t')
    df = df_tmp.loc[:, ["star_rating", "review_date", "helpful_votes", "total_votes", "review_body"]]
    # 日付で昇順に並べ替え
    df = df.iloc[::-1]

    # pick up items that are reviewed more than thr
    df.index.value_counts()
    thr_num = 200
    df_ot = df.loc[df.index.value_counts()>thr_num, :]
    #df_ot.loc[:, "star_rating"].hist()
    otind = df_ot.index.unique()
    print(otind)

    ## 
    # item id: 1
    i = 16
    #for i in range(len(otind)):
    dseq = df_ot.loc[otind[i], "star_rating"].values
    dseq_t = df_ot.loc[otind[i], "review_date"].values
    dseq_hv = df_ot.loc[otind[i], "helpful_votes"].values
    dseq_tv = df_ot.loc[otind[i], "total_votes"].values
    dseq_rev = df_ot.loc[otind[i], "review_body"].values
    dseq_rev_word = np.array([re.sub(r'<.*>|\,|\.|!|\(|\)|;|:', "", rev) for rev in dseq_rev])
    print(dseq, dseq_t)
else:
    fname = "../data/yelp_academic_dataset_review_mid.json"
    data_df = pd.read_json(fname, lines=True)

    #print(data_df)

    df_s = data_df.sort_values('date')
    #print(df_s)
    df_s_p = df_s.loc[(df_s['date']<'2017-04-01') & (df_s['date']>='2017-01-01')]
    #print(df_s_p)
    dseq = df_s_p.loc[:, 'stars'].values
    dseq_t = df_s_p.loc[:, 'date'].values
    dseq_hv = df_s_p.loc[:, 'useful'].values
    dseq_rev = df_s_p.loc[:, 'text'].values

## statistict
print('mean = ', np.mean(dseq))
print('median = ', np.median(dseq))

h, _ = np.histogram(dseq, bins=np.arange(1, 7))
print(h)
plt.hist(dseq, bins=np.arange(1,7))
h_prob = mset.obs_to_probs(h)
print(h_prob)

## MSET part
## make p values table
# max win size = 10
win_max = 10
mord = mset.MSET_ord(h_prob, win_max)
## calc pvalues in seq
pvalues = np.array(mord.calc_pvalues_allseq(dseq - 1))  # stars 1..5 -> values 0..4


#### 可視化
#mplt = mset_plot.MSET_plot_ord(mord, dseq - 1, pvalues)
#mplt.plot_pvalues()
#mplt.plot_dperiods()
#
#ind = np.where(np.sum(mplt.dpval, axis=0) > 0)
#print(ind, dseq_t[ind])
#for i in range(len(ind[0])):
#    print("date: ", dseq_t[ind[0][i]], "rev: ", dseq[ind[0][i]], "com: ", dseq_rev.iloc[ind[0][i]], "\n")
#print(mset.find_dperiod(pvalues, 0.01, 'ord'))


## analysis for distinctive periods
dp = mset.find_dperiod(pvalues, 0.01, 'ord')

dcom = dseq_rev[dp]
mincom = dcom[dseq[dp]<3]
mincomword = list(chain.from_iterable([com.split(' ') for com in mincom]))

plscom = dcom[dseq[dp]>3]
plscomword = list(chain.from_iterable([com.split(' ') for com in plscom]))

neucom = dseq_rev[~dp]
nmincom = neucom[dseq[~dp]<3]
nmincomword = list(chain.from_iterable([com.split(' ') for com in nmincom]))
nplscom = neucom[dseq[~dp]>3]
nmplscomword = list(chain.from_iterable([com.split(' ') for com in nplscom]))

## make histogram
allmincounter = Counter(mincomword+nmincomword)
mincounter = Counter(mincomword)
nmincounter = Counter(nmincomword)
print(len(mincomword))

allminkey = [x for x in sorted(allmincounter.items(), key=lambda x:x[1]) if x[1]>1]
allminkey = allminkey[:-15] #top15 をストップワードとする
minhist = mset.obs_to_probs([mincounter[minkeys[0]] for minkeys in allminkey])
nminhist = mset.obs_to_probs([nmincounter[minkeys[0]] for minkeys in allminkey])
plt.plot(minhist)
plt.plot(nminhist)
print(allminkey[100:120])

