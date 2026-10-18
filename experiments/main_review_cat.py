import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import glob
import mset
from mset import plot as mset_plot

dataflag = 'amzn'
#dataflag = 'yelp'

if dataflag == 'amzn':
    fname = "../data/amazon_reviews_us_Camera_v1_00_small.tsv"
    #fname = "../data/amazon_reviews_us_Camera_06-08.tsv"

    ## read data
    df_tmp = pd.read_csv(fname, index_col=3, header=0, delimiter='\t')
    df = df_tmp.loc[:, ["star_rating", "review_date", "helpful_votes", "total_votes", "review_body"]]
    # 日付で昇順に並べ替え
    df = df.iloc[::-1]

    # pick up items that are reviewed more than thr
    df.index.value_counts()
    thr_num = 30
    df_ot = df.loc[df.index.value_counts()>thr_num, :]
    #df_ot.loc[:, "star_rating"].hist()
    otind = df_ot.index.unique()
    print(otind)

    ## 
    # item id: 0
    i = 0
    #for i in range(len(otind)):
    dseq = df_ot.loc[otind[i], "star_rating"].values
    dseq_t = df_ot.loc[otind[i], "review_date"].values
    dseq_hv = df_ot.loc[otind[i], "helpful_votes"].values
    dseq_tv = df_ot.loc[otind[i], "total_votes"].values
    dseq_rev = df_ot.loc[otind[i], "review_body"]
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
    dseq_rev = df_s_p.loc[:, 'text']

## statistict
print('mean = ', np.mean(dseq))
print('median = ', np.median(dseq))

h, _ = np.histogram(dseq, bins=np.arange(1, 7))
print(h)
#plt.hist(dseq, bins=np.arange(1,7))


### MSET part
## param setting
neu_prob = mset.obs_to_probs(h)
print("neu_prob: ", neu_prob)

# max win size = 10
win_max = 10

## make p values table
mcat = mset.MSET_cat(neu_prob, win_max)
## calc pvalues in seq
pvalues = mcat.calc_pvalues_allseq(dseq - 1)  # stars 1..5 -> values 0..4


### 可視化
mplt = mset_plot.MSET_plot_cat(mcat, dseq - 1, pvalues)
mplt.plot_pvalues()
#mplt.plot_dperiods(True)
mplt.plot_dperiods()
ind = np.where(np.sum(mplt.dpval, axis=0) > 0)
print(ind, dseq_t[ind])
for i in range(len(ind[0])):
    print("date: ", dseq_t[ind[0][i]], "rev: ", dseq[ind[0][i]], "com: ", dseq_rev.iloc[ind[0][i]], "\n")

print(mset.find_dperiod(pvalues, 0.01, 'cat'))
