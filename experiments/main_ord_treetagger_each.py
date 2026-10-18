import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import mset
from mset import plot as mset_plot
import re
from itertools import chain
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

import treetaggerwrapper as ttw


def count_words(tagger, coms, hinshi):
    coms_tags = [tagger.TagText(com) for com in coms]
    coms_tags = [[tag.split('\t') for tag in com_tags] for com_tags in coms_tags]
    if hinshi == 'meishi':
        coms_words = [[tag[2] for tag in com_tags if len(tag[2])>2 and 'NN' in tag[1]] for com_tags in coms_tags]
    elif hinshi == 'meishi_doshi':
        coms_words = [[tag[2] for tag in com_tags if len(tag[2])>2 and ('NN' in tag[1] or 'VV' in tag[1])] for com_tags in coms_tags]
    else:
        print('hinshi == meishi, meishi_doshi')
        return -1, -1
    words = list(chain.from_iterable(coms_words))
    wordcounter = Counter(words)
    return coms_words, wordcounter

def calc_vec(d_com, n_com, hinshi, stops=0):
    # 単語のカウント
    d_words, d_counts = count_words(tagger, d_com, hinshi)
    n_words, n_counts = count_words(tagger, n_com, hinshi)
    all_words = d_words + n_words
    all_counts = d_counts + n_counts
    
    allkeys = [x for x in sorted(all_counts.items(), key=lambda x:x[1]) if x[1]>1]
    if stops > 0:
        allkeys = allkeys[:-int(len(allkeys)*stops*0.01)] #頻出単語上位stops％ をストップワードとする
    allkeys.sort()
    
    # データの並び替えとラベルづけ
    all_coms = d_com + n_com
    labs = [1] * len(d_words) + [0] * len(n_words)

    wordlist = [keys[0] for keys in allkeys]
    # ベクトル化
    wordlistvec = [np.array([Counter(com_words)[word] for word in wordlist]) for com_words in all_words]
    
    return d_counts, n_counts, all_coms, labs, wordlist, wordlistvec

def calc_rf_feature(outputfiles, d_count, n_count, all_com, labs, wordlist, wordvec, disp_num=10, n_trees=100):
    # randomforest
    clf = ExtraTreesClassifier(max_depth=2, random_state=0, n_estimators=n_trees, class_weight='balanced')
    clf.fit(wordvec, labs)
    y_pred = clf.predict(wordvec)
    print(classification_report(labs, y_pred), file=outputfiles)

    # features
    print("important words", file=outputfiles)
    feature = clf.feature_importances_
    f1 = pd.DataFrame({'number': range(0, len(feature)),'feature': feature[:]})
    f2 = f1.sort_values('feature', ascending=False)
    f3=f2.loc[:,'number']
    indices = np.argsort(feature)[::-1]
    for i in range(disp_num):
        keyword = wordlist[indices[i]]
        print(i+1, keyword, ": ", feature[indices[i]], file=outputfiles)
        print("count: ", d_count[keyword]+n_count[keyword], file=outputfiles)
        print("distinctive: ", d_count[keyword], ", neutral", n_count[keyword], '\n', file=outputfiles)
    print('\n\n', file=outputfiles)
    for i in range(disp_num):
        keyword = wordlist[indices[i]]
        print(i+1, keyword, ": ", feature[indices[i]], file=outputfiles)
        print("count: ", d_count[keyword]+n_count[keyword], file=outputfiles)
        print("distinctive: ", d_count[keyword], ", neutral", n_count[keyword], '\n', file=outputfiles)
        revcoms = [str(lab) + ': '+ com.replace(keyword, '\x1b[31m'+keyword+'\x1b[0m') for (com, lab, vec) in zip(all_com, labs, wordvec) if vec[indices[i]]>0]
        for revcom in revcoms:
            print(revcom, '\n', file=outputfiles)
        print("\n\n", file=outputfiles)



### main function

fname = "../data/amazon_reviews_us_Camera_arranged.tsv"

## read data
df_tmp = pd.read_csv(fname, index_col=3, header=0, delimiter='\t')
df = df_tmp.loc[:, ["star_rating", "review_date", "helpful_votes", "total_votes", "review_body", "product_title"]]
# 日付で昇順に並べ替え
df = df.iloc[::-1]

thr_num = 3000
df_ot = df.loc[df.index.value_counts()>thr_num, :]
#df_ot.loc[:, "star_rating"].hist()
otind = df_ot.index.unique()
print(otind)


tagger = ttw.TreeTagger(TAGLANG='en', TAGDIR='/Users/simonisi/Documents/ResearchTools/TreeTagger')

#hinshi='meishi'
hinshi='meishi_doshi'

for i in range(len(otind)):
    print(i, ": ", otind[i])
    write_file = "./results/amazon_reviews_us_Camera_" + hinshi + "_" + otind[i] + ".txt"
    with open(write_file, mode='w') as f:
        print("product id: ", otind[i], file=f)
        print("product name: ", df.loc[otind[i], "product_title"].values[0] , file=f)

        #for i in range(len(otind)):
        dseq = df_ot.loc[otind[i], "star_rating"].values
        dseq_t = df_ot.loc[otind[i], "review_date"].values
        dseq_hv = df_ot.loc[otind[i], "helpful_votes"].values
        dseq_tv = df_ot.loc[otind[i], "total_votes"].values
        dseq_rev = df_ot.loc[otind[i], "review_body"].values
        dseq_rev = [rev.lower() if type(rev) is str else str(rev) for rev in dseq_rev]
        dseq_rev = np.array([re.sub(r'<.*>|\,|\.|!|\(|\)|;|:|\?', "", rev) for rev in dseq_rev])

        print("len: ", len(dseq), file=f)
        #print(dseq, dseq_t, dseq_rev)
        #print('mean = ', np.mean(dseq))
        #print('median = ', np.median(dseq))

        h, _ = np.histogram(dseq, bins=np.arange(1, 7))
        print(h, file=f)
        #plt.hist(dseq, bins=np.arange(1,7))
        h_prob = mset.obs_to_probs(h)
        #print(h_prob)

        ## param setting
        neu_probs = [[np.sum(h_prob[:thr+1]), 1 - np.sum(h_prob[:thr+1])] for thr in range(4)]
        #print(neu_probs)

        ## make p values table
        # max win size = 10
        win_max = 10
        mord = mset.MSET_ord(h_prob, win_max)
        pvalues = np.array(mord.calc_pvalues_allseq(dseq - 1))  # stars 1..5 -> values 0..4
        dp = mset.find_dperiod(pvalues, 0.01, 'ord')

        # distinctive/neutral 区間の抽出
        dcom = dseq_rev[dp]
        neucom = dseq_rev[~dp]

        # それぞれのコメントの数
        print("poscom: ", np.sum(dseq[dp]>3)+np.sum(dseq[~dp]>3), ", distinctive: ", np.sum(dseq[dp]>3), ", neutral", np.sum(dseq[~dp]>3), file=f)
        print("negcom: ", np.sum(dseq[dp]<3)+np.sum(dseq[~dp]<3), ", distinctive: ", np.sum(dseq[dp]<3), ", neutral", np.sum(dseq[~dp]<3), file=f)
        # 文字列の解析
        d_count_pos, n_count_pos, all_com_pos, labs_pos, wordlist_pos, wordvec_pos = calc_vec(dcom[dseq[dp]>3].tolist(), neucom[dseq[~dp]>3].tolist(), hinshi)
        print("poscomwords: ", np.sum([d_count_pos[key] for key in d_count_pos.keys()])+np.sum([n_count_pos[key] for key in n_count_pos.keys()]), ", distinctive: ", np.sum([d_count_pos[key] for key in d_count_pos.keys()]), "neutral: ", np.sum([n_count_pos[key] for key in n_count_pos.keys()]), file=f)
        print("num of words in features of positive comments: ", len(wordlist_pos), file=f)

        d_count_neg, n_count_neg, all_com_neg, labs_neg, wordlist_neg, wordvec_neg = calc_vec(dcom[dseq[dp]<3].tolist(), neucom[dseq[~dp]<3].tolist(), hinshi)
        print("negcomwords: ", np.sum([d_count_neg[key] for key in d_count_neg.keys()])+np.sum([n_count_neg[key] for key in n_count_neg.keys()]), ", distinctive: ", np.sum([d_count_neg[key] for key in d_count_neg.keys()]), "neutral: ", np.sum([n_count_neg[key] for key in n_count_neg.keys()]), file=f)
        print("num of words in features of negative comments: ", len(wordlist_neg), '\n\n', file=f)

        # ランダムフォレストによる特徴量の重要度
        print("=================", file=f)
        print("positive comments", file=f)
        print("=================", file=f)
        calc_rf_feature(f, d_count_pos, n_count_pos, all_com_pos, labs_pos, wordlist_pos, wordvec_pos, disp_num=20)

        print("=================", file=f)
        print("negative comments", file=f)
        print("=================", file=f)
        calc_rf_feature(f, d_count_neg, n_count_neg, all_com_neg, labs_neg, wordlist_neg, wordvec_neg, disp_num=20)
