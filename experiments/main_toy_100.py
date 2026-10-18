import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import mset
from mset import plot as mset_plot

from sklearn.svm import OneClassSVM
from scipy.stats import multinomial

## param setting
np.random.seed(seed=0)
n_trial = 1000
dist_prob = [0.9, 0.025, 0.05, 0.025]

neu_prob1 = [0.25, 0.25, 0.25, 0.25]
#neu_prob2 = [0.2, 0.25, 0.25, 0.3]
#neu_prob2 = [0.15, 0.25, 0.25, 0.35]
#neu_prob4 = [0.1, 0.2, 0.3, 0.4]
#neu_prob2 = [0.4, 0.3, 0.2, 0.1]
neu_prob2 = [0.3, 0.25, 0.25, 0.2]

neu_prob = [neu_prob1, neu_prob2] #, neu_prob3] #, neu_prob4] #, neu_prob5, neu_prob6]
t1 = 20
t2 = 10
t3 = 20
t4 = 5
t5 = 20
t6 = 8
t7 = 20
dseq_all = []

n_prob = len(neu_prob)

for n in range(n_trial):
    dseq = []
    for i in range(n_prob):
        dseq1 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[i], t1) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t2) == 1)[1]])
        dseq2 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[i], t3) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t4) == 1)[1]])
        dseq3 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[i], t5) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t6) == 1)[1]])
        dseq4 = np.where(np.random.multinomial(1, neu_prob[i], t7) == 1)[1]
        dseq_tmp = np.concatenate([np.concatenate([dseq1, dseq2]), np.concatenate([dseq3, dseq4])])
        dseq.append(dseq_tmp)

    dseq_all.append(dseq)

gt = list(np.arange(t1, t1+t2))+list(np.arange(t1+t2+t3,t1+t2+t3+t4))+list(np.arange(t1+t2+t3+t4+t5,t1+t2+t3+t4+t5+t6))
gtflag = np.zeros(len(dseq_all[0][0]))
gtflag[gt]=1
## mset
## make p values table
# max win size = 10
win_max = 10

mcat = [mset.MSET_cat(nprob, win_max) for nprob in neu_prob]

pvalues = [[mcat[i].calc_pvalues_allseq(dseq_all[n][i]) for i in range(n_prob)] for n in range(n_trial)]
mplt = [[mset_plot.MSET_plot_cat(mcat[i], dseq_all[n][i], pvalues[n][i]) for i in range(n_prob)] for n in range(n_trial)]

for i in range(n_prob):
    print("===============================================")
    print("Toy: ", i+1)
    print("neutral prob: ", neu_prob[i])
    print("====")
    print("MSET")
    print("====")
    ## disp result of MSET
    rec_all = []
    prec_all = []
    f_all = []
    acc_all = []
    for n in range(n_trial):
        mplt[n][i].calc_dperiods()
        est = (np.sum(mplt[n][i].dpval, axis=0)>0)
        TP = np.sum(est & (gtflag==1))
        FN = np.sum(~est & (gtflag==1))
        FP = np.sum(est & (gtflag==0))
        TN = np.sum(~est & (gtflag==0))
        rec = TP / (TP+FN)
        prec = TP / (TP + FP + ((TP+FP)==0))
        f = 2 * rec * prec / (rec + prec + ((rec+prec)==0))
        acc = (TP+TN)/(TP+FN+FP+TN)
        rec_all.append(rec)
        prec_all.append(prec)
        f_all.append(f)
        acc_all.append(acc)
    print("precition: ", np.mean(prec_all))
    print("recall: ", np.mean(rec_all))
    print("f: ", np.mean(f_all))
    print("accuracy: ", np.mean(acc_all))

    print("\n\n==========")
    print("one class SVM")
    print("==========")
    ## one class svm
    win_sizes = [5, 10]

    for win_size in win_sizes:
        rec_all = []
        prec_all = []
        acc_all = []
        f_all = []
        for n in range(n_trial):
            X_train = np.random.multinomial(win_size, neu_prob[i], 1000)
            clf = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
            clf.fit(X_train)
            subseq = []
            for t in range(len(dseq_all[n][i])-win_size+1):
                h_tmp, _ = np.histogram(dseq_all[n][i][t:t+win_size], bins=np.arange(0, 5))
                subseq.append(h_tmp)
            pred = clf.predict(subseq)
            est = np.zeros(len(dseq_all[n][i]))
            for t in range(len(pred)):
                if pred[t] < 0:
                    est[t:t+win_size]=1
            TP = np.sum((est==1) & (gtflag==1))
            FN = np.sum((est==0) & (gtflag==1))
            FP = np.sum((est==1) & (gtflag==0))
            TN = np.sum((est==0) & (gtflag==0))
            rec = TP / (TP+FN)
            prec = TP / (TP + FP + ((TP+FP)==0))
            f = 2 * rec * prec / (rec + prec + ((rec+prec)==0))
            acc = (TP+TN)/(TP+FN+FP+TN)
            rec_all.append(rec)
            prec_all.append(prec)
            f_all.append(f)
            acc_all.append(acc)
        print("window: ", win_size)
        print("precition: ", np.mean(prec_all))
        print("recall: ", np.mean(rec_all))
        print("f: ", np.mean(f_all))
        print("accuracy: ", np.mean(acc_all))
        print("\n")

    print("\n\n==========")
    print("likelihood")
    print("==========")
    # likelihood
    thrs = [0.01, 0.001, 0.0001]
    for win_size in win_sizes:
        for thr in thrs:
            rec_all = []
            prec_all = []
            f_all = []
            acc_all = []
            rv = multinomial(win_size, neu_prob[i])
            for n in range(n_trial):
                subseq = []
                for t in range(len(dseq_all[n][i])-win_size+1):
                    h_tmp, _ = np.histogram(dseq_all[n][i][t:t+win_size], bins=np.arange(0, 5))
                    subseq.append(h_tmp)
                est = np.zeros(len(dseq_all[n][i]))
                pred = rv.pmf(subseq)
                for t in range(len(pred)):
                    if pred[t] < thr:
                        est[t:t+win_size]=1
                TP = np.sum((est==1) & (gtflag==1))
                FN = np.sum((est==0) & (gtflag==1))
                FP = np.sum((est==1) & (gtflag==0))
                TN = np.sum((est==0) & (gtflag==0))
                rec = TP / (TP+FN)
                prec = TP / (TP + FP + ((TP+FP)==0))
                f = 2 * rec * prec / (rec + prec + ((rec+prec)==0))
                acc = (TP+TN)/(TP+FN+FP+TN)
                rec_all.append(rec)
                prec_all.append(prec)
                f_all.append(f)
                acc_all.append(acc)
            print("window: ", win_size)
            print("thr: ", thr)
            print("precition: ", np.mean(prec_all))
            print("recall: ", np.mean(rec_all))
            print("f: ", np.mean(f_all))
            print("accuracy: ", np.mean(acc_all))
            print("\n")
