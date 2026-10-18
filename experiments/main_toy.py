import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial
import mset
from mset import plot as mset_plot

## param setting
np.random.seed(seed=4)
dist_prob = [0.9, 0.025, 0.05, 0.025]
neu_prob = [0.25, 0.25, 0.25, 0.25]
neu_prob2 = [0.3, 0.25, 0.25, 0.2]
neu_prob = [neu_prob, neu_prob2]
t1 = 20
t2 = 10
t3 = 20
t4 = 5
t5 = 20
t6 = 8
t7 = 20
dseq1 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[0], t1) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t2) == 1)[1]])
dseq2 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[0], t3) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t4) == 1)[1]])
dseq3 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[0], t5) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t6) == 1)[1]])
dseq4 = np.where(np.random.multinomial(1, neu_prob[0], t7) == 1)[1]
dseq = np.concatenate([np.concatenate([dseq1, dseq2]), np.concatenate([dseq3, dseq4])])
print(dseq)
gt = list(np.arange(t1, t1+t2))+list(np.arange(t1+t2+t3,t1+t2+t3+t4))+list(np.arange(t1+t2+t3+t4+t5,t1+t2+t3+t4+t5+t6))
gtflag = np.zeros(len(dseq))
gtflag[gt]=1
print(gtflag)

dseq21 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[1], t1) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t2) == 1)[1]])
dseq22 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[1], t3) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t4) == 1)[1]])
dseq23 = np.concatenate([np.where(np.random.multinomial(1, neu_prob[1], t5) == 1)[1],np.where(np.random.multinomial(1, dist_prob, t6) == 1)[1]])
dseq24 = np.where(np.random.multinomial(1, neu_prob[1], t7) == 1)[1]
dseq2 = np.concatenate([np.concatenate([dseq21, dseq22]), np.concatenate([dseq23, dseq24])])
dseq = [dseq, dseq2]
print(dseq)


## disp hist
print('mean = ', np.mean(dseq[0]))
print('median = ', np.median(dseq[0]))

h, _ = np.histogram(dseq[0], bins=np.arange(0, 5))
print(h)

print('mean = ', np.mean(dseq[1]))
print('median = ', np.median(dseq[1]))

h, _ = np.histogram(dseq[1], bins=np.arange(0, 5))
print(h)

# counts of the values 0..3 in the two toy sequences, side by side
plt.hist(dseq, bins=np.arange(-0.5, 4.5), label=['seq 1', 'seq 2'])
plt.xticks(np.arange(4))
plt.xlabel('value')
plt.ylabel('count')
plt.legend()

n_probs = len(dseq)  # both toy sequences

## mset
## make p values table
# max win size = 10
win_max = 10

mcat = [mset.MSET_cat(nprob, win_max) for nprob in neu_prob]
pvalues = [mcat[i].calc_pvalues_allseq(dseq[i]) for i in range(n_probs)]
mplt = [mset_plot.MSET_plot_cat(mcat[i], dseq[i], pvalues[i]) for i in range(n_probs)]

## disp result of MSET
for i in range(n_probs):
    mplt[i].plot_pvalues()
    #mplt.plot_dperiods(True)
    mplt[i].plot_dperiods()
    est = (np.sum(mplt[i].dpval, axis=0)>0)
    TP = np.sum(est & (gtflag==1))
    FN = np.sum(~est & (gtflag==1))
    FP = np.sum(est & (gtflag==0))
    TN = np.sum(~est & (gtflag==0))
    rec = TP / (TP+FN)
    prec = TP / (TP + FP)
    acc = (TP+TN)/(TP+FN+FP+TN)
    print("result of MSET, seq:", i+1)
    print("recall: ", rec)
    print("precition: ", prec)
    print("accuracy: ", acc)


cm = mset_plot.generate_cmap(['white', 'black'])
## one class svm
from sklearn.svm import OneClassSVM
win_sizes = [5, 10]
for i in range(n_probs):
    plt.figure()
    for j,win_size in enumerate(win_sizes):
        X_train = np.random.multinomial(win_size, neu_prob[i], 1000)
        clf = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
        clf.fit(X_train)
        subseq = []
        for t in range(len(dseq[i])-win_size+1):
            h_tmp, _ = np.histogram(dseq[i][t:t+win_size], bins=np.arange(0, 5))
            subseq.append(h_tmp)
        pred = clf.predict(subseq)
        est = np.zeros(len(dseq[i]))
        for t in range(len(pred)):
            if pred[t] < 0:
                est[t:t+win_size]=1
        tmp = np.array(est)
        TP = np.sum((est==1) & (gtflag==1))
        FN = np.sum((est==0) & (gtflag==1))
        FP = np.sum((est==1) & (gtflag==0))
        TN = np.sum((est==0) & (gtflag==0))
        rec = TP / (TP+FN)
        prec = TP / (TP + FP)
        plt.subplot(len(win_sizes), 1, j+1)
        plt.imshow(tmp[np.newaxis, :], cmap=cm)
#        plt.grid(True)
        plt.title("seq {}, OCSVM, win:{}, prec:{:.3f}, rec:{:.3f}".format(i+1, win_size, prec, rec))
    #plt.show()

# likelihood
from scipy.stats import multinomial

thrs = [0.01, 0.001, 0.0001]
for i in range(n_probs):
    for win_size in win_sizes:
        plt.figure()
        for j,thr in enumerate(thrs):
            rv = multinomial(win_size, neu_prob[i])
            subseq = []
            for t in range(len(dseq[i])-win_size+1):
                h_tmp, _ = np.histogram(dseq[i][t:t+win_size], bins=np.arange(0, 5))
                subseq.append(h_tmp)
            est = np.zeros(len(dseq[i]))
            pred = rv.pmf(subseq)
            for t in range(len(pred)):
                if pred[t] < thr:
                    est[t:t+win_size]=1
            tmp = np.array(est)
            TP = np.sum((est==1) & (gtflag==1))
            FN = np.sum((est==0) & (gtflag==1))
            FP = np.sum((est==1) & (gtflag==0))
            TN = np.sum((est==0) & (gtflag==0))
            rec = TP / (TP+FN)
            prec = TP / (TP + FP)
            plt.subplot(len(thrs), 1, j+1)
            plt.imshow(tmp[np.newaxis, :], cmap=cm)
#            plt.grid(True)
            plt.title("seq {}, LD, win:{}, thr:{}, prec:{:.3f}, rec:{:.3f}".format(i+1, win_size, thr, prec, rec))
plt.show()
