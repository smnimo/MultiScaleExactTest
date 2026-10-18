""" MSET: multiscale exact test
based on met.py

2019/01/25

listprod, fact, multinom_p, obs_to_probs, multinom_cases and all_multinom_cases come from met.py
(Multinomial Exact Tests), Copyright (c) 2009, R. Dreas Nielsen, GNU General Public License v3 or later.
"""

import copy
from operator import mul
from functools import reduce, lru_cache
import numpy as np
from scipy.stats import multinomial

# probabilities that differ only by rounding count as equal (ties of the exact test, p-value == alpha)
TIES = 1 + 1e-9

## functions
def listprod( alist ):
    """Return the product of all elements of a list."""
    if alist == []:
        return 1
    else:
        return reduce(mul, alist)

def fact(n):
    """Return the factorial."""
    if n is None:
        return 1
    elif n == 1 or n == 0:
        return 1
    else:
        return reduce(mul, range(1, n+1), 1)

def multinom_p(probs, obs):
    """Return the probability of a vector (list) of observations, 'obs', given a probability
    for each class of observation as specified in the vector (list) 'probs'.  The elements
    of 'probs' should sum to 1.0, and both 'probs' and 'obs' should be of the same length.
    No checks on these constraints are made.
    """
    return fact(sum(obs))/listprod(map(fact, obs)) * listprod([pow(probs[i], obs[i]) for i in range(len(probs))])

def obs_to_probs(obs):
    """Standardize a list of observations to be a sequence of probabilities that sum to 1.0."""
    lsum = float(sum(obs))
    if lsum == 0:
        return obs
    else:
        return [ x/lsum for x in obs ]

def multinom_cases( caselist, counts, categories, items ):
    """Add to the list of cases a new case consisting of the 'counts' list, after
    distributing 'items' items over 'categories' remaining categories.
    """
    if categories == 1:
        # There's only one category left to be filled, so put all remaining items in it.
        counts.append(items)
        caselist.append(counts)
    elif items == 0:
        # There are no more items, so put 0 in all remaining categories
        for n in range(categories):
            counts.append(0)
        caselist.append(counts)
    else:
        for n in range(items+1):
            newcounts = copy.copy(counts)
            newcounts.append(n)
            multinom_cases( caselist, newcounts, categories-1, items-n )

def all_multinom_cases( categories, items ):
    """Returns a list of all multinomial combinations (each a list) of 'items' items distributed
    in all possible ways over 'categories' categories."""
    if categories==0 and items>0:
        raise MSET_Error("Can't distribute %d items over 0 cases." % items)
    all_cases = []
    multinom_cases(all_cases, [], categories, items)
    return all_cases


def calc_pvalue_exact( ref_obs, site_obs):
    """Calculate the distribution of two-sided exact test probabilities based on the likelihood of observing the given distribution of reference area samples.  
    Arguments:
        ref_obs		:	A list of reference category counts. neutral な状態での確率分布を与える頻度分布（確率分布でも可）
        site_obs	:	A list of site category counts. 観測
    Cases as likely as site_obs up to rounding (relative 1e-9) are included.
    """
    ref_probs = obs_to_probs(ref_obs)
    site_p = multinom_p(ref_probs, site_obs)
    probs = null_probs(tuple(ref_probs), len(site_obs), int(sum(site_obs)))
    p_value = sum(probs[probs<=site_p*TIES])
    return p_value

@lru_cache(maxsize=None)
def null_probs(ref_probs, categories, items):
    """Probabilities of all cases of all_multinom_cases(categories, items) under ref_probs (a tuple).
    Cached, because calc_pvalue_exact is called for every case of every window size."""
    probs = np.array([multinom_p(ref_probs, case) for case in all_multinom_cases(categories, items)])
    probs.flags.writeable = False
    return probs

def is_significant(pvalues, alpha):
    """p < alpha; a p-value equal to alpha up to rounding (e.g. 0.1**2 with a share of 0.1) is not significant.
    The padding value 2 of calc_pvalues_allseq is never significant."""
    return np.asarray(pvalues, dtype=float)*TIES < alpha

def check_values(seq, nbins):
    """Return seq as an int array, raising MSET_Error unless all values are integers in 0..nbins-1."""
    seq = np.asarray(seq)
    if seq.size and seq.dtype.kind == 'f' and np.all(seq == np.round(seq)):
        seq = seq.astype(int)
    if seq.size and (seq.dtype.kind not in 'iub' or seq.min() < 0 or seq.max() >= nbins):
        raise MSET_Error("values must be integers in 0..%d: %s" % (nbins-1, np.unique(seq)))
    return seq.astype(int)

def find_dperiod(pvalues, alpha, mset_type):
    if mset_type=='cat':
        dpflag = np.zeros_like(pvalues)
        dpflag[is_significant(pvalues, alpha)]=1
        dp = np.zeros(len(pvalues[0]))
        index = np.where(dpflag==1)
        for i in range(len(index[0])):
            win_size = index[0][i]+1
            t = index[1][i]
            dp[t-win_size+1:t+1] = 1
        return dp>0
    elif mset_type=='ord':
        dpflag = np.zeros_like(pvalues)
        dpflag[is_significant(pvalues, alpha)]=1
        dpflag = np.sum(dpflag, axis=2)
        dp = np.zeros(len(pvalues[0]))
        index = np.where(dpflag>0)
        for i in range(len(index[0])):
            win_size = index[0][i]+1
            t = index[1][i]
            dp[t-win_size+1:t+1] = 1
        return dp>0
    else:
        print('mset_type: cat or ord')

def find_dperiod_value(dseq, ms, pvalues, alpha, mset_type):
    """Distinctive values of each time point: a (ms.nbins x len(dseq)) bool array.
    dseq holds values 0..nbins-1, the same sequence as given to calc_pvalues_allseq; row i of the result is value i.
    cat: values whose share in a distinctive window exceeds ms.neu_prob.
    ord: for each distinctive threshold, the levels on the side the window is biased to.
    """
    dseq = check_values(dseq, ms.nbins)
    if mset_type=='cat':
        dpflag = np.zeros_like(pvalues)
        dpflag[is_significant(pvalues, alpha)]=1
        dp = np.zeros([ms.nbins, len(pvalues[0])])
        index = np.where(dpflag==1)
        for i in range(len(index[0])):
            win_size = index[0][i]+1
            t = index[1][i]
            nhist = np.bincount(dseq[t-win_size+1:t+1], minlength=ms.nbins)
            dind = np.where(np.array(obs_to_probs(nhist)) > np.array(ms.neu_prob))
            dp[dind, t-win_size+1:t+1]=1
        return dp>0
    elif mset_type=='ord':
        # union over thresholds, windows and window sizes: for a distinctive threshold thr (values <= thr or > thr),
        # the levels on the side the window is biased to are distinctive values of the whole window
        dpflag = is_significant(pvalues, alpha)
        dp = np.zeros([ms.nbins, len(pvalues[0])])
        index = np.where(dpflag)
        for i in range(len(index[0])):
            win_size = index[0][i]+1
            t = index[1][i]
            thr = index[2][i]
            n_low = np.sum(dseq[t-win_size+1:t+1] <= thr)
            p_low = ms.neu_probs[thr][0]
            if n_low/win_size > p_low:
                dp[:thr+1, t-win_size+1:t+1] = 1
            elif n_low/win_size < p_low:
                dp[thr+1:, t-win_size+1:t+1] = 1
        return dp>0
    else:
        print('mset_type: cat or ord')

## Class
class MSET_Error(Exception):
    pass

class MSET_cat:
    """MSET for a categorical value. Sequences hold integer values 0..nbins-1 (nbins = len(neu_prob)); other values
    raise MSET_Error. calc_pvalues_allseq returns [win_size-1][t] p-values of the windows ending at t (2 where the
    window does not fit)."""
    def __init__(self, neu_prob, win_max):
        self.neu_prob = neu_prob
        self.nbins = len(neu_prob)
        self.win_max = win_max
        self.allpairs = [all_multinom_cases(self.nbins, win_size) for win_size in range(1, win_max+1)]
        self.p_values_list = [[calc_pvalue_exact(neu_prob, pair) for pair in self.allpairs[win_size]] for win_size in range(win_max)]

    def neu_prob_update(self, neu_prob):
        self.neu_prob = neu_prob
        self.p_values_list = [[calc_pvalue_exact(neu_prob, pair) for pair in self.allpairs[win_size]] for win_size in range(self.win_max)]

    def calc_pvalue_seq(self, obs_seq):
        if len(obs_seq) > self.win_max:
            raise MSET_Error("Length of observation must be less than win_max")
            return -1
        key = np.bincount(check_values(obs_seq, self.nbins), minlength=self.nbins)
        return self.p_values_list[np.sum(key)-1][self.allpairs[np.sum(key)-1].index(key.tolist())]

    def calc_pvalue_hist(self, obs_hist):
        if len(obs_hist) != len(self.neu_prob):
            raise MSET_Error("Length of observation must be less than win_max")
            return -1
        if np.sum(obs_hist) > self.win_max:
            raise MSET_Error("Length of observation must be less than win_max")
            return -1
        if type(obs_hist) == list:
            return self.p_values_list[np.sum(obs_hist)-1][self.allpairs[np.sum(obs_hist)-1].index(obs_hist)]
        else:
            return self.p_values_list[np.sum(obs_hist)-1][self.allpairs[np.sum(obs_hist)-1].index(obs_hist.tolist())]

    def calc_pvalues_allseq(self, seq):
        return [[self.calc_pvalue_seq(seq[t-win_size+1:t+1]) if t >= (win_size-1) else 2 for t in range(len(seq))] for win_size in range(1, self.win_max+1)]


class MSET_ord:
    """MSET for an ordinal value: a binomial exact test for each threshold thr (values <= thr or > thr).
    Sequences hold integer values 0..nbins-1; calc_pvalues_allseq returns [win_size-1][t][thr] p-values."""
    def __init__(self, neu_prob, win_max):
        self.nbins = len(neu_prob)
        self.nthrs = self.nbins - 1
        self.neu_probs = [[np.sum(neu_prob[:thr+1]), 1 - np.sum(neu_prob[:thr+1])] for thr in range(self.nthrs)]
        self.win_max = win_max
        self.allpairs = [all_multinom_cases(2, win_size) for win_size in range(1, win_max+1)]
        self.p_values_list = [[[calc_pvalue_exact(self.neu_probs[thr], pair) for thr in range(self.nthrs)] for pair in self.allpairs[win_size]] for win_size in range(win_max)]

    def neu_prob_update(self, neu_prob):
        self.nbins = len(neu_prob)
        self.nthrs = self.nbins - 1
        self.neu_probs = [[np.sum(neu_prob[:thr+1]), 1 - np.sum(neu_prob[:thr+1])] for thr in range(self.nthrs)]
        self.p_values_list = [[[calc_pvalue_exact(self.neu_probs[thr], pair) for thr in range(self.nthrs)] for pair in self.allpairs[win_size]] for win_size in range(self.win_max)]

    def calc_pvalue_seq(self, obs_seq):
        if len(obs_seq) > self.win_max:
            raise MSET_Error("Length of observation must be less than win_max")
            return -1
        key = np.bincount(check_values(obs_seq, self.nbins), minlength=self.nbins)
        key_thr = [[np.sum(key[0:thr+1]), np.sum(key[thr+1:])] for thr in range(self.nthrs)]
        return [self.p_values_list[np.sum(key)-1][self.allpairs[np.sum(key)-1].index(key_thr[thr])][thr] for thr in range(self.nthrs)]

    def calc_pvalues_allseq(self, seq):
        return [[self.calc_pvalue_seq(seq[t-win_size+1:t+1]) if t >= (win_size-1) else (2*np.ones(self.nthrs)).tolist() for t in range(len(seq))] for win_size in range(1, self.win_max+1)]


