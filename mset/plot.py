import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from . import core as mset

alpha1 = 0.01
alpha2 = 0.05

def classify_pvalues(pvalues, sig1, sig2, other, padding):
    """Class of each p-value: sig1 for p < alpha1, sig2 for alpha1 <= p < alpha2, other for p >= alpha2,
    padding for the windows that do not fit (p-value 2). Significance as in mset.is_significant."""
    pvalues = np.asarray(pvalues, dtype=float)
    bias_win = np.full_like(pvalues, other)
    bias_win[pvalues == 2] = padding
    bias_win[mset.is_significant(pvalues, alpha2)] = sig2
    bias_win[mset.is_significant(pvalues, alpha1)] = sig1
    return bias_win

class MSET_plot_cat:
    def __init__(self, mcat, seq, pvalues):
        self.mcat = mcat
        self.seq = seq
        self.pvalues = pvalues
        self.bias_win = classify_pvalues(pvalues, 0, 1, 2, 3)
        self.dpval = None

    def plot_pvalues(self):
        ## 可視化
        # pvlaues
        plt.subplot(2, 1, 1)
        im = plt.imshow(self.pvalues, interpolation='nearest')
        plt.colorbar(im)
        plt.grid(True)
        #plt.hot() 

        # bias 区間
        plt.subplot(2, 1, 2)
        cm = generate_cmap(['darkblue', 'green', 'white', 'gray'])
        im = plt.imshow(self.bias_win, interpolation='nearest', cmap=cm, vmin=0, vmax=3)
        plt.colorbar(im)
        plt.grid(True)
        #plt.gray()
        plt.show()

    def plot_dperiods(self, disp_val=False, alpha=alpha1):
        """Plot the distinctive values (row i: value i, drawn with color i+1); self.dpval holds the colors."""
        if disp_val:
            self.print_windows(alpha)
        self.calc_dperiods(alpha)
        dpval = self.dpval * np.arange(1, self.mcat.nbins+1)[:, None]

        self.dpval = dpval
        cm = generate_cmap(['white', 'red', 'yellow', 'green', 'pink'])
        #TODO need to update
        im = plt.imshow(dpval, interpolation='nearest', vmin=0, vmax=4, cmap=cm) 
        plt.grid(True)
        plt.show()

    def calc_dperiods(self, alpha=alpha1):
        """self.dpval: (nbins x T) 0/1, the values whose share in a distinctive window (p < alpha) exceeds neu_prob"""
        self.dpval = mset.find_dperiod_value(self.seq, self.mcat, self.pvalues, alpha, 'cat').astype(float)

    def print_windows(self, alpha=alpha1):
        index = np.where(mset.is_significant(self.pvalues, alpha))
        print(index)
        for i in range(len(index[0])):
            win_size = index[0][i]+1
            t = index[1][i]
            dist_seq = self.seq[t-win_size+1:t+1]
            key_prob = mset.obs_to_probs(np.bincount(mset.check_values(dist_seq, self.mcat.nbins), minlength=self.mcat.nbins))
            print("t: ", t, "win_size: ", win_size)
            print("pvalue: ", self.pvalues[win_size-1][t])
            print("dist_seq: ", dist_seq)
            print("key: ", key_prob)
            print("neu: ", self.mcat.neu_prob)
            print(np.where(np.array(key_prob) > np.array(self.mcat.neu_prob)))

class MSET_plot_ord:
    def __init__(self, mord, seq, pvalues):
        self.mord = mord
        self.seq = seq
        self.pvalues = pvalues
        self.bias_win = classify_pvalues([pvalues[:, :, i] for i in range(mord.nthrs)], 0, 0.5, 1, 2)
        self.dpval = None

    def plot_pvalues(self):
        ## 可視化
        # pvlaues
        for thr in range(self.mord.nthrs):
            plt.subplot(self.mord.nthrs, 2, thr * 2 + 1)
            im = plt.imshow(self.pvalues[:,:,thr], interpolation='nearest')
            plt.hot()
            plt.colorbar(im)
            plt.grid(True)

            # bias 区間
            plt.subplot(self.mord.nthrs, 2, thr * 2 + 2)
            im = plt.imshow(self.bias_win[thr], interpolation='nearest')
            plt.colorbar(im)
            plt.grid(True)
            plt.gray()
        plt.show()

    def plot_dperiods(self, disp_val=False, alpha=alpha1, mode='union'):
        self.calc_dperiods(alpha, mode, disp_val)
        im = plt.imshow(self.dpval, interpolation='nearest') 
        plt.grid(True)
        plt.hot() 
        plt.show()

    def calc_dperiods(self, alpha=alpha1, mode='union', disp_val=False):
        """self.dpval: (nbins x T) 0/1 distinctive values (row i: value i) of the windows with p < alpha.
        mode='union': for each distinctive threshold, the levels on the side the window is biased to, united over
            thresholds, windows and window sizes (mset.find_dperiod_value).
        mode='legacy': the rule of the AAAI 2020 code; the lower side (values <= thr) is marked only at the time points
            where no lower level is marked yet.
        """
        if mode == 'union':
            self.dpval = mset.find_dperiod_value(self.seq, self.mord, self.pvalues, alpha, 'ord').astype(float)
        elif mode == 'legacy':
            self.dpval = self.legacy_dperiods(alpha, disp_val)
        else:
            raise ValueError("mode must be 'union' or 'legacy': %s" % mode)

    def legacy_dperiods(self, alpha=alpha1, disp_val=False):
        dpval = np.zeros([self.mord.nbins, len(self.seq)])
        sig = mset.is_significant(self.pvalues, alpha)
        index = [np.where(sig[:, :, thr]) for thr in range(self.mord.nthrs)]
        for thr in range(self.mord.nthrs):
            for i in range(len(index[thr][0])):
                win_size = index[thr][0][i]+1
                t = index[thr][1][i]
                dist_seq = self.seq[t-win_size+1:t+1]
                key_prob = mset.obs_to_probs(np.bincount(mset.check_values(dist_seq, self.mord.nbins), minlength=self.mord.nbins))
                key_prob = np.sum(key_prob[0:thr+1])
                if disp_val:
                    print("t: ", t, "win_size: ", win_size, "thr: ", thr)
                    print("pvalue: ", self.pvalues[win_size-1][t][thr])
                    print("dist_seq: ", dist_seq)
                    print(key_prob, self.mord.neu_probs[thr][0])

                if key_prob > self.mord.neu_probs[thr][0]:
                    for w in range(win_size):
                        if np.sum(dpval[0:thr, t-w]) == 0:
                            dpval[0:thr+1, t-w]=1
                            if disp_val:
                                print("t: ", t, "w: ", w)
                else:
                    dpval[thr+1:, t-win_size+1:t+1]=1
        return dpval

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)
