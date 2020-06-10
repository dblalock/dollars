#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

import history


def plot_pdf(sym, resolution='day', window_len=1):
    _, axes = plt.subplots(2, 4, figsize=(16, 8))

    sym = sym.upper()
    rets = history.load_pdf(sym, resolution=resolution, window_len=window_len)
    logrets = np.log2(rets)
    retmean = rets.mean()
    retstd = rets.std()
    gauss_rets = np.random.randn(20000) * retstd + retmean
    gauss_rets = np.maximum(rets.min(), gauss_rets)
    loggauss_rets = np.log2(gauss_rets)
    # logretmean = logrets.mean()
    # logretstd = logrets.std()
    # loggauss_rets = np.random.randn(20000) * logretstd + logretmean
    # gauss_rets = 2 ** loggauss_rets
    # print("log mean, std = ", logretmean, logretstd)
    # print("mean, std = ", 2 ** logretmean, 2 ** logretstd)
    print("mean, std = ", retmean, retstd)

    def distplot(vals, ax, using_gauss=False):
        rug = not using_gauss
        sb.distplot(vals, ax=ax, rug=rug, norm_hist=True)
        sb.distplot(vals, ax=ax, rug=rug, hist=False, norm_hist=True,
                    kde_kws=dict(cumulative=True))

    # for i, ax in enumerate(axes[0]):
    for i, ax in enumerate(axes[:, 0]):
        ax.set_title(f'{sym} {window_len}-{resolution} Relative Returns')
        distplot(rets, ax=ax)
        ax.plot([1, 1], [0, ax.get_ylim()[1]], 'k--')
        # if i == 1:
        #     ax.semilogy()
    # for i, ax in enumerate(axes[1]):
    for i, ax in enumerate(axes[:, 1]):
        ax.set_title(f'{sym} {window_len}-{resolution} Log2 Relative Returns')
        distplot(logrets, ax=ax)
        ax.plot([0, 0], [0, ax.get_ylim()[1]], 'k--')
    for i, ax in enumerate(axes[:, 2]):
        ax.set_title(f'{sym} {window_len}-{resolution} Gauss Relative Returns')
        distplot(gauss_rets, ax=ax, using_gauss=True)
        ax.plot([1, 1], [0, ax.get_ylim()[1]], 'k--')
    for i, ax in enumerate(axes[:, 3]):
        ax.set_title(f'{sym} {window_len}-{resolution} Log2 Gauss Relative Returns')
        distplot(loggauss_rets, ax=ax, using_gauss=True)
        ax.plot([0, 0], [0, ax.get_ylim()[1]], 'k--')
    for ax in axes[1]:
        ax.semilogy()
    for i, ax in enumerate(axes[:, 2]):
        ax.set_xlim(axes[i, 0].get_xlim())
    for i, ax in enumerate(axes[:, 3]):
        ax.set_xlim(axes[i, 1].get_xlim())
    plt.tight_layout()
    plt.savefig(f'../figs/densities/{sym}-{resolution}-{window_len}.pdf')


def main():
    # sym = 'midu'
    sym = 'tqqq'
    # sym = 'spy'
    # sym = '^gspc'
    # sym = 'qqq'
    resolution = 'month'
    window_len = 18
    # window_len = 6
    plot_pdf(sym, resolution, window_len)
    return


if __name__ == '__main__':
    main()
