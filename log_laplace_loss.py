
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

gen = np.random.default_rng()


def simulate():
    N = int(1e6)

    for sigma in np.arange(1, 601) / 1000.:
        X = np.maximum(1e-20, gen.laplace(
            loc=1., scale=sigma/np.sqrt(2), size=N))
        print("{:.3f}\t{:.4f}\t%".format(
            sigma, (np.exp(np.log(X).mean()) - 1) * 100))


def fit():
    df = pd.read_csv('laplace_log_loss.csv', names=['std', 'loss'], header=0)
    # ar = np.readtxt('laplace_log_loss.csv')

    # print()

    # print(df.dtypes)

    _, ax = plt.subplots()

    # x, y = df['std'], -df['loss'] #  / 100.   # y was in percent
    x = df['std'].values
    y = 1 + (df['loss'].values / 100.)  # y was in pct
    y = 1 - y  # have it start at 0 and go up instead

    maxstd = .15

    # n = len(x) // 10
    # n = len(x) // 4
    # n = len(x) // 2
    # n = len(x) // 1
    n = (x <= maxstd).sum()

    x, y = x[:n], y[:n]

    # f_pred = np.poly1d(np.polyfit(x[:n], y[:n], deg=3))
    # coeffs = np.polyfit(x[:n], y[:n], deg=5)
    # coeffs = np.polyfit(x[:n], y[:n], deg=4)
    # coeffs = np.polyfit(x[:n], y[:n], deg=3)
    # coeffs = np.polyfit(x, y, deg=2)
    coeffs = np.polyfit(x[:n], y[:n], deg=2)
    print("coeffs: ", coeffs)
    # coeffs /= max(1, coeffs.max())
    f_pred = np.poly1d(coeffs)
    y_hat = f_pred(x)

    # do the polyfit ourselves so we can enforce no offset
    X = np.vstack([x, x * x]).T  # linear + quadratic, no offset
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)
    print("w: ", w)
    # y_hat2 = X @ w
    y_hat3 = X @ w

    X = (x * x).reshape(-1, 1)  # purely quadratic
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)
    print("w: ", w)
    y_hat4 = X @ w

    # y_hat3 = -0.07141908*x + 1.20942464*x*x
    # y_hat4 = 0.76416482*x*x  # yep, looks pretty good

    ax.plot(x, y, lw=4)
    ax.plot(x, y_hat, 'k-', label='ax^2 + bx + c')
    # ax.plot(x, y_hat2)
    ax.plot(x, y_hat3, 'g--', label='ax^2 + bx')
    ax.plot(x, y_hat4, 'r--', label='ax^2')
    # ax.set_ylim([0, 1.05])
    # ax.set_ylim([-.003, .05])
    ax.set_ylim([-.003, .02])
    # ax.semilogx()
    # ax.semilogy()

    plt.legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    # simulate()
    fit()

