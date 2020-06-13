
import numpy as np
# from pprint import pprint
import numba
from sklearn import linear_model as linear
import torch


TINY_VAL = 1e-10


# @numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
@numba.njit(fastmath=True)  # njit = no python, cache binary
def compute_relative_changes(seq):
    """more numerically stable than naive solution"""
    multipliers = np.zeros_like(seq)
    multipliers[0] = 1
    cumprod = seq[0]
    for i in range(1, len(seq)):
        multipliers[i] = seq[i] / (cumprod + 1e-20)
        cumprod *= multipliers[i]
    return multipliers


# XXX this function is not useful here since yf gives us "adjusted" prices
# that already take into account dividends;
# see https://ca.help.yahoo.com/kb/finance/adjusted-close-sln28256.html
@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def compute_compound_returns(prices, dividends):
    returns = np.zeros(len(prices))
    initial_price = max(prices[0], TINY_VAL)
    nshares = 1
    for i in range(len(prices)):
        returns[i] = nshares * prices[i] / initial_price
        nshares += nshares * dividends[i] / max(prices[i], TINY_VAL)  # 0 -> 1c

    return returns


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def compute_prices(returns, dividends):
    # returns is price changes with dividends reinvested; yf calls this
    # "adjusted prices"; see
    # https://ca.help.yahoo.com/kb/finance/adjusted-close-sln28256.html
    N = len(returns)
    prices = np.zeros(N)
    final_price = returns[-1]
    prices[N - 1] = final_price
    bonus_shares = dividends[N - 1] / max(prices[N - 1], TINY_VAL)
    for i in range(N - 2, -1, -1):
        multiplier = returns[i] / returns[i + 1]
        prices[i] = multiplier * prices[i + 1]
        prices[i] *= (1 + bonus_shares)

        # compute bonus shares for next iter; yf multiplies *previous* returns
        # by a value less than 1, so this undoes that (in theory)
        bonus_shares = dividends[i] / max(prices[i], TINY_VAL)

    return prices


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def maxdrawdown(mins, maxs=None):
    if maxs is None:
        maxs = mins
    N = len(mins)
    # if N < 2:
    #     return 0
    # cummin = seq[-1]
    cummins = np.full(mins.shape, np.max(mins) + 1)
    cummins[-1] = mins[-1]
    for j in range(N-2, -1, -1):
        cummins[j] = min(mins[j], cummins[j + 1])

    drawdowns = (maxs - cummins) / (maxs + 1e-20)
    return np.max(drawdowns)
    # return drawdowns, cummins
    # idx = np.argmax(drawdowns)
    # return drawdowns[i]


def unexplained_variance_loglinear(returns, weights='sqrt'):
    N = len(returns)
    if weights == 'sqrt':
        # weights = np.sqrt(np.arange(1, N + 1))
        weights = np.log2(np.arange(1, N + 1))
        weights /= weights.sum()

    # print("returns shape", returns.shape)

    X = np.arange(N).reshape(-1, 1)
    y = np.log2(returns)
    est = linear.LinearRegression().fit(X, y, weights)
    yhat = est.predict(X)

    # mu_y = y.mean()
    ynorm = y - y.mean()
    yerrs = y - yhat
    if weights is None:
        weights = np.ones_like(y)
    sse = (ynorm * ynorm * weights).sum()
    rss = (yerrs * yerrs * weights).sum()

    fraction_unexplained = rss / sse
    return fraction_unexplained


def compute_relative_changes_in_cols(X):
    N, D = X.shaped
    ret = np.empty((N, D), dtype=np.float64)
    for d in range(D):
        X[:, d] = compute_relative_changes(X[:, d])
    return ret


def covmat_for_assets(adjusted_prices):
    """adjusted_prices is #timesteps x #assets"""
    X = compute_relative_changes_in_cols(adjusted_prices)
    X -= X.mean(axis=0)
    return X.T @ X


def covmat(X):
    X = X - X.mean(axis=0)
    return X.T @ X


def corr(x, y):
    # return np.corrcoef(df[col], df0[col])[0, 1]  # returns 2x2 matrix
    x = x - x.mean()
    y = y - y.mean()
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    return (x * y).sum()


def l1_project(v, target_l1_norm=1):
    # XXX this won't make v bigger, only smaller
    # also, not actually projecting onto nearest point on L1 ball; just zeroing
    # out smallest stuff, which is what gets you the most sparsity and is
    # what I actually want

    weights = v.copy()
    # weights.data = weights.data * (max_leverage / leverage)
    # project onto L1 ball by zeroing out smallest weights
    absweights = torch.abs(weights)
    sortidxs = torch.argsort()
    sortvals = absweights[sortidxs]

    target_reduction = absweights.sum() - target_l1_norm
    cumsum = 0
    i = 0
    while(cumsum) < target_reduction:
        cumsum += sortvals[i]
        weights.data[sortidxs[i]] = 0
        i += 1
    return weights
