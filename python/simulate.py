#!/usr/bin/env python

# import datetime
# import numpy as np
import torch

from . import mathutils


def compute_returns(log2_relchanges, weights, margin_rate_per_timestep=.00125,
                    time_weights=None):
    """
    args:
        log2_relchanges: #timesteps x #assets array of relative changes in
            compounded prices, assuming all dividends are reinvested,
        weights: fraction of portfolio invested in each asset; if sum
         of weights is >1, assumes you're using margin
        margin_cost_per_timestep: no effect if sum of weights is <= 1
        time_weights: 'log2' to weight returns at time steps in proportion to
            log2 of timestep index
    returns: log2 of total relative return, relative returns at each timestep
    """
    # N, D = adjusted_prices.shape
    N, D = log2_relchanges.shape
    # multipliers = torch.log2(adjusted_prices)

    # in theory, no abs; just borrow at the risk-free rate to short stuff; but
    # in reality, shorting a $100 stock uses $100 of margin
    leverage = torch.abs(weights).sum()
    margin = torch.max(0, leverage - 1)
    # assert weights.min() >= 0  # not confident in handling of neg weights
    if time_weights == 'log2':
        time_weights = torch.log2(torch.arange(N))
    if time_weights is not None:
        time_weights /= time_weights.mean()

    relreturns = log2_relchanges @ weights
    if time_weights is not None:
        relreturns *= time_weights
    margin_cost = margin * margin_rate_per_timestep
    relreturns -= margin_cost
    return relreturns.sum(), torch.pow(2, relreturns)


# def optimize_weights(adjusted_prices, initial_weights='kelly', max_iters=100,
def optimize_weights(log2_relchanges, initial_weights='kelly', max_iters=100,
                     max_leverage=1.5, lev_penalty_hparam=10,
                     verbose=1, **simulation_kwargs):
    # N, D = adjusted_prices.shape
    N, D = log2_relchanges.shape
    if initial_weights is None:
        initial_weights = torch.ones(D) / N
    elif initial_weights == 'kelly':
        # initial_weights = kelly_optimize_weights(adjusted_prices)
        initial_weights = kelly_optimize_weights(log2_relchanges)
    weights = torch.tensor(initial_weights, requires_grad=True)

    opt = torch.optim.SGD([weights], lr=.1, momentum=.9)

    # relchanges = mathutils.compute_relative_changes_in_cols(adjusted_prices)
    # log2_relchanges = torch.log2(relchanges)

    for it in range(max_iters):
        log2_ret, _ = compute_returns(
            log2_relchanges, weights, **simulation_kwargs)
        loss = -log2_ret

        leverage = torch.abs(weights).sum()
        lev_diff = torch.max(leverage - max_leverage, -.05)
        # lev_diff = leverage - max_leverage
        # soft_lev_penalty = torch.exp(torch.log(1 + lev_diff))
        # soft_lev_penalty = torch.exp(lev_penalty * 10) * np.sqrt(N)
        soft_lev_penalty = torch.exp(lev_diff * lev_penalty_hparam)
        loss += soft_lev_penalty

        loss.backward()
        opt.step()
        opt.zero_grad()
        if (leverage > max_leverage):
            weights.data = mathutils.l1_project(weights, max_leverage)
        if verbose > 0:
            relreturn = torch.pow(2, log2_ret)
            print(f'{it}): return = {relreturn}')


# def kelly_optimize_weights(adjusted_prices, risk_free_rate=.0001):
def kelly_optimize_weights(log2_relchanges, risk_free_rate=.0001):
    # NOTE: risk free rate needs to use time scale as adjusted prices

    # cov = mathutils.covmat_for_assets(adjusted_prices)
    cov = mathutils.covmat(log2_relchanges)
    # N = len(log2_relchanges)
    # cagrs = (adjusted_prices[-1] / adjusted_prices[0]) ** (1. / N)
    # r = cagrs - risk_free_rate
    r = (2 ** log2_relchanges.mean()) - risk_free_rate

    # numerically stable version of weights = cov^(-1) @ r
    weights, _ = torch.solve(r.reshape(-1, 1), cov)
    return weights


# def assess_weighting_algo(log2_relchanges, train_start_idx,
def assess_weighting_algo(f_weight, log2_relchanges, train_start_idx,
                          train_end_idx, test_end_idx, **simulation_kwargs):
    X_train = log2_relchanges[train_start_idx:train_end_idx]
    X_test = log2_relchanges[train_end_idx:test_end_idx]
    weights = f_weight(X_train)
    return compute_returns(X_test, weights, **simulation_kwargs)


def backtest_weighting_algo(
        f_weight, adjusted_prices_df, start_date='1920-1-1',
        train_nyears=10, test_nyears=5, stride_nyears=1, tick_duration='month'):
    assert tick_duration == 'month'  # TODO daily and annual returns

    # TODO just identify a bunch of start and end indices and append all the
    # results to a list and return it



def main():
    pass


if __name__ == '__main__':
    main()
