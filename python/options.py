#!/usr/bin/env python

import collections
import pandas as pd
import numpy as np
from joblib import Memory
import torch

import history

_memory = Memory('.')


class OptionOrder(collections.namedtuple(
        'OptionOrder', 'optiontype ordertype relstrike relprice'.split())):
    pass


def reloutcomes_of_option(order, samples, addtobuff=None):
    if addtobuff is None:
        addtobuff = np.zeros_like(samples)

    strike = order.relstrike
    optype = order.optiontype
    # print("got optype: ", optype)
    assert optype in ('call', 'put')
    ordertype = order.ordertype
    assert ordertype in ('buy', 'sell')

    if optype == 'call':
        mask = samples > strike
        if ordertype == 'buy':
            addtobuff[mask] += samples[mask] - strike
        else:
            addtobuff[mask] -= samples[mask] - strike
    else:
        mask = samples < strike
        if ordertype == 'buy':
            addtobuff[mask] += strike - samples[mask]
        else:
            addtobuff[mask] -= strike - samples[mask]

    offset = -order.relprice if order.ordertype == 'buy' else order.relprice
    return addtobuff + offset


def reloutcomes_of_options(orders, samples):
    # if you just want expected value, you can just compute that separately
    # for each option; but if you want the set of possible outcomes (evaluated
    # at each relative price in samples), you need this function; example use
    # case is finding set of options such that sharpe ratio is maximized, or
    # expectation over max downside is minimized

    # codomain = max(0, samples.min() - .01), samples.max() + .01
    # try_values = np.linspace(codomain[0], codomain[1], npoints)
    ret = np.zeros_like(samples)

    # total_price = np.sum(relprices) if relprices is not None else 0

    # N = len(relstrikes)
    # for i in range(N):

    for order in orders:
        reloutcomes_of_option(order, samples, addtobuff=ret)
        # strike = orders[i].relstrike
        # optype = orders[i].otype
        # assert optype in ('call', 'put')
        # ordertype = orders[i].ordertype
        # assert ordertype in ('buy', 'sell')

        # if optype == 'call':
        #     mask = samples > strike
        #     if ordertype == 'buy':
        #         ret[mask] += samples[mask] - strike
        #     else:
        #         ret[mask] -= samples[mask] - strike
        # else:
        #     mask = samples < strike
        #     if ordertype == 'buy':
        #         ret[mask] += strike - samples[mask]
        #     else:
        #         ret[mask] -= strike - samples[mask]
    # return ret - total_price

    return ret


# def _reloutcomes_of_options(relstrikes, optiontypes, ordertypes, samples,
#                             relprices=None):
#     orders = []
#     if relprices is None:
#         relprices = np.zeros_like(relstrikes)
#     for strike, optype, otype, relprice in zip(
#             relstrikes, optiontypes, ordertypes, relprices):
#         orders.append(OptionOrder(relstrike=strike, optiontype=optype,
#                                   ordertype=otype, relprice=relprice))
#     return reloutcomes_of_options(orders, samples=samples)


# def _optionorders_from_idxs(calls, puts, samples, curprice,
#                             callidxs=None, callordertypes=None,
#                             putidxs=None, putordertypes=None):
#     callidxs = callidxs if callidxs is not None else []
#     putidxs = putidxs if putidxs is not None else []
#     callordertypes = callordertypes if callordertypes is not None else []
#     putordertypes = putordertypes if putordertypes is not None else []

#     orders = []
#     for i, (idx, otype) in enumerate(zip(callidxs, callordertypes)):
#         strike = calls['strike'].values[idx] / curprice
#         otype = callordertypes[i]
#         price = calls['ask'][i] if otype == 'buy' else calls['bid'][i]
#         relprice = price / curprice
#         orders.append(OptionOrder(optiontype='call', ordertype=otype,
#                                   strike=strike, relprice=relprice))
#     for i, (idx, otype) in enumerate(zip(putidxs, putordertypes)):
#         strike = puts['strike'].values[idx] / curprice
#         otype = putordertypes[i]
#         price = puts['ask'][i] if otype == 'buy' else puts['bid'][i]
#         relprice = price / curprice
#         orders.append(OptionOrder(optiontype='put', ordertype=otype,
#                                   strike=strike, relprice=relprice))
#     return orders


# def reloutcomes_of_options_at_idxs(calls, puts, curprice, samples,
#                                    callidxs=None, callordertypes=None,
#                                    putidxs=None, putordertypes=None):
#     orders = _optionorders_from_idxs(calls, puts, curprice,
#                                      callidxs, callordertypes,
#                                      putidxs, putordertypes)
#     return reloutcomes_of_options(orders, samples)


def optionorder_from_idx(df, curprice, idx, optiontype, ordertype):
    strike = df['strike'].iloc[idx]
    relstrike = strike / curprice
    price = df['ask'].iloc[idx] if ordertype == 'buy' else df['bid'].iloc[idx]
    relprice = price / curprice
    return OptionOrder(relstrike=relstrike, relprice=relprice,
                       optiontype=optiontype, ordertype=ordertype)


# def reloutcomes_of_option_at_idx(df, curprice, idx, optiontype,
#                                  ordertype, samples):
#     # print("idx: ", idx)
#     # print("df shape: ", df.shape)
#     order = optionorder_from_idx(df, curprice, idx, optiontype, ordertype)
#     return reloutcomes_of_option(order, samples)


def relvalue_of_call(relstrike, samples, relprice=0):
    """How much you should be wiling to pay for a call option

    returns E[max(0, X - relstrike)], where samples are drawn from X

    also subtracts off relprice at the end

    args:
        relstrike: strike price over current price
        samples: sorted relative returns at appropriate timescale
        relprice:
    returns:
        fraction of current price a call at this price is worth, with
        no discount factor and risk neutrality
    """
    return np.maximum(0, samples - relstrike).mean() - (relprice or 0)

    '''
    More specifically, returns:
        E[max(0, final_rel_price - relstrike)] - E[final_rel_price - relstrike]

    So this is what having the call option is worth *above the net present
    value of just holding the underlying*.
    '''

    # returns = samples - relstrike
    # return np.maximum(0, returns).mean()
    # return np.maximum(0, returns).mean() / (relstrike + 1e-20)
    # mask = returns > relstrike
    # N = mask.sum()
    # if N < 1:
    #     return 0
    # # return samples[mask].mean() * mask.mean()
    # return returns[mask].sum() / len(samples)

    # retmean = samples.mean()
    # retstd = samples.std()
    # gauss_rets = np.random.randn(20000) * retstd + retmean
    # gauss_rets = np.maximum(samples.min(), gauss_rets)

    # rets = currentprice * samples

    # mask = samples < relstrike
    # p_below_strike = mask.mean()
    # E_X_lt_s = samples[mask].mean()
    # return p_below_strike * (relstrike - E_X_lt_s)


def relvalue_of_put(relstrike, samples, relprice=0):
    """expected absolute payoff of buying a put at $0 with a current share
        price of $1, with the given strike price and sample of end prices"""
    return np.maximum(0, relstrike - samples).mean() - (relprice or 0)


# def _infer_underlying_curprice(calls, puts):
#     mask = calls['inTheMoney']
#     low = calls['strike'].loc[mask].max()
#     high = calls['strike'].loc[~mask].min()

#     mask = puts['inTheMoney']
#     put_high = puts['strike'].loc[mask].max()
#     put_low = puts['strike'].loc[~mask].min()

#     low = max(low, put_low)
#     high = min(high, put_high)
#     return (high + low) / 2


def _compute_relvalues(strikes, rets, optiontype, curprice=1., prices=None):
    relstrikes = np.asarray(strikes / curprice)
    if prices is not None:
        relprices = prices / curprice
    else:
        relprices = np.zeros_like(relstrikes)

    # print("relstrikes shape: ", relstrikes.shape)
    # print("relstrikes len: ", len(relstrikes))
    # print("relprices shape: ", relprices.shape)
    # print("relprices len: ", len(relprices))
    # print("rets shape: ", rets.shape)

    assert optiontype in 'call put'.split()
    if optiontype == 'call':
        relvalues = [relvalue_of_call(relstrikes[i], rets, relprices[i])
                     for i in range(len(relstrikes))]
    else:
        relvalues = [relvalue_of_put(relstrikes[i], rets, relprices[i])
                     for i in range(len(relstrikes))]
    return np.array(relvalues)


def _compute_call_relvalues(strikes, rets, curprice=1., prices=None):
    return _compute_relvalues(
        strikes, rets, 'call', curprice=curprice, prices=prices)


def _compute_put_relvalues(strikes, rets, curprice=1., prices=None):
    return _compute_relvalues(
        strikes, rets, 'put', curprice=curprice, prices=prices)


def compute_option_returns(sym, date, rets, curprice=None):
    calls, puts = history.options_for_symbol(sym, date)
    if curprice is None:
        curprice = history._infer_underlying_curprice(calls, puts)

    # calls['breakeven_buy'] = calls['ask'] + calls['strike']
    # puts['breakeven_s'] = puts['strike'] - puts['price']

    # print(calls['strike bid ask'.split()].head(60))
    # return

    # calls['ask'] *= 2  # TODO rm after debug

    # # TODO support midpoint or last or whatever
    # calls['price'] = calls['ask']
    # puts['price'] = puts['ask']

    # calls['relprice'] = calls['ask'] / calls['strike']
    # puts['relprice'] = puts['ask'] / puts['strike']

    print("curprice: ", curprice)

    # calls['breakeven_buy'] = calls['ask'] + calls['strike']
    # calls['breakeven_sell'] = calls['bid'] + calls['strike']
    # puts['breakeven_buy'] = puts['strike'] - puts['ask']
    # puts['breakeven_sell'] = puts['strike'] - puts['bid']

    # # sanity check yf prices
    # # print("calls shape: ", calls.shape)
    # calls = calls.loc[calls['breakeven_buy'] >= curprice]
    # # print("calls shape: ", calls.shape)
    # puts = puts.loc[puts['breakeven_sell'] <= curprice]
    # # print("calls strike shape", calls['strike'].shape)

    calls['maxdownside_buy'] = calls['ask']
    calls['maxdownside_sell'] = np.inf
    puts['maxdownside_buy'] = puts['ask']
    puts['maxdownside_sell'] = puts['strike'] - puts['bid']

    calls['relvalue'] = _compute_call_relvalues(
        calls['strike'], rets, curprice=curprice)
    puts['relvalue'] = _compute_put_relvalues(
        puts['strike'], rets, curprice=curprice)
    for df in [calls, puts]:
        df['value'] = df['relvalue'] * curprice
        df['value_buy'] = df['value'] - df['ask']
        df['value_sell'] = df['bid'] - df['value']
        df['relret_buy'] = 1 + (df['value_buy'] / df['ask'])
        df['relret_sell'] = 1 + (df['value_sell'] / df['bid'])

        # df['adjret_buy'] = df['relret_buy'] / df['maxdownside_buy']
        # df['adjret_sell'] = df['relret_buy'] / df['maxdownside_sell']
        # df['adjret_buy'] = 1 + (df['relret_buy'] - 1) / df['maxdownside_buy']
        # df['adjret_sell'] = 1 + (df['relret_sell'] - 1) / df['maxdownside_sell']
        df['adjret_buy'] = df['value_buy'] / df['maxdownside_buy']
        df['adjret_sell'] = df['value_sell'] / df['maxdownside_sell']

    # keys = ('strike ask breakeven_buy breakeven_sell relvalue value'.split() +
    #         'value_buy value_sell relret_buy relret_sell'.split())
    # for df in [puts]:
    # for name, df in [('puts', puts)]:
    for name, df in [('calls', calls), ('puts', puts)]:
        print('================================ ', name)
        print('------------------------ buy', sym, name, date)
        keys = ('strike bid ask breakeven_buy relvalue value'.split() +
                'value_buy relret_buy adjret_buy'.split())
        # use_df = df.loc[(df['relret_buy'] > 1) & df['relret_buy'].notnull()]
        use_df = df.loc[df['relret_buy'].notnull()]
        print(use_df[keys].head(25))
        print(use_df[keys].tail(58))
        print('------------------------ sell', sym, name, date)
        keys = ('strike bid ask breakeven_sell relvalue value'.split() +
                'value_sell relret_sell adjret_sell'.split())
        # use_df = df.loc[(df['relret_sell'] > 1) & df['relret_sell'].notnull()]
        use_df = df.loc[df['relret_sell'].notnull()]
        print(use_df[keys].head(25))
        print(use_df[keys].tail(58))

    return calls, puts


@_memory.cache
def _compute_individual_option_returns(calls, puts, curprice, samples):
    # just support pairs of options for now
    outcomearrays = {}
    all_zeros = np.zeros_like(samples)
    for optype, df in [('call', calls), ('put', puts)]:
        for ordertype in ('buy', 'sell'):
            for i in range(-1, df.shape[0]):
                key = (optype, ordertype, df['strike'].iloc[i])
                if i < 0:
                    outcomearrays[key] = all_zeros.copy()
                    continue
                optorder = optionorder_from_idx(
                    df, curprice, i,
                    optiontype=optype, ordertype=ordertype)
                outcomearrays[key] = optorder, reloutcomes_of_option(
                    optorder, samples)
                # outcomearrays[key] = reloutcomes_of_option_at_idx(
                #     df=df, curprice=curprice, idx=i, optiontype=optype,
                #     ordertype=ordertype, samples=samples)

    return outcomearrays


def optimize_option_returns(sym, date, samples, curprice=None):
    calls, puts = history.options_for_symbol(sym, date)
    if curprice is None:
        curprice = history._infer_underlying_curprice(calls, puts)

    # print(puts.shape)
    # return
    if sym.upper() == 'TQQQ' and date.startswith('2022'):
        puts = puts.head(65)
        calls = calls.loc[calls['strike'] != 97]  # crap data # TODO rm
        puts = puts.loc[~(puts['strike'].isin([101, 107, 108, 109, 113]))]  # crap data # TODO rm
        puts = puts.loc[puts['strike'] < 145]  # crap data # TODO rm

    outcomearrays = _compute_individual_option_returns(
        calls, puts, curprice, samples)

    best_keys = (-1, -1)  # no options at all
    best_score = 0
    best_outcomes = np.zeros_like(samples)

    # consider all pairs of 2 options
    # tmp = np.zeros_like(samples)
    scores = []
    for k0 in outcomearrays:
        order0, outcomes0 = outcomearrays[k0]
        for k1 in outcomearrays:
            order1, outcomes1 = outcomearrays[k1]
            tmp = outcomes0 + outcomes1
            # expected gain, relative to price, after buying the options; so
            # anything above 0 makes money on avg, and 1 means you expect to
            # make back the current share price when chosen options expire
            meanrelret = tmp.mean()

            # price = 0
            # if order0.ordertype == 'buy':
            #     price += order0.relprice
            # else:
            #     price -= order0.relprice
            # if order1.ordertype == 'buy':
            #     price += order1.relprice
            # else:
            #     price -= order1.relprice

            # lowest return, as a fraction of current share price
            worst_reloutcome = tmp.min()

            # expected gain minus max amount we could lose
            score = meanrelret + worst_reloutcome
            scores.append(score)

            # minret = tmp.min()
            # score = max(0, 1 + meanrelret) / minret + 1e-20
            # score = max(0, meanrelret) / (minret + 1e-20)
            # score = meanrelret
            # if score > 5:
            #     print("k0, k1, score:", k0, k1, score)
            if score > best_score:
                best_score = score
                best_keys = (k0, k1)
                best_outcomes = tmp.copy()

    scores = np.array(scores)
    # print("best score: ", best_score)
    # print("mean std min max score: ", scores.mean(), scores.std(), scores.min(), scores.max())

    return best_keys, best_score, best_outcomes

    # for i in range(-1, ncalls):
    #     for j in range(-1, ncalls):
    #             # outcomes =
    #         # for k in range(nputs):  # only consider calls for now
    #         #     for ll in range(nputs):


def debug_optimize_options(sym, date, samples, curprice=None):
    calls, puts = history.options_for_symbol(sym, date)
    if curprice is None:
        curprice = history._infer_underlying_curprice(calls, puts)

    # print(puts.shape)
    # return
    puts = puts.head(65)

    outcomearrays = _compute_individual_option_returns(
        calls, puts, curprice, samples)

    outcomes = outcomearrays[('call', 'buy', 175)]
    print("outcomes: ")
    print(outcomes[:20])
    print(outcomes[-20:])
    print("samples")
    print(samples[:20])
    print(samples[-20:])
    print("nsamples: ", len(samples))



def _maximin_ratio(outcomes0, outcomes_minratio, outcomes_maxratio,
                   lamda0=1, niters=100):
    """find a ratio lamda such that
            min((outcomes0 + lamda * outcomes_minratio).min(),
                (outcomes0 + lamda * outcomes_maxratio).min())
        is maximized
    """

    lamda = torch.tensor(lamda0, requires_grad=True)

    x0 = torch.tensor(outcomes0)
    xa = torch.tensor(outcomes_minratio)
    xb = torch.tensor(outcomes_maxratio)

    opt = torch.optim.SGD([lamda], .1, momentum=.9)

    for t in range(niters):
        totals_a = x0 + lamda * xa
        totals_b = x0 + lamda * xb
        worst = torch.min(totals_a, totals_b).min()
        # worst_a = totals_a.min()
        # worst_b = totals_b.min()
        # loss = -torch.min(worst_a, worst_b)
        loss = -worst
        loss.backward()
        opt.step()
        opt.zero_grad()

    return lamda, -loss  # lowest possible return


def i_can_haz_arbitrage(op0, op1, minrelratio, maxrelratio, samples=None):
    """
    args:
        op0, op1: OptionOrder objects
        minrelratio, maxrelratio: if the underlying of op0 changes by a factor
        of alpha relative to current price, the underlying of op1 is assumed
        to change by a factor beta, such that
            `minrelratio*alpha <= beta <= maxrelratio*alpha`
        samples: array of relative changes in price of op0 underlying on which
            to compute the returns
    """

    if samples is None:
        samples = np.linspace(-.25, .25, 500)  # +/- 25% in steps of .1%

    # for each relative change in op0, compute smallest and largest relative
    # change in op1
    op0_samples = samples
    op1_minsamples = samples * minrelratio
    op1_maxsamples = samples * maxrelratio

    outcomes0 = reloutcomes_of_option(op0, op0_samples)
    outcomes1min = reloutcomes_of_option(op1, op1_minsamples)
    outcomes1max = reloutcomes_of_option(op1, op1_maxsamples)

    lamda, lowest = _maximin_ratio(outcomes0, outcomes1min, outcomes1max)
    if lowest < 1.00001:
        return lowest, 0

    # TODO allow stuff like 3:2 ratios, instead of just N:1 or 1:N
    if lamda < 1:
        lamdalow = 1. / np.floor(1. / lamda)
        lamdahigh = 1. / np.ceil(1. / lamda)
    else:
        lamdalow = np.floor(1. / lamda)
        lamdahigh = np.floor(1. / lamda)

    totals_a = outcomes0 + lamdalow * outcomes1min
    totals_b = outcomes0 + lamdalow * outcomes1max
    worstlow = np.minimum(totals_a, totals_b).min()

    totals_a = outcomes0 + lamdahigh * outcomes1min
    totals_b = outcomes0 + lamdahigh * outcomes1max
    worsthigh = np.minimum(totals_a, totals_b).min()

    ret = max(worstlow, worsthigh)
    if ret < 1:
        return ret, 0
    retlamda = lamdalow if worstlow > worsthigh else lamdahigh

    return ret, retlamda




# TODO pick up here by finding stuff that's super correlated and checking
# for option arbitrage opportunities (or at least really good deals)

def main_optimize_returns():
    sym = 'tqqq'
    # sym = 'spy'
    # sym = 'qqq'

    resolution = 'month'
    window_len = 18
    option_date = '2022-01-20'

    # resolution = 'day'
    # window_len = 10
    # option_date = '2020-06-18'

    start_date = '2002-01-04' if sym.endswith('qqq') else None
    rets = history.load_pdf(sym, resolution=resolution,
                            window_len=window_len, start_date=start_date)

    keys, score, outcomes = optimize_option_returns(sym, date=option_date, samples=rets, curprice=None)
    print(keys, score)
    print("outcomes.mean(), outcomes.std(), outcomes.min(), outcomes.max()")
    print(outcomes.mean(), outcomes.std(), outcomes.min(), outcomes.max())


def main_compute_returns():
    sym = 'tqqq'
    # sym = 'spy'
    # sym = 'qqq'

    resolution = 'month'
    window_len = 18
    option_date = '2022-01-20'

    # resolution = 'day'
    # window_len = 10
    # option_date = '2020-06-18'

    start_date = '2002-01-04' if sym.endswith('qqq') else None
    rets = history.load_pdf(sym, resolution=resolution,
                            window_len=window_len, start_date=start_date)

    compute_option_returns(sym, date=option_date, rets=rets, curprice=None)


def main_arbitrage():
    sym = 'tqqq'
    # sym = 'spy'
    # sym = 'qqq'


def main():
    # main_compute_returns()
    # main_optimize_returns()
    main_arbitrage()


    # print(calls.shape)


    # rets = np.array([0, 0, 0, 0, 1])
    # # curprice = 90
    # curprice = 100
    # relvalue_of_underlying = rets.mean()
    # # for strike in [0, 10, 20, 40, 70, 80, 90, 160, 320, 640]:
    # #     relstrike = strike / curprice
    # for relstrike in [0, .25, .5, .75, 1.]:
    #     strike = int(curprice * relstrike)
    #     relvalue = relvalue_of_call(relstrike, rets)
    #     # relvalue = relvalue_of_put(relstrike, rets)
    #     # print(f'{strike} = {relstrike}: {relvalue}')
    #     # print('{:4d} = {:5.3f}: {:.3f} ({:.3f})'.format(
    #     #     strike, relstrike, relvalue, relvalue - relvalue_of_underlying))
    #     print('{:4d} = {:5.3f}: {:.3f}'.format(strike, relstrike, relvalue))


if __name__ == '__main__':
    main()
