
from __future__ import absolute_import

import datetime
import numpy as np
import os
import pandas as pd
import pickle
import time
import yfinance as yf

# from . import mathutils
import mathutils

# from dateutil.relativedelta import relativedelta

from joblib import Memory

_memory = Memory('.')

# START_DATE = '2004-11-19'  # NOTE: has to be a trading day; 2004 to get GLD
START_DATE = '2009-11-16'  # NOTE: has to be a trading day; 09-11-13 to get DG
# START_DATE = '2002-09-03'  # NOTE: has to be a trading day
END_DATE = '2020-05-01'  # fixed end to pretend all downloaded the same day

HISTORIES_DIR = '../data/histories'

# RISK_FREE_RATE_ANNUAL = 1.012
# RISK_FREE_RATE_MONTHLY = RISK_FREE_RATE_ANNUAL ** (1./12)
RISK_FREE_RATE_MONTHLY = 1.001
RISK_FREE_RATE_ANNUAL = RISK_FREE_RATE_MONTHLY ** 12
MAINTAINANCE_MARGIN = .25

# def _leveraged_df():
LEVERAGED_DF = pd.read_csv('../data/leverage-symbol-mappings.csv')
LEVERAGED_DF['Symbol'] = LEVERAGED_DF['Symbol'].str.upper()
LEVERAGED_DF['LeveragedSymbol'] = LEVERAGED_DF['LeveragedSymbol'].str.upper()
LEVERAGED_SYMBOLS = set(LEVERAGED_DF['LeveragedSymbol'])

LEVERAGED_DF = LEVERAGED_DF.loc[LEVERAGED_DF['isMonthly'] == 0] # TODO support these # noqa
# LEVERAGED_MUTS_DF = LEVERAGED_DF.loc[LEVERAGED_DF['isMonthly'] != 0]
# LEVERAGED_DF.drop('isMonthly', axis=1, inplace=True)
# LEVERAGED_MUTS_DF.drop('isMonthly', axis=1, inplace=True)


if not os.path.exists(HISTORIES_DIR):
    os.makedirs(HISTORIES_DIR)


@_memory.cache
def _get_tickers_df():
    df0 = pd.read_csv('../data/nasdaqlisted.txt', sep='|')
    df1 = pd.read_csv('../data/otherlisted.txt', sep='|')
    df = pd.concat([df0, df1], axis=0, ignore_index=True, sort=False)
    df = df[['Symbol', 'Security Name']]
    # df.drop_duplicates(subset=['Symbol'], keep='last', inplace=True)
    df.rename(columns={'Security Name': 'Name'}, inplace=True)

    lev_symbols = LEVERAGED_DF['LeveragedSymbol']
    # lev_names = LEVERAGED_DF['Leverage'].map(lambda x: str(x))
    lev_names = LEVERAGED_DF['Leverage'].astype('object')
    lev_tickers_df = pd.DataFrame.from_dict({
        'Symbol': lev_symbols, 'Name': lev_names})
    df = pd.concat([lev_tickers_df, df], axis=0, ignore_index=True, sort=False)

    df.drop_duplicates(subset=['Symbol'], keep='last', inplace=True)
    df.sort_values(by='Symbol', axis=0, inplace=True)

    return df


# def all_symbols():
#     # return sorted(_get_tickers_df()['Symbol'])
#     ret = _get_tickers_df()['Symbol']
#     return np.array(list(set(ret) + set(LEVERAGED_SYMBOLS)))


def all_symbols_and_names():
    # return sorted(_get_tickers_df()['Symbol'])
    df = _get_tickers_df()
    return df['Symbol'].values, df['Name'].values


def _parse_iso_date(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d')


def _blacklisted_phrases():
    return ('tactical', 'hedged')
    # return ('closed end', 'strategic', 'tactical', 'hedged')


def _blacklisted_symbols():
    return set(['NC'])


@_memory.cache
def _check_symbol_relevant(sym, name='', sizeCutoffBytes=100e3,
                           start_date=START_DATE, minAnnualRet=1.01,
                           minAnnualRetOverDayStd=1.5):
    sym = sym.upper()
    cutoff_date = start_date and _parse_iso_date(start_date) or None
    blacklisted_phrases = _blacklisted_phrases()
    path = _history_path_for_symbol(sym)

    if sym in LEVERAGED_SYMBOLS:
        return True  # we handpicked these; enough history once we impute it

    name = name.lower()
    if any([phrase in name for phrase in blacklisted_phrases]):
        return False  # exclude closed-end funds, tactical funds

    # if name.startwith('^'):
    #     return False  # can't invest in an index directly

    if os.path.getsize(path) < sizeCutoffBytes:
        return False  # too small

    df = pd.read_csv(path, nrows=2)
    if df.shape[0] < 2:
        print(f"WARNING: got shape {df.shape} for symbol {sym}")
        return False  # empty df

    start_date = _parse_iso_date(df['Date'].iloc[0])
    if cutoff_date and (start_date > cutoff_date):
        return False  # too recent

    # check whether returns are consitently above cutoff at various
    # timescales
    df = pd.read_csv(path)
    end_idx = df.shape[0] - 1  # no idea why it can't deal with -1
    end_date = _parse_iso_date(df['Date'].iloc[end_idx])
    closes = df['Close'].values
    for initial_idx in [0, -750, -1500, -2250]:  # 250 trading days/yr
        try:
            initial_val = max(mathutils.TINY_VAL, closes[initial_idx])
        except IndexError:
            continue  # if not old enough, but passed age test, it's okay
        total_return = df['Close'].values[-1] / initial_val

        start_date = _parse_iso_date(df['Date'].iloc[initial_idx])
        timediff = end_date - start_date
        timediff_years = timediff.days / 365.25
        annualized_ret = total_return ** (1. / timediff_years)
        if annualized_ret < minAnnualRet:
            return False

    # way too volatile
    reldiffs = (df['Close'] - df['Open']) / df['Open']
    if minAnnualRetOverDayStd / reldiffs.std() < minAnnualRetOverDayStd:
        return False

    # # inconsistent volume; not even traded every day
    # if (df['Volume'] > 0).mean() < .95:  # needs pretty good liquidity
    #     return False

    return True


@_memory.cache
def all_relevant_symbols(**kwargs):
    symbols, names = all_symbols_and_names()
    ret = []
    # assert 'SDS' in symbols
    # assert 'SPDN' in symbols
    # assert _check_symbol_relevant('SDS', '-1', **kwargs)
    # assert _check_symbol_relevant('SPDN', '-2', **kwargs)
    # for sym, name in [['SDS', 'foo'], ['SPDN', 'foo']]:
    for sym, name in zip(symbols, names):
        print("checking symbol: ", sym)
        if _check_symbol_relevant(sym, name, **kwargs):
            ret.append(sym)  # found a decent one!
    # assert 'SDS' in ret
    # assert 'SPDN' in ret
    return ret


# def blacklisted_symbols():
#     return ['KF']  # yahoo finance API different than everything on internet

# def save_company_infos():
#     df = _get_tickers_df()

#     marketCaps = []
#     for ticker in df['Symbol'][:20]:
#         print("ticker: ", ticker)
#         try:
#             info = yf.Ticker(ticker).info
#             marketCaps.append(info['marketCap'])
#         except IndexError:
#             marketCaps.append(-1)

#     # print("marketCaps: ", marketCaps)
#     df['marketCap'] = marketCaps

    # print(df.shape)
    # print(df.head())
    # print(df.dtypes)


def _history_path_for_symbol(symbol):
    return os.path.join(HISTORIES_DIR, symbol.upper() + '.csv')


def _download_history_for_symbol(symbol, start='1921-01-01', **kwargs):
    # if startat is not None and symbol < startat:
    #     return

    # TODO this is currently broken for old symbols
    # ...although I remember concluding I couldn't fix it?

    kwargs.setdefault('period', 'max')  # download all data by default
    print(f"downloading history for {symbol}...")
    df = yf.Ticker(symbol).history(start=start, **kwargs)
    df.to_csv(_history_path_for_symbol(symbol))
    time.sleep(max(.5, 1 + np.random.randn()))


def _info_path_for_symbol(sym):
    return f'../data/infos/{sym}.pickle'


def _download_info_for_symbol(sym):
    print(f"downloading info for symbol: {sym}")
    d = yf.Ticker(sym).info
    with open(_info_path_for_symbol(sym), 'wb') as f:
        pickle.dump(d, f)
    time.sleep(max(.5, 1 + np.random.randn()))


def download_infos():
    for symbol in all_relevant_symbols():
        _download_history_for_symbol(symbol)


def load_info(sym):
    path = _info_path_for_symbol(sym)
    if not os.path.exists(path):
        _download_info_for_symbol(sym)
    with open(_info_path_for_symbol(sym), 'rb') as f:
        return pickle.load(f)


def download_histories():
    for symbol in all_symbols():
        _download_history_for_symbol(symbol)


def download_100y_old_histories(startat=None):
    """these fail if you just ask for max"""
    df = pd.read_csv('../data/tickers-over-100-yrs-old.txt', names=['Symbol'])
    symbols = df['Symbol']

    # print(symbols)
    # for sym in symbols[:2]:  # TODO rm after debug
    for sym in symbols:  # TODO rm after debug
        _download_history_for_symbol(
            sym, startat=startat, period=None, start='1921-1-1')

    # symbols that still fail:
        # BF.B, BIO.B, GEF.B, LEN.B, RDS.B, STZ.B, TDW.B,
        # they seem to also not work on the yahoo finance website


# @numba.njit(fastmath=True, cache=True)
# @numba.njit(fastmath=True)
def _extend_history(endval, relChanges, leverage=1):
    N = len(relChanges)
    ret = np.zeros_like(relChanges)
    # want to div by ((relChanges[i] - 1) * leverage + 1);
    # equivalent to dividing by (relChanges[i] * leverage + correction)
    correction = 1 - leverage
    # print("correction: ", correction)
    ret[N - 1] = endval / (relChanges[-1] * leverage + correction)
    # print("divby: ", (relChanges[-1] * leverage + correction))
    if N <= 1:
        return
    for i in range(N - 2, -1, -1):
        # ret[i] = ret[i + 1] * ((relChanges[i] - 1) * leverage + 1)
        ret[i] = ret[i + 1] / (relChanges[i] * leverage + correction)
    return ret


def _impute_history_for_leveraged_etf(
        lev_history_df, base_history_df, leverage):

    startdate = lev_history_df.index[0]
    # print("base_endidx where: ", np.where(base_history_df.index == startdate)[0])
    # print("lev history df start end dates: ", lev_history_df.index.values[0], lev_history_df.index.values[-1])
    # print("base history df start end dates: ", base_history_df.index.values[0], base_history_df.index.values[-1])
    # print("startdate: ", startdate)
    # if lev_history_df.index.value[0] > base_history_df.index.value[0]:
    #     return lev_history_df
    base_endidx = np.where(base_history_df.index == startdate)[0][0]
    # print("base_endidx: ", base_endidx)
    imputed_head = base_history_df.iloc[:base_endidx].copy()

    # pull out daily relative price changes; tricky endpoints here; first
    # relchange is always 1, and we actually want the relchange associated
    # with the first day in the leveraged one, not the day before it
    base_prices = base_history_df['price'].iloc[:base_endidx + 1].values
    base_relchanges = mathutils.compute_relative_changes(base_prices)[1:]

    # print("daily_history df cols", lev_history_df.columns.values)
    # print("tracked_history df cols", base_history_df.columns.values)

    # tracked_prices_head = imputed_head['price'].values
    initialval = lev_history_df['Close'].values[0]
    # relchanges = mathutils.compute_relative_changes(initialval)
    # relchanges = base_df_head['rel24h'].values
    new_closes = _extend_history(initialval, base_relchanges, leverage)
    # new_dividends = np.zeros_like(new_closes)
    # print("initialval: ", initialval)
    # print("leverage: ", leverage)
    # print("imputed_head shape", imputed_head.shape)
    # print("imputed_head['Close']", imputed_head['Close'].shape)
    # print("new_closes", new_closes.shape)
    # print("imputed_head['Close']", imputed_head['Close'].values[:10])
    # print("new_closes", new_closes[:10])
    # print("base_relchanges", base_relchanges[:10])
    # print("base_relchanges", base_relchanges[-3:])
    # print("imputed_head['Close']", imputed_head['Close'].values[-10:])
    # print("new_closes", new_closes[-3:])

    imputed_head['Close'] = new_closes
    imputed_head['Dividends'] = 0  # pessimistic, but I'm okay with that
    imputed_head['relDay'] = -1
    imputed_head['rel24h'] = mathutils.compute_relative_changes(new_closes)
    imputed_head['price'] = imputed_head['Close'] * (  # since no dividends
        lev_history_df['price'].values[0] / lev_history_df['Close'].values[0])

    ret = pd.concat([imputed_head, lev_history_df],
                    axis=0, ignore_index=False, sort=False)

    return ret


def _load_history_for_leveraged_symbol(symbol, start_date=START_DATE):
    # print("LEVERAGED_DF", LEVERAGED_DF)
    symbol = symbol.upper()
    idx = np.where(
        LEVERAGED_DF['LeveragedSymbol'].str.upper() == symbol)[0][0]
    # print("idx: ", idx)
    # print("LEVERAGED_DF['Symbol'].values[idx]", LEVERAGED_DF['Symbol'].values[idx])
    # print("LEVERAGED_DF['Leverage'].values[idx]", LEVERAGED_DF['Leverage'].values[idx])
    tracks_symbol = LEVERAGED_DF['Symbol'].values[idx]
    leverage = float(LEVERAGED_DF['Leverage'].values[idx])

    leveraged_history = _load_history_for_symbol(
        symbol, start_date=None, can_recurse=False)

    # if leveraged symbol itself is old enough, just no need to impute
    if start_date is not None:
        startdatetime = _parse_iso_date(start_date)
        if leveraged_history.index[0] <= startdatetime:
            return leveraged_history[startdatetime:]

    tracks_history = _load_history_for_symbol(
        tracks_symbol, start_date=start_date)
    if tracks_history is None:  # underlying symbol not old enough
        return None

    df = _impute_history_for_leveraged_etf(
        leveraged_history, tracks_history, leverage)

    if start_date is not None:
        if startdatetime < df.index[0]:
            return None   # fail fast if start_date is too early
        df = df[startdatetime:]

    return df


def _load_history_for_symbol(symbol, start_date=START_DATE, can_recurse=True):
    symbol = symbol.upper()

    if can_recurse and symbol in LEVERAGED_SYMBOLS:
        return _load_history_for_leveraged_symbol(symbol, start_date=start_date)

    path = _history_path_for_symbol(symbol)
    if not os.path.exists(path):
        _download_history_for_symbol(symbol)
    df = pd.read_csv(path)
    df.fillna(inplace=True, axis=0, method='ffill')  # forward fill nans

    print("loading history for symbol: ", symbol)

    if start_date is not None:
        # dates = [_parse_iso_date(datestr) for datestr in df['Date']]
        try:
            idx = np.where(df['Date'].values == start_date)[0][0]
        except IndexError:
            return None  # asked for a date before the start date
        # print("start_date: ", start_date)
        # print("idx: ", idx)
        df = df.iloc[idx:]
    # df['Date'] = df['Date'].apply(_parse_iso_date)
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

    # handle missing values
    opens = df['Open'].values
    closes = df['Close'].values
    if opens.max() == 0:
        opens = closes
    if closes.max() == 0:
        closes = opens
    opens = np.maximum(mathutils.TINY_VAL, opens)
    closes = np.maximum(mathutils.TINY_VAL, closes)

    df['rel24h'] = mathutils.compute_relative_changes(closes)
    df['relDay'] = closes / opens
    df['price'] = mathutils.compute_prices(closes, df['Dividends'].values)

    df = df.loc[df['Date'] <= END_DATE]

    return df[['Date', 'Close', 'Dividends',
               'rel24h', 'relDay', 'price']]


@_memory.cache
def _load_monthly_history_for_symbol(symbol, start_date=START_DATE):
    dailydf = _load_history_for_symbol(symbol, start_date)
    if dailydf is None:
        return dailydf  # fail fast if symbol doesn't go back far enough
    df = dailydf.asfreq('M', method='ffill')
    df['maxClose'] = dailydf['Close'].resample('M').max()
    df['minClose'] = dailydf['Close'].resample('M').min()
    df['relret'] = mathutils.compute_relative_changes(df['Close'].values)
    df.drop(['Date', 'rel24h', 'relDay', 'Dividends'], axis=1, inplace=True)
    return df


@_memory.cache
def _get_returns_daily_stds_df_for_symbol(sym, start_date):
    df = _load_history_for_symbol(sym, start_date=start_date)
    priceret_total = df['price'].values[-1] / max(mathutils.TINY_VAL, df['price'].values[0])
    # print("final and initial prices: ",
    #     df['Close'].values[-1], df['Close'].values[0])
    # print("final and initial dates: ",
    #     df['Date'].values[-1], df['Date'].values[0])

    ret_total = df['Close'].values[-1] / df['Close'].values[0]
    ndays = df.shape[0]

    start_datetime = _parse_iso_date(start_date)
    end_datetime = _parse_iso_date(df['Date'].iloc[df.shape[0] - 1])
    timediff = end_datetime - start_datetime
    nyears = timediff.days / 365.25

    ret_24h = ret_total ** (1. / ndays)
    ret_annual = ret_total ** (1. / nyears)
    return dict(
        symbol=sym,
        priceRetTot=priceret_total,
        retTot=ret_total,
        ret24h=ret_24h,
        retAnnual=ret_annual,
        stdDaily=df['relDay'].std(),
        std24h=df['rel24h'].std())


def _monthly_corr_with_qqq(df, start_date=START_DATE, col='relret'):
    df0 = _load_monthly_history_for_symbol('QQQ', start_date=start_date)
    x, y = df[col].values.copy(), df0[col].values.copy()
    return mathutils.corr(x, y)


def _get_monthly_stats_for_symbol(sym, start_date=START_DATE):
    df = _load_monthly_history_for_symbol(sym, start_date=start_date)
    if df is None:
        return None
    # nmonths = df.shape[0]
    # nyears = nmonths / 12
    # ndays = nyears * 365.25
    # print("ndays, nyears = ", ndays, ndays / 365.25)
    startprice = max(mathutils.TINY_VAL, df['price'].values[0])
    priceret_total = df['price'].values[-1] / startprice

    closes = df['Close']
    returns = df['Close'].values
    relret = df['relret'].values.copy()
    ret_total = returns[-1] / returns[0]

    timediff = df.index[-1] - df.index[0]
    # print(diff.days)
    ndays = timediff.days
    nyears = ndays / 365.25
    nmonths = nyears * 12

    # print("ndays: ", ndays)

    ret = {}

    ret['24h'] = ret_total ** (1. / ndays)
    ret['monthly'] = ret_total ** (1. / nmonths)
    ret['annual'] = ret_total ** (1. / nyears)
    ret['relMean'] = df['relret'].mean()  # arithmetic, not geometric
    ret['relMedian'] = df['relret'].median()

    def _key_for_percentile(p):
        return 'monthlyRetPctile={:02d}'.format(p)

    # add in quantiles of returns across months
    percentiles = [1, 5, 10, 50, 90, 95, 99]
    vals = np.percentile(df['relret'].values, percentiles)
    for p, val in zip(percentiles, vals):
        ret[_key_for_percentile(p)] = val

    # compute returns in different time periods

    # print('dates: ')
    # print(df.index.values[-61])
    # print(df.index.values[-121])
    # print(df.index.values[-181])

    # weights = np.sqrt(np.arange(1, len(relret) + 1))
    weights = np.log2(np.arange(1, len(relret) + 1))
    weights /= weights.mean()
    logret = np.log(relret)
    ret['wcagr'] = np.exp((logret * weights).sum()) ** (1. / nyears)
    # ret['wcagr'] = np.exp((logret).sum()) ** (1. / nyears)  # correct

    ret['cagr5y'] = (returns[-1] / returns[-61]) ** (1./5)
    ret['cagr10y'] = (returns[-1] / returns[-121]) ** (1./10)
    # ret['cagr15y'] = (returns[-1] / returns[-181]) ** (1./15)

    ret['cagr2020'] = (closes['2020-4-30'] / closes['2020-1-31']) ** 4
    if start_date < '2007-09-30':
        ret['cagr2008'] = (closes['2009-9-30'] / closes['2007-09-30']) ** (1./2)
        ret['cagr2008fall'] = (closes['2008-10-31'] / closes['2008-8-31']) ** 12
        ret['cagrCrash'] = min(ret['cagr2008'], ret['cagr2020'])

    # TODO add in some sort of weighted cagr that more heavily weights
    # recent returns

    # mu = df['relret'].mean()
    # diffs = df['relret'] - mu
    # ret['pathEfficiency'] = mu / np.abs(diffs).mean()
    # ret['upwardFrac'] = mu / np.abs(diffs).mean()
    # relret = df['relret'].values
    # diffs = relret[1:] - relret[:-

    # print("mu: ", mu)

    # print(sym, 'closes:')
    # print(closes['2007-9-30'])
    # print(closes['2009-9-30'])
    # print(closes['2020-1-31'])
    # print(closes['2020-4-30'])
    # print("ratios: ")
    # print(closes['2007-9-30'] / closes['2009-9-30'])
    # print(closes['2020-1-31'] / closes['2020-4-30'])
    # print("cagrs:")
    # print(ret['cagr2008'])
    # print(ret['cagr2020'])

    # print("cagr15y: ", ret['cagr15y'])
    # print("annual: ", ret['annual'])

    # "risk" metrics
    neg_returns_mask = df['relret'] < 1
    sharpe_denom = returns.std()
    sortino_denom = returns[neg_returns_mask].std()
    # ret['maxDrawdown'] = mathutils.maxdrawdown(returns)
    # drawdowns, cummins = mathutils.maxdrawdown(
    ret['maxDrawdown'] = mathutils.maxdrawdown(  # close instead of maxclose
        mins=df['minClose'].values, maxs=df['Close'].values)
    # peakIdx = np.argmax(drawdowns)
    # peak = returns[peakIdx]
    # peakdate = df.index[peakIdx]
    # valleyIdx = np.where(df['minClose'].values == cummins[peakIdx])[0][0]
    # valleyval = returns[valleyIdx]
    # valleydate = df.index[valleyIdx]
    # print("drawdown stats: ")
    # print("peak   idx, val, date: ", peakIdx, peak, peakdate)
    # print("valley idx, val, date: ", valleyIdx, valleyval, valleydate)

    ret['sharpe'] = (ret['monthly'] - RISK_FREE_RATE_MONTHLY) / sharpe_denom
    ret['sortino'] = (ret['monthly'] - RISK_FREE_RATE_MONTHLY) / sortino_denom

    # scoring functions
    if 'cagr15y' in ret:
        ret['cagrStable'] = np.min([ret['cagr5y'], ret['cagr10y'], ret['cagr15y']])
    else:
        ret['cagrStable'] = np.min([ret['cagr5y'], ret['cagr10y']])

    ret['stableSharpe'] = (ret['cagrStable'] - RISK_FREE_RATE_MONTHLY) / sharpe_denom
    ret['stableSortino'] = (ret['cagrStable'] - RISK_FREE_RATE_MONTHLY) / sortino_denom
    ret['stableQuad'] = ret['cagrStable'] - .55 * df['relret'].var()
    ret['monthStd'] = df['relret'].std()

    # upward movement over total variation distance; on raw returns and log
    # returns; latter makes more sense / doesn't just focus on recent history
    diffs = returns[1:] - returns[:-1]
    ret['upwardFracRaw'] = (returns[-1] - returns[0]) / np.abs(diffs).sum()
    logret = np.log(relret)
    diffs = logret[1:] - logret[:-1]
    ret['upwardFracLog'] = logret.mean() / np.abs(diffs).mean()
    ret['cagrEfficiency'] = 1 + ret['upwardFracLog'] * (ret['annual'] - 1)
    ret['cagrStableEfficiency'] = 1 + ret['upwardFracLog'] * (ret['cagrStable'] - 1)

    frac = mathutils.unexplained_variance_loglinear(returns)
    ret['unexplainedFrac'] = frac
    # ret['cagrRss'] = ret_total / frac
    rss_multiplier = ((1 - frac)**2)
    # rss_multiplier = (1 - frac)
    # rss_multiplier = 1. / frac
    # rss_multiplier = np.exp(-(1 - rss_multiplier) / .1)
    # rss_multiplier = np.exp(-frac)
    rss_multiplier = np.exp(-(1 - rss_multiplier))
    # rss_multiplier = np.exp(-frac * frac / .1)
    ret['cagrRss'] = (ret['wcagr'] - 1) * rss_multiplier + 1

    # drawdown_coef = (1 - ret['maxDrawdown']) / ret['maxDrawdown']
    # drawdown_coef = (1. / ret['maxDrawdown']) - 1  # most extreme version
    # drawdown_coef = 1. - ret['maxDrawdown']
    maxleverage = (1. - MAINTAINANCE_MARGIN) / ret['maxDrawdown']
    maxmargin = max(0, maxleverage - 1) / 1.5  # at least 1.5x margin of safety
    # maxmargin = max(maxmargin, 1)  # would never actually do >2x leverage
    margin_coef = .98 * maxmargin
    drawdown_coef = 1 + margin_coef
    # drawdown_coef = 1. / ret['maxDrawdown']
    ret['maxLev'] = maxleverage
    ret['cagrDrawdown'] = (ret['wcagr'] - 1) * drawdown_coef + 1

    corr = _monthly_corr_with_qqq(df, start_date=start_date)

    corrmultiplier = np.sqrt(2 / (1 + corr))
    ret['cagrCorr'] = (ret['wcagr'] - 1) * corrmultiplier + 1
    ret['cagrCorrDown'] = (ret['cagrDrawdown'] - 1) * corrmultiplier + 1
    ret['cagrRssCorr'] = (ret['cagrRss'] - 1) * corrmultiplier + 1
    # ret['cagrDrawdown'] = ret['annual'] * (1 - ret['maxDrawdown'])
    # ret['cagrAll'] = ret['cagrRssCorr'] * (1 - ret['maxDrawdown'])

    # ret['cagrAll'] = (ret['cagrCorrDown'] - 1) / frac + 1
    ret['cagrAll'] = (ret['cagrCorrDown'] - 1) * rss_multiplier + 1
    ret['wcagrAll'] = (ret['wcagr'] - 1) * (
        drawdown_coef * corrmultiplier * rss_multiplier) + 1

    info = load_info(sym)
    try:
        ret['fwdOverTrailingPE'] = info['forwardPE'] / info['trailingPE']
        ret['peg'] = info['forwardPE'] ** 2 / info['trailingPE']
    except (KeyError, TypeError):
        pass

    # misc other stats
    ret.update(dict(
        symbol=sym,
        priceRetTot=priceret_total,
        retTot=ret_total,
        qqqCorr=corr))

    return ret


# @_memory.cache
def get_monthly_stats_df(start_date=START_DATE):
    dicts = [_get_monthly_stats_for_symbol(sym, start_date)
             for sym in all_relevant_symbols(start_date=start_date)]
    dicts = [d for d in dicts if d is not None]  # not enough history
    return pd.DataFrame.from_records(dicts)


@_memory.cache
def get_returns_daily_stds_df(start_date=START_DATE):
    symbols = all_relevant_symbols()
    dicts = []
    for sym in symbols:
        dicts.append(_get_returns_daily_stds_df_for_symbol(sym, start_date))
    return pd.DataFrame.from_records(dicts)


@_memory.cache
def load_master_df(start_date=None, dateCutoff=START_DATE, impute=False):
    sym2closes = {}
    for symbol in all_relevant_symbols(start_date=dateCutoff):
        df = _load_monthly_history_for_symbol(symbol, start_date=start_date)
        sym2closes[symbol] = df['Close'].values

    # need to make all histories the same length if they aren't already
    if start_date is None:
        maxlen = np.max([len(ar) for ar in sym2closes.values()])
        new_sym2closes = {}
        # preprend_buff = np.zeros(maxlen, dtype=np.float)


        # XXX TODO need to actually align stuff based on dates and/or fail
        # fast if start and end dates don't match up like we think they should



        for sym, ar in sym2closes.items():
            newar = np.zeros(maxlen, dtype=np.float)
            newar[-len(ar):] = ar
            new_sym2closes[sym] = newar
        sym2closes = new_sym2closes

    df = pd.DataFrame.from_dict(sym2closes)
    df.fillna(inplace=True, axis=0, method='ffill')  # forward fill nans
    df.fillna(0, inplace=True)  # impute 0s before stock has any valid values
    return df


def _clean_options_df(df):
    # df = df.loc[df['contractSize'] == 'REGULAR']
    df = df.loc[(pd.to_datetime('today') - df['lastTradeDate']).astype('timedelta64[D]') < 10]
    df = df['strike lastPrice bid ask volume impliedVolatility inTheMoney'.split()]
    df.fillna(0, inplace=True)
    maxprices = np.maximum(df['bid'].values, df['ask'].values)
    df = df.loc[maxprices > 0]

    return df


def _infer_underlying_curprice(calls, puts):
    mask = calls['inTheMoney']
    low = calls['strike'].loc[mask].max()
    high = calls['strike'].loc[~mask].min()

    mask = puts['inTheMoney']
    put_high = puts['strike'].loc[mask].max()
    put_low = puts['strike'].loc[~mask].min()

    low = max(low, put_low)
    high = min(high, put_high)
    return (high + low) / 2


@_memory.cache
def options_for_symbol(symbol, date=None, curprice=None):
    calls, puts = yf.Ticker(symbol).option_chain(date=date)
    # print("calls.columns", calls.dtypes)
    if curprice is None:
        curprice = _infer_underlying_curprice(calls, puts)

    calls, puts = _clean_options_df(calls), _clean_options_df(puts)

    calls['breakeven_buy'] = calls['ask'] + calls['strike']
    calls['breakeven_sell'] = calls['bid'] + calls['strike']
    puts['breakeven_buy'] = puts['strike'] - puts['ask']
    puts['breakeven_sell'] = puts['strike'] - puts['bid']

    # sanity check yf prices
    calls = calls.loc[calls['breakeven_buy'] >= curprice]
    puts = puts.loc[puts['breakeven_sell'] <= curprice]

    return calls, puts


@_memory.cache
def load_history(sym, resolution='week', start_date=None):
    if resolution == 'day':
        df = _load_history_for_symbol(sym, start_date=start_date)
        df['relret'] = df['rel24h']
    elif resolution == 'week':
        df = _load_history_for_symbol(sym, start_date=start_date)
        if df is None:
            return None
        df = df.asfreq('W-SUN', method='ffill')
        df['relret'] = mathutils.compute_relative_changes(df['Close'].values)
        df.drop(['rel24h', 'relDay'], axis=1, inplace=True)
    elif resolution == 'month':
        df = _load_monthly_history_for_symbol(sym, start_date=start_date)
    else:
        raise ValueError('resolution must be one of: "{}"'.format(
            '", "'.join('day, week, month'.split())))
    return df


def load_pdf(sym, resolution='day', window_len=1, start_date=None):
    df = load_history(sym, resolution=resolution, start_date=start_date)
    assert window_len >= 1
    closes = df['Close'].values
    # if window_len == 1:
    #     return np.sort(closes)
    return np.sort(closes[window_len:] / closes[:-window_len])
    # return df['Close'].rolling(duration_len) - df['Close'].rolling(duration_len)
    # log2closes = np.log2(df['Close'].values)
    # series = pd.Series(log2closes)
    # return np.pow(2, series.rolling(duration_len).sum())


@_memory.cache
def find_correlations_with(sym='^GSPC', **history_kwargs):
    history_kwargs.setdefault('start_date', '2011-01-04')

    spydf = load_history(sym, **history_kwargs)
    sym_returns = spydf['relret'].values
    sym_returns_std = sym_returns.std()
    # sym_returns_offset = sym_returns - 1
    # sym_returns_normed = sym_returns_offset / np.linalg.norm(sym_returns_offset)  # noqa
    # sym_returns_normed = sym_returns_offset / sym_returns_offset.std()
    # sym_offset_returns = sym_returns - 1
    # histlen = len(sym_closes)

    # sym2corr = {}
    syms = []
    corrs = []
    # minrets = []
    # maxrets = []
    stdratios = []
    gapmins = []
    gapmaxs = []
    gapmeans = []
    gapstds = []
    gappct1s = []
    gappct5s = []
    gappct95s = []
    gappct99s = []

    # for s in ['IVV', 'SPY', 'VV']:
    # for s in ['IVV', 'SPY', 'VV', 'SCHX']:
    # for s in ['IVV', 'SPY', 'VV', 'SPDN', 'SDS']:
    for s in all_relevant_symbols():
        df = load_history(s, **history_kwargs)
        if df is None:
            continue

        returns = df['relret'].values

        corr = mathutils.corr(returns, sym_returns)
        syms.append(s)
        corrs.append(corr)

        # offset_returns = returns - 1
        # returns_normed = offset_returns / np.linalg.norm(offset_returns)
        # returns_normed = offset_returns / offset_returns.std()
        # std_ratio = sym_returns_std / returns.std()
        # stdratios.append(1. / std_ratio)
        std_ratio = np.round(returns.std() / sym_returns_std) * np.sign(corr)
        # std_ratio *= np.sign(corr) or 1  # handle corr of exactly 0
        stdratios.append(int(std_ratio))

        scaled_returns = 1 + ((returns - 1) / (std_ratio + 1e-20))
        # scaled_returns = returns
        diffs = scaled_returns - sym_returns
        # diffs = returns_normed - sym_returns_normed
        gapmins.append(diffs.min())
        gapmaxs.append(diffs.max())
        gapmeans.append(diffs.mean())
        gapstds.append(diffs.std())

        vals = np.percentile(diffs, [1, 5, 95, 99])
        gappct1s.append(vals[0])
        gappct5s.append(vals[1])
        gappct95s.append(vals[2])
        gappct99s.append(vals[3])

    return pd.DataFrame.from_dict({
        'Symbol': syms, 'Corr': corrs, 'stdratio': stdratios,
        'gapmins': gapmins, 'gapmaxs': gapmaxs,
        'gapmeans': gapmeans, 'gapstds': gapstds,
        'gappct5s': gappct5s, 'gappct95s': gappct95s,
        'gappct1s': gappct1s, 'gappct99s': gappct99s,
        })


def main():
    # print(LEVERAGED_DF['LeveragedSymbol'])
    # print(LEVERAGED_DF['LeveragedSymbol'].isin(['SDS']))
    # assert 'SDS' in list(LEVERAGED_DF['LeveragedSymbol'].values)

    # df = _get_tickers_df()
    # assert 'SDS' in df['Symbol'].values
    # return


    # # syms = all_symbols_and_names()['Symbol'].values
    # # syms, _ = all_symbols_and_names()
    # syms = all_relevant_symbols()
    # assert 'SDS' in syms
    # assert 'SPDN' in syms
    # print("has stuff I want: ", 'SPDN' in list(syms), 'SDS' in list(syms))
    # # print("list has stuff I want: ", 'SPDN' in LEVERAGED_SYMBOLS, 'SDS' in LEVERAGED_SYMBOLS)
    # print(_check_symbol_relevant('SPDN'))
    # print(_check_symbol_relevant('SDS'))
    # return

    # spydf = load_history('^GSPC', resolution='week', start_date=None)
    # spydf = load_history('SPY', resolution='week', start_date='2011-01-04')
    # return


    # TODO figure out not just corr, but stats about distro of
    # movements together for options arbitrage



    # df = find_correlations_with('^GSPC')
    # df = find_correlations_with('SPY')
    # df = find_correlations_with('^NDX')
    df = find_correlations_with('QQQ')
    # print(sym2corr.head())
    df['abscorr'] = np.abs(df['Corr'])
    df = df.sort_values('abscorr', ascending=False, axis=0)
    df = df.loc[df['abscorr'] > .98]
    df['gapmins'] *= 100  # pct easier to read
    df['gapmaxs'] *= 100  # pct easier to read
    df['gapmeans'] *= 100  # pct easier to read
    df['gapstds'] *= 100  # pct easier to read
    df['gappct1s'] *= 100  # pct easier to read
    df['gappct5s'] *= 100  # pct easier to read
    df['gappct95s'] *= 100  # pct easier to read
    df['gappct99s'] *= 100  # pct easier to read
    # df = df.loc[df['']
    print(df.shape)
    print(df)
    return

    # # rets = load_pdf('tqqq', resolution='day')
    # # sym = 'tqqq'
    # # calls, puts = yf.Ticker('spy').option_chain(date='2020-06-19')
    # calls, puts = options_for_symbol('spy', '2020-06-18')
    # # calls, puts = options_for_symbol('tqqq', '2020-06-18')
    # # calls, puts = options_for_symbol('tqqq', '2022-01-20')
    # print("calls cols: ", calls.columns)
    # # calls = calls[[strike]]
    # print(calls.shape)
    # return

    # # download_100y_old_histories()
    # # df = load_master_df()
    # # df = _load_monthly_history_for_symbol('ZVV', start_date=None)
    # _download_history_for_symbol('ZVV', start_date=None)
    # # print("df shape: ", df.shape)
    # return

    # print("number of relevant symbols: ", len(all_relevant_symbols()))

    # _download_history_for_symbol('^DJI')
    # return


    df = get_monthly_stats_df()
    # df = df.loc[df['cagrStable'] > 1.05]
    # df = df.loc[df['annual'] > 1.05]
    # df = df.loc[df['annual'] > 1.1]
    # df = df.loc[df['wcagr'] > 1.15]
    # df = df.loc[df['wcagr'] > 1.18]
    df = df.loc[~df['symbol'].isin(_blacklisted_symbols())]
    # df = df.loc[df['wcagr'] > 1.2]
    # df = df.loc[df['qqqCorr'] < .25]
    print("monthly stats: ")
    # df.sort_values(by='cagrStable', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='qqqCorr', axis=0, inplace=True, ascending=True)
    # df.sort_values(by='cagrStable', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='cagrEfficiency', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='cagrRss', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='cagrRssCorr', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='cagrDrawdown', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='cagrAll', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='wcagrAll', axis=0, inplace=True, ascending=False)
    df.sort_values(by='peg', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='maxLev', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='wcagr', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='annual', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='relMedian', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='relMean', axis=0, inplace=True, ascending=False)
    # df['stableSharpe'] *= 1e2
    # df['stableSortino'] *= 1e2
    # df = df['symbol annual cagrEfficiency upwardFracLog cagr15y cagrStable stableSharpe qqqCorr'.split()]
    # df = df['symbol annual cagrDrawdown maxDrawdown cagrRssCorr cagrRss unexplainedFrac cagr5y cagrStable qqqCorr'.split()]
    # df = df['symbol annual cagrDrawdwn maxDrawdown unexplainedFrac cagrRssCorr cagrRss qqqCorr'.split()]
    # df = df['symbol annual cagrAll maxDrawdown unexplainedFrac qqqCorr'.split()]
    # df = df['symbol wcagr annual wcagrAll maxLev maxDrawdown unexplainedFrac qqqCorr'.split()]
    # df = df['symbol wcagr cagrRss wcagrAll maxLev maxDrawdown unexplainedFrac qqqCorr'.split()]
    df = df['symbol annual peg wcagrAll cagrDrawdown maxLev maxDrawdown unexplainedFrac qqqCorr'.split()]
    # df = df['symbol annual wcagr maxDrawdown unexplainedFrac qqqCorr'.split()]
    # df = df['symbol annual relMean cagrAll cagrCorrDown maxDrawdown unexplainedFrac qqqCorr'.split()]
    # df = df['symbol annual cagrDrawdown maxDrawdown unexplainedFrac qqqCorr'.split()]
    print(df.head(25))
    # print(df.loc[df['symbol'] == 'DPZ'])
    # print(df.loc[df['symbol'] == 'DG'])
    return



if __name__ == '__main__':
    main()
