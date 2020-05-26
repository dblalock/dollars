
import datetime
import numpy as np
import os
import pandas as pd
import time
import yfinance as yf
from pprint import pprint
import numba
from sklearn import linear_model as linear

import matplotlib.pyplot as plt

# from dateutil.relativedelta import relativedelta

from joblib import Memory

_memory = Memory('.')

START_DATE = '2004-11-19'  # NOTE: has to be a trading day; 2004 to get GLD
# START_DATE = '2002-09-03'  # NOTE: has to be a trading day

HISTORIES_DIR = 'histories'

# RISK_FREE_RATE_ANNUAL = 1.012
# RISK_FREE_RATE_MONTHLY = RISK_FREE_RATE_ANNUAL ** (1./12)
RISK_FREE_RATE_MONTHLY = 1.001
RISK_FREE_RATE_ANNUAL = RISK_FREE_RATE_MONTHLY ** 12
TINY_VAL = 1e-10

# def _leveraged_etfs_df():
LEVERAGED_ETFS_DF = pd.read_csv('leverage-symbol-mappings.csv')
LEVERAGED_ETFS_DF = LEVERAGED_ETFS_DF.loc[LEVERAGED_ETFS_DF['isMonthly'] == 0]
LEVERAGED_ETFS_DF.drop('isMonthly', axis=1, inplace=True)


if not os.path.exists(HISTORIES_DIR):
    os.makedirs(HISTORIES_DIR)


@_memory.cache
def _get_tickers_df():
    df0 = pd.read_csv('nasdaqlisted.txt', sep='|')
    df1 = pd.read_csv('otherlisted.txt', sep='|')
    df = pd.concat([df0, df1], axis=0, ignore_index=True, sort=False)
    df = df[['Symbol', 'Security Name']]
    df.drop_duplicates(subset=['Symbol'], keep='last', inplace=True)
    df.sort_values(by='Symbol', axis=0, inplace=True)
    df.rename(columns={'Security Name': 'Name'}, inplace=True)

    return df


def all_symbols():
    # return sorted(_get_tickers_df()['Symbol'])
    return _get_tickers_df()['Symbol']


def all_symbols_and_names():
    # return sorted(_get_tickers_df()['Symbol'])
    return _get_tickers_df()


def _parse_iso_date(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d')


def _blacklisted_phrases():
    return ('tactical', 'hedged')
    # return ('closed end', 'strategic', 'tactical', 'hedged')


def _check_symbol_relevant(sym, name, sizeCutoffBytes=100e3,
                           dateCutoff=START_DATE, minAnnualRet=1.01,
                           minAnnualRetOverDayStd=1.5):
    cutoff_date = _parse_iso_date(dateCutoff)
    blacklisted_phrases = _blacklisted_phrases()

    path = _history_path_for_symbol(sym)

    name = name.lower()
    if any([phrase in name for phrase in blacklisted_phrases]):
        return False  # exclude closed-end funds, tactical funds

    if os.path.getsize(path) < sizeCutoffBytes:
        return False  # too small

    df = pd.read_csv(path, nrows=2)
    if df.shape[0] < 2:
        print(f"WARNING: got shape {df.shape} for symbol {sym}")
        return False  # empty df

    start_date = _parse_iso_date(df['Date'].iloc[0])
    if start_date > cutoff_date:
        return False  # too recent

    # check whether returns are consitently above cutoff at various
    # timescales
    df = pd.read_csv(path)
    end_idx = df.shape[0] - 1  # no idea why it can't deal with -1
    end_date = _parse_iso_date(df['Date'].iloc[end_idx])
    closes = df['Close'].values
    for initial_idx in [0, -750, -1500, -2250]:  # 250 trading days/yr
        initial_val = max(TINY_VAL, closes[initial_idx])
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

    # inconsistent volume; not even traded every day
    if (df['Volume'] > 0).mean() < .98:
        return False

    return True


@_memory.cache
def all_relevant_symbols(**kwargs):
    ret = []

    df = all_symbols_and_names()
    symbols, names = df['Symbol'], df['Name']
    for sym, name in zip(symbols, names):
        print("checking symbol: ", sym)
        if _check_symbol_relevant(sym, name, **kwargs):
            ret.append(sym)  # found a decent one!
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
    return os.path.join(HISTORIES_DIR, symbol + '.csv')


def _download_history_for_symbol(symbol, startat=None, **kwargs):
    if startat is not None and symbol < startat:
        return
    kwargs.setdefault('period', 'max')  # download all data by default
    print(f"downloading history for {symbol}...")
    df = yf.Ticker(symbol).history(**kwargs)
    df.to_csv(_history_path_for_symbol(symbol))
    time.sleep(max(.5, 1 + np.random.randn()))


def download_histories(startat=None):
    for symbol in all_symbols():
        _download_history_for_symbol(symbol, startat=startat)


def download_100y_old_histories(startat=None):
    """these fail if you just ask for max"""
    df = pd.read_csv('tickers-over-100-yrs-old.txt', names=['Symbol'])
    symbols = df['Symbol']

    # print(symbols)
    # for sym in symbols[:2]:  # TODO rm after debug
    for sym in symbols:  # TODO rm after debug
        _download_history_for_symbol(
            sym, startat=startat, period=None, start='1921-1-1')

    # symbols that still fail:
        # BF.B, BIO.B, GEF.B, LEN.B, RDS.B, STZ.B, TDW.B,
        # they seem to also not work on the yahoo finance website


# @numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
@numba.njit(fastmath=True)  # njit = no python, cache binary
def _compute_relative_changes(seq):
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
def _compute_compound_returns(prices, dividends):
    returns = np.zeros(len(prices))
    initial_price = max(prices[0], TINY_VAL)
    nshares = 1
    for i in range(len(prices)):
        returns[i] = nshares * prices[i] / initial_price
        nshares += nshares * dividends[i] / max(prices[i], TINY_VAL)  # 0 -> 1c

    return returns


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def _compute_prices(returns, dividends):
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
# def _maxdrawdown_jit(mins, maxs):
def _maxdrawdown(mins, maxs=None):
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
    base_endidx = np.where(base_history_df.index == startdate)[0][0]
    # print("base_endidx: ", base_endidx)
    imputed_head = base_history_df.iloc[:base_endidx].copy()

    # pull out daily relative price changes; tricky endpoints here; first
    # relchange is always 1, and we actually want the relchange associated
    # with the first day in the leveraged one, not the day before it
    base_prices = base_history_df['price'].iloc[:base_endidx + 1].values
    base_relchanges = _compute_relative_changes(base_prices)[1:]

    # print("daily_history df cols", lev_history_df.columns.values)
    # print("tracked_history df cols", base_history_df.columns.values)

    # tracked_prices_head = imputed_head['price'].values
    initialval = lev_history_df['Close'].values[0]
    # relchanges = _compute_relative_changes(initialval)
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
    imputed_head['rel24h'] = _compute_relative_changes(new_closes)
    imputed_head['price'] = imputed_head['Close'] * (  # since no dividends
        lev_history_df['price'].values[0] / lev_history_df['Close'].values[0])

    ret = pd.concat([imputed_head, lev_history_df],
                    axis=0, ignore_index=False, sort=False)

    return ret


def _load_history_for_leveraged_symbol(symbol, start_date=START_DATE):
    # print("LEVERAGED_ETFS_DF", LEVERAGED_ETFS_DF)
    idx = np.where(
        LEVERAGED_ETFS_DF['LeveragedSymbol'].str.lower() == symbol)[0][0]
    print("idx: ", idx)
    print("LEVERAGED_ETFS_DF['Symbol'].values[idx]", LEVERAGED_ETFS_DF['Symbol'].values[idx])
    print("LEVERAGED_ETFS_DF['Leverage'].values[idx]", LEVERAGED_ETFS_DF['Leverage'].values[idx])
    tracks_symbol = LEVERAGED_ETFS_DF['Symbol'].values[idx]
    leverage = float(LEVERAGED_ETFS_DF['Leverage'].values[idx])
    tracks_history = _load_history_for_symbol(
        tracks_symbol, start_date=None)
    leveraged_history = _load_history_for_symbol(
        symbol, start_date=None)
    return _impute_history_for_leveraged_etf(
        leveraged_history, tracks_history, leverage)


def _load_history_for_symbol(symbol, start_date=START_DATE):
    df = pd.read_csv(_history_path_for_symbol(symbol))
    df.fillna(inplace=True, axis=0, method='ffill')  # forward fill nans

    print("symbol: ", symbol)

    if start_date is not None:
        # dates = [_parse_iso_date(datestr) for datestr in df['Date']]
        idx = np.where(df['Date'].values == start_date)[0][0]
        print("start_date: ", start_date)
        print("idx: ", idx)
        df = df.iloc[idx:]
    # df['Date'] = df['Date'].apply(_parse_iso_date)
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

    # cutoff_date =

    # handle missing values
    opens = df['Open'].values
    closes = df['Close'].values
    if opens.max() == 0:
        opens = closes
    if closes.max() == 0:
        closes = opens
    opens = np.maximum(TINY_VAL, opens)
    closes = np.maximum(TINY_VAL, closes)

    # day_changes = opens[1:] - opens[:-1]
    # intraday_changes = closes - opens

    # print("df cols", df.dtypes)
    # print("df shape: ", df.shape)

    # dividends = df['Dividends'].values.astype(np.float64)

    # EDIT: looks like the price info is "adjusted price" which takes into
    # account dividends already
    #   -so what I actually need to do is construct a price col, not a
    #   total return one
    #   -https://ca.help.yahoo.com/kb/finance/adjusted-close-sln28256.html

    # splits = df['Stock Splits'].values
    # idxs = np.where(splits != 0)[0]
    # print("splits: ", splits[idxs])
    # print("split dates: ", df['Date'].values[idxs])
    # # assert(dividends.shape == splits.shape)
    # splits[splits == 0] = 1.  # 0 = no change, so really 1

    # # this is probably wrong, but the combo of reverse splits and dividends
    # # yields insane total returns
    # if splits.min() < 1:
    #     splits[:idxs[-1]] = 1
    # # splits[splits < 1] = 1

    # split_coeffs = np.cumprod(splits[::-1])[::-1]
    # assert split_coeffs.min() > 0
    # diffs = split_coeffs[1:] - split_coeffs[-1:]
    # assert diffs.min() <= 0
    # old_dividends = dividends.copy()
    # dividends /= split_coeffs
    # # dividends *= split_coeffs
    # if splits.min() >= 1:
    #     assert np.all(dividends <= old_dividends)
    # # print("adjusted and orig dividends:")
    # print(dividends[dividends > 0][:20])
    # print(old_dividends[dividends > 0][:20])

    # df = df['Date Dividends'.split()]
    # print("min close: ", closes.min())
    # print("max close: ", closes.max())

    # dividends = df['Dividends'].values
    # assert dividends.min() == 0
    # print("closes num nans: ", np.isnan(closes).sum())
    # print("dividends num nans: ", np.isnan(dividends).sum())
    df['rel24h'] = _compute_relative_changes(closes)
    df['relDay'] = closes / opens
    df['price'] = _compute_prices(closes, df['Dividends'].values)
    # df['relDiv'] = dividends / closes
    # df['returns'] = _compute_compound_returns(closes, dividends)

    # priceret_total = df['Close'].values[-1] / max(TINY_VAL, df['Close'].values[0])
    # print("final and initial prices: ", df['Close'].values[-1], df['Close'].values[0])
    # print("final and initial dates: ", df['Date'].values[-1], df['Date'].values[0])
    # ret_total = df['returns'].values[-1]

    return df[['Date', 'Close', 'Dividends',
               'rel24h', 'relDay', 'price']]
    # df.drop(['Open High Low Close Volume'.split() + ['Stock Splits']],
    #         axis=1, inplace=True)
    # return df


@_memory.cache
def _load_monthly_history_for_symbol(symbol, start_date=START_DATE):
    dailydf = _load_history_for_symbol(symbol, start_date)
    df = dailydf.asfreq('M', method='ffill')
    df['maxClose'] = dailydf['Close'].resample('M').max()
    df['minClose'] = dailydf['Close'].resample('M').min()
    df['relMonth'] = _compute_relative_changes(df['Close'].values)
    df.drop(['Date', 'rel24h', 'relDay', 'Dividends'], axis=1, inplace=True)
    return df


# def _maxdrawdown(mins, maxs=None):
#     if maxs is None:
#         maxs = mins
#     return _maxdrawdown_jit(mins, maxs)

@_memory.cache
def _get_returns_daily_stds_df_for_symbol(sym, start_date):
    df = _load_history_for_symbol(sym, start_date=start_date)
    priceret_total = df['price'].values[-1] / max(TINY_VAL, df['price'].values[0])
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


def _monthly_corr_with_qqq(df, start_date=START_DATE, col='relMonth'):
    df0 = _load_monthly_history_for_symbol('QQQ', start_date=start_date)
    x, y = df[col].values, df0[col].values
    x -= x.mean()
    y -= y.mean()
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)

    # print("x, y")
    # print(x[:12])
    # print(y[:12])

    return (x * y).sum()
    # return np.corrcoef(df[col], df0[col])[0, 1]  # returns 2x2 matrix


# def _lstsq_stats(returns, weights='sqrt'):
def _unexplained_variance_loglinear(returns, weights='sqrt'):
    N = len(returns)
    if weights == 'sqrt':
        weights = np.sqrt(np.arange(N))

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
    # return sse, rss

    # return_total = returns[-1] / returns[0]
    # return return_total / fraction_unexplained


    # yhat_norm =

    # w, rss, _, _ = np.linalg.listsq(x, y)
    # w, rss, _, _ = np.linalg.listsq(x, y)

    # yhat = x.reshape(-1, 1) @ w


def _get_monthly_stats_for_symbol(sym, start_date=START_DATE):
    df = _load_monthly_history_for_symbol(sym, start_date=start_date)
    # nmonths = df.shape[0]
    # nyears = nmonths / 12
    # ndays = nyears * 365.25
    # print("ndays, nyears = ", ndays, ndays / 365.25)
    startprice = max(TINY_VAL, df['price'].values[0])
    priceret_total = df['price'].values[-1] / startprice

    closes = df['Close']
    returns = df['Close'].values
    relret = df['relMonth'].values
    ret_total = returns[-1] / returns[0]

    # start_datetime = df[]
    # end_datetime = _parse_iso_date(df['Date'].iloc[df.shape[0] - 1])

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
    ret['mean_monthly'] = df['relMonth'].mean()  # arithmetic, not geometric

    def _key_for_percentile(p):
        return 'monthlyRetPctile={:02d}'.format(p)

    # add in quantiles of returns across months
    percentiles = [1, 5, 10, 50, 90, 95, 99]
    vals = np.percentile(df['relMonth'].values, percentiles)
    for p, val in zip(percentiles, vals):
        ret[_key_for_percentile(p)] = val

    # compute returns in different time periods

    # print('dates: ')
    # print(df.index.values[-61])
    # print(df.index.values[-121])
    # print(df.index.values[-181])

    ret['cagr5y'] = (returns[-1] / returns[-61]) ** (1./5)
    ret['cagr10y'] = (returns[-1] / returns[-121]) ** (1./10)
    ret['cagr15y'] = (returns[-1] / returns[-181]) ** (1./15)
    ret['cagr2008'] = (closes['2009-9-30'] / closes['2007-9-30']) ** (1./2)
    ret['cagr2008fall'] = (closes['2008-10-31'] / closes['2008-8-31']) ** 12
    ret['cagr2020'] = (closes['2020-4-30'] / closes['2020-1-31']) ** 4

    # mu = df['relMonth'].mean()
    # diffs = df['relMonth'] - mu
    # ret['pathEfficiency'] = mu / np.abs(diffs).mean()
    # ret['upwardFrac'] = mu / np.abs(diffs).mean()
    # relret = df['relMonth'].values
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
    neg_returns_mask = df['relMonth'] < 1
    sharpe_denom = returns.std()
    sortino_denom = returns[neg_returns_mask].std()
    # ret['maxDrawdown'] = _maxdrawdown(returns)
    # drawdowns, cummins = _maxdrawdown(
    ret['maxDrawdown'] = _maxdrawdown(  # close instead of maxclose
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
    ret['cagrStable'] = np.min([ret['cagr5y'], ret['cagr10y'], ret['cagr15y']])
    ret['cagrCrash'] = min(ret['cagr2008'], ret['cagr2020'])
    ret['stableSharpe'] = (ret['cagrStable'] - RISK_FREE_RATE_MONTHLY) / sharpe_denom
    ret['stableSortino'] = (ret['cagrStable'] - RISK_FREE_RATE_MONTHLY) / sortino_denom
    ret['stableQuad'] = ret['cagrStable'] - .55 * df['relMonth'].var()
    ret['monthStd'] = df['relMonth'].std()

    # upward movement over total variation distance; on raw returns and log
    # returns; latter makes more sense / doesn't just focus on recent history
    diffs = returns[1:] - returns[:-1]
    ret['upwardFracRaw'] = (returns[-1] - returns[0]) / np.abs(diffs).sum()
    logret = np.log(relret)
    diffs = logret[1:] - logret[:-1]
    ret['upwardFracLog'] = logret.mean() / np.abs(diffs).mean()
    ret['cagrEfficiency'] = 1 + ret['upwardFracLog'] * (ret['annual'] - 1)
    ret['cagrStableEfficiency'] = 1 + ret['upwardFracLog'] * (ret['cagrStable'] - 1)

    frac = _unexplained_variance_loglinear(returns)
    ret['unexplainedFrac'] = frac
    ret['cagrRss'] = ret_total / frac

    # drawdown_coef = (1 - ret['maxDrawdown']) / ret['maxDrawdown']
    # drawdown_coef = (1. / ret['maxDrawdown']) - 1  # most extreme version
    drawdown_coef = 1. - ret['maxDrawdown']
    # drawdown_coef = 1. / ret['maxDrawdown']
    ret['cagrDrawdown'] = (ret['annual'] - 1) * drawdown_coef + 1

    corr = _monthly_corr_with_qqq(df, start_date=start_date)
    corrmultiplier = np.sqrt(2 / (1 + corr))
    ret['cagrCorr'] = (ret['annual'] - 1) * corrmultiplier + 1
    ret['cagrCorrDown'] = (ret['cagrDrawdown'] - 1) * corrmultiplier + 1
    ret['cagrRssCorr'] = (ret['cagrRss'] - 1) * corrmultiplier + 1
    # ret['cagrDrawdown'] = ret['annual'] * (1 - ret['maxDrawdown'])
    # ret['cagrAll'] = ret['cagrRssCorr'] * (1 - ret['maxDrawdown'])

    ret['cagrAll'] = (ret['cagrCorrDown'] - 1) / frac + 1

    # print("relmonth")
    # print(df['relMonth'].head())
    # print(df['relMonth'].tail())

    # ret['stable/Pct05'] = ret['stableRet'] * ret[_key_for_percentile(5)]
    # ret['stable/Pct10'] = ret['stableRet'] * ret[_key_for_percentile(10)]

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
             for sym in all_relevant_symbols()]
    return pd.DataFrame.from_records(dicts)


@_memory.cache
def get_returns_daily_stds_df(start_date=START_DATE):
    symbols = all_relevant_symbols()
    dicts = []
    # for sym in ['ABEV']:
    # for sym in symbols[:100]:
    for sym in symbols:
        dicts.append(_get_returns_daily_stds_df_for_symbol(sym, start_date))
    return pd.DataFrame.from_records(dicts)


def _load_master_df(start_date=START_DATE):
    symbols = all_relevant_symbols()
    sym = symbols[0]
    df = _load_history_for_symbol(sym)
    for symbol in all_symbols()[1:2]:
        pass # TODO


# def _leverage_for_symbol(sym):
#     df = pd.read_csv('leverage-symbol-mappings.csv')
#     return dict(zip([df['Symbol'], df['Leverage']]))


def main():

    # df = get_monthly_stats_df()
    # df = df.loc[df['cagrStable'] > 1.05]
    # # df = df.loc[df['annual'] > 1.05]
    # df = df.loc[df['annual'] > 1.1]
    # # df = df.loc[df['qqqCorr'] < .25]
    # print("monthly stats: ")
    # # df.sort_values(by='cagrStable', axis=0, inplace=True, ascending=False)
    # # df.sort_values(by='qqqCorr', axis=0, inplace=True, ascending=True)
    # # df.sort_values(by='cagrStable', axis=0, inplace=True, ascending=False)
    # # df.sort_values(by='cagrEfficiency', axis=0, inplace=True, ascending=False)
    # # df.sort_values(by='cagrRss', axis=0, inplace=True, ascending=False)
    # # df.sort_values(by='cagrRssCorr', axis=0, inplace=True, ascending=False)
    # # df.sort_values(by='cagrDrawdown', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='cagrAll', axis=0, inplace=True, ascending=False)
    # df['stableSharpe'] *= 1e2
    # df['stableSortino'] *= 1e2
    # # df = df['symbol annual cagrEfficiency upwardFracLog cagr15y cagrStable stableSharpe qqqCorr'.split()]
    # # df = df['symbol annual cagrDrawdown maxDrawdown cagrRssCorr cagrRss unexplainedFrac cagr5y cagrStable qqqCorr'.split()]
    # # df = df['symbol annual cagrDrawdown maxDrawdown unexplainedFrac cagrRssCorr cagrRss qqqCorr'.split()]
    # # df = df['symbol annual cagrAll maxDrawdown unexplainedFrac cagrRssCorr cagrRss qqqCorr'.split()]
    # df = df['symbol annual cagrAll cagrCorrDown maxDrawdown unexplainedFrac qqqCorr'.split()]
    # # df = df['symbol annual cagrDrawdown maxDrawdown unexplainedFrac qqqCorr'.split()]
    # print(df.head(50))
    # # print(df.head())
    # return

    # d = _get_monthly_stats_for_symbol('msft')
    # # d = _get_monthly_stats_for_symbol('aapl')
    # d = _get_monthly_stats_for_symbol('shy')

    # # d = _get_monthly_stats_for_symbol('tlt')
    # # d = _get_monthly_stats_for_symbol('qqq')
    # d = _get_monthly_stats_for_symbol('apt')
    # # d = _get_monthly_stats_for_symbol('gld')
    # # d = _get_monthly_stats_for_symbol('nflx')
    # import pprint
    # pprint.pprint(d)
    # return

    # _download_history_for_symbol('^GSPC')

    # df = _load_monthly_history_for_symbol('tlt')
    # df = _load_history_for_symbol('tlt')
    # df = _load_history_for_leveraged_symbol('tmf')
    # df = _load_history_for_leveraged_symbol('spxl')
    # df = _load_history_for_leveraged_symbol('need')
    df = _load_history_for_leveraged_symbol('cure')
    # df = _load_history_for_leveraged_symbol('tqqq')
    print("history df cols: ", df.columns.values)
    print("leverage df cols: ", LEVERAGED_ETFS_DF.columns.values)
    print(df['Close'].index[:10])
    df['Close'].plot()
    plt.gca().semilogy()
    plt.show()
    return

    # # df0 = _load_history_for_symbol('spy')
    # # print(_monthly_corr_with_qqq(df, start_date=START_DATE, col='relMonth'))
    # print(_monthly_corr_with_qqq(df, start_date=START_DATE))


    # # # df = _load_monthly_history_for_symbol('tqqq', start_date=None)
    # # # df = _load_monthly_history_for_symbol('msft', start_date=None)
    # # df = _load_monthly_history_for_symbol('aapl', start_date=None)
    # # print(df.head())
    # # print(df.tail())
    # # # print(df['Close']['2002-9-30'])
    # # for col in 'Close minClose maxClose'.split():
    # for col in ['Close']:
    #     print('{:8}: start-end = {}-{}'.format(
    #         col, df[col]['2007-12-31'], df[col]['2008-12-31']))
    #     print('{:8}: start-end = {}-{}'.format(
    #         col, df[col]['2020-01-31'], df[col]['2020-04-30']))
    # # print(df['Close']['2002-10-31'])
    # return
    # print(df.tail(20))
    # # diff = df.index[-1] - df.index[0]
    # # print(diff.days)
    # # ret_2008 = df['Close']['2020-4-30']
    # ret_2008 = df['Close']['2007-9-30']
    # print(ret_2008)
    # return
    # # df['Close'].plot()
    # # import matplotlib.pyplot as plt
    # # plt.gca().semilogy()
    # # plt.show()
    # # return

    # x = np.array([1, 2, 3, 4, 2, 5])
    # print('drawdown stats:')
    # # print(_maxdrawdown(np.array([1, 2, 3, 4, 2, 5])))
    # # print(_maxdrawdown(np.array([1, 2, 3, 4, 3, 5])))
    # print(_maxdrawdown(np.array([0, 2, 3, 4, 3, 2, 1, 5])))


    # symbols = all_relevant_symbols()
    # # print(_check_symbol_relevant('JAKK', 'foo'))
    # print("num relevant symbols:", len(symbols))
    # np.savetxt('relevant-symbols.txt', symbols, fmt='%s', delimiter='\n')
    # return

    # print(get_returns_daily_stds_df())
    # return

    # df = get_returns_daily_stds_df()
    # df['ratios'] = df['priceRetTot'] / df['std24h']
    # df = df['symbol ratios retAnnual priceRetTot std24h'.split()]
    # df.sort_values(by='ratios', axis=0, inplace=True, ascending=False)
    # print(df.head(50))
    # return

    # df = get_returns_daily_stds_df()
    # df = df['symbol retAnnual retTot priceRetTot std24h'.split()]
    # df.sort_values(by='retAnnual', axis=0, inplace=True, ascending=False)
    # print(df.head(50))

    # df = get_returns_daily_stds_df()
    # # df['ratios'] = df['priceRetTot'] / df['std24h']
    # df['ratios'] = df['retAnnual'] / df['std24h']
    # df = df['symbol retAnnual ratios std24h'.split()]
    # # df.sort_values(by='retAnnual', axis=0, inplace=True, ascending=False)
    # df.sort_values(by='ratios', axis=0, inplace=True, ascending=False)
    # print(df.head(50))
    # # print(df.iloc[50:100])

    # _download_history_for_symbol('CLM')
    # _load_history_for_symbol('CLM')
    # df = _load_history_for_symbol('HYB')
    # print(_get_returns_daily_stds_df_for_symbol('HYB', START_DATE))
    # print(df['Stock Splits'])
    # return

    # _download_history_for_symbol('A')
    # _download_history_for_symbol('AA')
    # _download_history_for_symbol('AAAU')
    # _download_history_for_symbol('AADR')

    # for symbol in all_symbols():
    #     if not os.path.exists(_history_path_for_symbol(symbol)):
    #         print("trying to download history for missed symbol: ", symbol)
    #         _download_history_for_symbol(symbol)

    # print(all_relevant_symbols())

    # df0 = _load_history_for_symbol('msft')[-10:]
    # df1 = _load_history_for_symbol('aapl')[-10:]

    # # df = pd.

    # print(df0)
    # print(df1)

    # pass

    # download_histories()
    # download_histories(startat='ATRI')
    # download_100y_old_histories()

    # df = _get_tickers_df()

    # marketCaps = []
    # for ticker in df['Symbol'][:20]:
    #     print("ticker: ", ticker)
    #     try:
    #         info = yf.Ticker(ticker).info
    #         marketCaps.append(info['marketCap'])
    #     except IndexError:
    #         marketCaps.append(-1)

    # print("marketCaps: ", marketCaps)
    # # df['marketCap'] = marketCaps

    # print(df.shape)
    # print(df.head())
    # print(df.dtypes)

    # msft = yf.Ticker('msft')
    # msft = yf.Ticker('tlt')
    # # msft = yf.Ticker('aapl')
    # # # # no info about how long data goes back; spotty info in general; eg
    # # # # last dividend value is None, even though it gives dividends
    # # # pprint(msft.info)

    # # df = msft.history(period='30m')
    # # print(df)

    # print(msft.actions)  # dividends, splits
    # # # print(msft.splits)
    # # # print(msft.dividends)

    # # # doesn't cache the download;
    # # # see https://github.com/ranaroussi/yfinance/blob/476cf81beb55efec78eb0719ce1a42e9fbd9421a/yfinance/base.py#L150
    # df = msft.history(period='max')
    # print(df.shape)
    # print(df[:5])
    # print(df[-5:])


if __name__ == '__main__':
    main()
