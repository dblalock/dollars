
import datetime
import numpy as np
import os
import pandas as pd
import time
import yfinance as yf
from pprint import pprint
import numba

# from dateutil.relativedelta import relativedelta

from joblib import Memory

_memory = Memory('.')

# START_DATE = '2004-11-19'  # NOTE: has to be a trading day
START_DATE = '2002-09-03'  # NOTE: has to be a trading day

HISTORIES_DIR = 'histories'

# RISK_FREE_RATE_ANNUAL = 1.012
# RISK_FREE_RATE_MONTHLY = RISK_FREE_RATE_ANNUAL ** (1./12)
RISK_FREE_RATE_MONTHLY = 1.001
RISK_FREE_RATE_ANNUAL = RISK_FREE_RATE_MONTHLY ** 12


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
    return ('closed end', 'strategic', 'tactical', 'hedged')


@_memory.cache
def all_relevant_symbols(sizeCutoffBytes=100e3, dateCutoff='2004-11-20',
                         minAnnualRet=1.01, minAnnualRetOverDayStd=2.):
    ret = []
    cutoff_date = _parse_iso_date(dateCutoff)
    blacklisted_phrases = _blacklisted_phrases()

    df = all_symbols_and_names()
    symbols, names = df['Symbol'], df['Name']
    for sym, name in zip(symbols, names):
        print("checking symbol: ", sym)
        path = _history_path_for_symbol(sym)

        name = name.lower()
        if any([phrase in name for phrase in blacklisted_phrases]):
            continue  # exclude closed-end funds, tactical funds

        if os.path.getsize(path) < sizeCutoffBytes:
            continue  # too small

        df = pd.read_csv(path, nrows=2)
        if df.shape[0] < 2:
            print(f"WARNING: got shape {df.shape} for symbol {sym}")
            continue  # empty df

        start_date = _parse_iso_date(df['Date'].iloc[0])
        if start_date > cutoff_date:
            continue  # too recent

        df = pd.read_csv(path)
        end_idx = df.shape[0] - 1  # no idea why it can't deal with -1
        end_date = _parse_iso_date(df['Date'].iloc[end_idx])
        timediff = end_date - start_date
        # timediff_years = timediff.years + (timediff.months / 12.)
        timediff_years = timediff.days / 365.25
        total_return = df['Open'][end_idx] / max(.00001, df['Open'][0])
        annualized_ret = total_return ** (1. / timediff_years)
        if annualized_ret < minAnnualRet:
            continue

        # way too volatile
        reldiffs = (df['Close'] - df['Open']) / df['Open']
        if minAnnualRetOverDayStd / reldiffs.std() < minAnnualRetOverDayStd:
            continue

        # inconsistent volume; not even traded every day
        if (df['Volume'] > 0).mean() < .98:
            continue

        # reverse splits are sorta a red flag, but mostly stuff like KF and
        # CLM has discrepancies between downloaded data

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
    initial_price = max(prices[0], .0001)
    nshares = 1
    for i in range(len(prices)):
        returns[i] = nshares * prices[i] / initial_price
        nshares += nshares * dividends[i] / max(prices[i], .00001)  # 0 -> 1c

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
    bonus_shares = dividends[N - 1] / max(prices[N - 1], .00001)
    for i in range(N - 2, -1, -1):
        multiplier = returns[i] / returns[i + 1]
        prices[i] = multiplier * prices[i + 1]
        prices[i] *= (1 + bonus_shares)

        # compute bonus shares for next iter; yf multiplies *previous* returns
        # by a value less than 1, so this undoes that (in theory)
        bonus_shares = dividends[i] / max(prices[i], .00001)

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


# def _maxdrawdown(mins, maxs=None):
#     if maxs is None:
#         maxs = mins
#     return _maxdrawdown_jit(mins, maxs)

@_memory.cache
def _get_returns_daily_stds_df_for_symbol(sym, start_date):
    df = _load_history_for_symbol(sym, start_date=start_date)
    priceret_total = df['price'].values[-1] / max(.00001, df['price'].values[0])
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


def _monthly_corr_with_spy(df, start_date, col='Close'):
    df0 = _load_monthly_history_for_symbol('SPY', start_date=start_date)
    return np.corrcoef(df[col], df0[col])[0, 1]  # returns 2x2 matrix


def _get_monthly_stats_df_for_symbol(sym, start_date=START_DATE):
    df = _load_monthly_history_for_symbol(sym, start_date=start_date)
    startprice = max(.00001, df['price'].values[0])
    priceret_total = df['price'].values[-1] / startprice
    ret_total = df['Close'].values[-1] / df['Close'].values[0]
    ndays = df.shape[0]

    # start_datetime = df[]
    # end_datetime = _parse_iso_date(df['Date'].iloc[df.shape[0] - 1])

    timediff = df.index[-1] - df.index[0]
    # print(diff.days)
    nyears = timediff.days / 365.25
    nmonths = nyears * 12

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
    closes = df['Close']
    returns = df['Close'].values
    ret['cagr5y'] = (returns[-1] / returns[-61]) ** (1./60)
    ret['cagr10y'] = (returns[-1] / returns[-121]) ** (1./120)
    ret['cagr15y'] = (returns[-1] / returns[-181]) ** (1./180)
    ret['cagr2008'] = (closes['2007-9-30'] / closes['2009-9-30']) ** (1./24)
    ret['cagr2020'] = (closes['2020-1-31'] / closes['2020-4-30']) ** (1./3)

    # "risk" metrics
    neg_returns_mask = df['relMonth'] < 1
    sharpe_denom = returns.std()
    sortino_denom = returns[neg_returns_mask].std()
    # ret['maxDrawdown'] = _maxdrawdown(returns)
    # drawdowns, cummins = _maxdrawdown(
    ret['maxDrawdown'] = _maxdrawdown(
        mins=df['minClose'].values, maxs=df['maxClose'].values)
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
        spyCorr=_monthly_corr_with_spy(df, start_date=start_date)))

    return ret





# @_memory.cache
def get_returns_daily_stds_df(start_date=START_DATE):
    symbols = all_relevant_symbols()
    dicts = []
    # for sym in ['ABEV']:
    # for sym in symbols[:100]:
    for sym in symbols:
        dicts.append(_get_returns_daily_stds_df_for_symbol(sym, start_date))
    return pd.DataFrame.from_records(dicts)



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
    opens = np.maximum(.00001, opens)
    closes = np.maximum(.00001, closes)

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

    # priceret_total = df['Close'].values[-1] / max(.00001, df['Close'].values[0])
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
    df.drop(['Date', 'rel24h', 'relDay', 'Dividends'], axis=1, inplace=True)
    df['maxClose'] = dailydf['Close'].resample('M').max()
    df['minClose'] = dailydf['Close'].resample('M').min()
    df['relMonth'] = _compute_relative_changes(df['Close'].values)
    return df


def _load_master_df(maxStartDate='2004-11-20'):  # right after GLD
    symbols = all_relevant_symbols()
    sym = symbols[0]
    df = _load_history_for_symbol(sym)
    for symbol in all_symbols()[1:2]:
        pass # TODO


def _leverage_for_symbol(sym):
    df = pd.read_csv('leverage-symbol-mappings.csv')
    return dict(zip([df['Symbol'], df['Leverage']]))


def main():

    # # df = _load_monthly_history_for_symbol('tqqq', start_date=None)
    # # df = _load_monthly_history_for_symbol('msft', start_date=None)
    # df = _load_monthly_history_for_symbol('aapl', start_date=None)
    # # print(df.head())
    # # print(df.tail())
    # # print(df['Close']['2002-9-30'])
    # for col in 'Close minClose maxClose'.split():
    #     print('{:8}: start-end = {}-{}'.format(
    #         col, df[col]['2007-12-31'], df[col]['2008-12-31']))
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

    # d = _get_monthly_stats_df_for_symbol('msft')
    d = _get_monthly_stats_df_for_symbol('aapl')
    import pprint
    pprint.pprint(d)

    # symbols = all_relevant_symbols()
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
