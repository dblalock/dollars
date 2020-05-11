
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

START_DATE = '2004-11-19'  # NOTE: has to be a trading day

HISTORIES_DIR = 'histories'


if not os.path.exists(HISTORIES_DIR):
    os.makedirs(HISTORIES_DIR)


@_memory.cache
def _get_tickers_df():
    df0 = pd.read_csv('nasdaqlisted.txt', sep='|')
    df1 = pd.read_csv('otherlisted.txt', sep='|')
    df = pd.concat([df0, df1], axis=0, ignore_index=True, sort=False)
    df = df[['Symbol', 'Security Name']]
    df.drop_duplicates(subset=['Symbol'], keep='last', inplace=True)

    return df


def all_symbols():
    return sorted(_get_tickers_df()['Symbol'])


def _parse_iso_date(datestr):
    return datetime.datetime.strptime(datestr, '%Y-%m-%d')


@_memory.cache
def all_relevant_symbols(sizeCutoffBytes=100e3, dateCutoff='2004-11-20',
                         minAnnualRet=1.01, minAnnualRetOverDayStd=2.):
    ret = []
    cutoff_date = _parse_iso_date(dateCutoff)
    for sym in all_symbols():
        print("checking symbol: ", sym)
        path = _history_path_for_symbol(sym)

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
        total_return = df['Open'][end_idx] / df['Open'][0]
        annualized_ret = total_return ** (1. / timediff_years)
        if annualized_ret < minAnnualRet:
            continue

        # way too volatile
        reldiffs = (df['Close'] - df['Open']) / df['Open']
        if minAnnualRetOverDayStd / reldiffs.std() < minAnnualRetOverDayStd:
            continue

        ret.append(sym)  # found a decent one!

    return ret


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
    cumprod = 1
    for i in range(1, len(seq)):
        multipliers[i] = seq[i] / (cumprod + 1e-20)
        cumprod *= multipliers[i]
    return multipliers


@numba.njit(fastmath=True, cache=True)  # njit = no python, cache binary
def _compute_compound_returns(prices, dividends):
    returns = np.zeros(len(prices))
    initial_price = max(prices[0], .01)
    nshares = 1
    for i in range(len(prices)):
        returns[i] = nshares * prices[i] / initial_price
        nshares += nshares * dividends[i] / max(prices[i], .01)  # 0 -> 1c

    return returns


@_memory.cache
def _get_returns_stds_df_for_symbol(sym, start_date):
    df = _load_history_for_symbol(sym, start_date=start_date)
    ret_total = df['returns'].values[-1]
    ndays = df.shape[0]

    # print("df.shape: ", df.shape)
    # print(df['Date'].values[0], df['Date'].values[-1])
    # print(df['Close'].values[0], df['Close'].values[-1])

    start_datetime = _parse_iso_date(start_date)
    end_datetime = _parse_iso_date(df['Date'].iloc[df.shape[0] - 1])
    timediff = end_datetime - start_datetime
    nyears = timediff.days / 365.25

    ret_24h = ret_total ** (1. / ndays)
    ret_annual = ret_total ** (1. / nyears)
    return dict(
        symbol=sym,
        retTot=ret_total,
        ret24h=ret_24h,
        retAnnual=ret_annual,
        stdDaily=df['relDay'].std(),
        std24h=df['rel24h'].std())


# @_memory.cache
def get_returns_stds_df(start_date=START_DATE):
    symbols = all_relevant_symbols()
    dicts = []
    for sym in symbols:
        dicts.append(_get_returns_stds_df_for_symbol(sym, start_date))
    return pd.DataFrame.from_records(dicts)


def _load_history_for_symbol(symbol, start_date=None):
    df = pd.read_csv(_history_path_for_symbol(symbol))
    df.fillna(inplace=True, axis=0, method='ffill')  # forward fill nans

    print("symbol: ", symbol)

    if start_date is not None:
        # dates = [_parse_iso_date(datestr) for datestr in df['Date']]
        idx = np.where(df['Date'].values == start_date)[0][0]
        # print("start_date: ", start_date)
        # print("idx: ", idx)
        df = df.iloc[idx:]

    # handle missing values
    opens = df['Open'].values
    closes = df['Close'].values
    if opens.max() == 0:
        opens = closes
    if closes.max() == 0:
        closes = opens
    opens = np.maximum(.01, opens)
    closes = np.maximum(.01, closes)

    dividends = df['Dividends'].values

    # day_changes = opens[1:] - opens[:-1]
    # intraday_changes = closes - opens

    # print("df cols", df.dtypes)
    # print("df shape: ", df.shape)

    # NOTE: yahoo finance data already accounts for splits, so we don't
    # have to do anything with those






    # TODO prices are adjusted for splits, but dividends aren't; need to
    # account for this cuz right now numbers are bonkers for certain stocks








    # df = df['Date Dividends'.split()]
    # print("min close: ", closes.min())
    # print("max close: ", closes.max())
    assert dividends.min() == 0
    # print("closes num nans: ", np.isnan(closes).sum())
    # print("dividends num nans: ", np.isnan(dividends).sum())
    df['rel24h'] = _compute_relative_changes(closes)
    df['relDay'] = closes / opens
    df['relDiv'] = dividends / closes
    df['returns'] = _compute_compound_returns(closes, dividends)

    return df[['Date', 'Close', 'Dividends',
               'rel24h', 'relDay', 'relDiv', 'returns']]
    # df.drop(['Open High Low Close Volume'.split() + ['Stock Splits']],
    #         axis=1, inplace=True)
    # return df


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

    # symbols = all_relevant_symbols()

    # print(get_returns_stds_df())
    df = get_returns_stds_df()

    df['ratios'] = df['retAnnual'] / df['std24h']
    df = df['symbol ratios retAnnual std24h'.split()]
    df.sort_values(by='ratios', axis=0, inplace=True, ascending=False)
    print(df.head(50))


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
