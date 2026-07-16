import numpy as np
import yfinance as yf
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
from math import ceil, log, sqrt
from pathlib import Path
import requests
from config import YAHOO_MAP
from datetime import datetime

def write_tickers(
        NYSE_FILE: str,
        NASDAQ_FILE: str
):

    nyse = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        sep="|"
    )

    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        sep="|"
    )

    nyse.to_csv(NYSE_FILE, index=False)
    nasdaq.to_csv(NASDAQ_FILE, index=False)

def pull_tickers_sample(
    sample_size: int,
    NYSE_FILE: str,
    NASDAQ_FILE: str
) -> list[str]:
    nyse = pd.read_csv(NYSE_FILE)
    nasdaq = pd.read_csv(NASDAQ_FILE)

    ALL_TICKERS = (
        pd.concat([
            nasdaq["Symbol"],
            nyse["ACT Symbol"]
        ])
        .dropna()
        .astype(str)
        .unique()
    )
    return list(
        np.random.choice(
            ALL_TICKERS,
            size=min(sample_size, len(ALL_TICKERS)),
            replace=False
        )
    )

def garman_klass_vol(
    df: pd.DataFrame,
    window: int = 30
) -> pd.DataFrame:
    
    df_copy = df.copy()
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    
    rs = 0.5 * log_hl**2 - (2 * log(2) - 1) * log_co**2
    df_copy['Volatility'] = (rs.rolling(window=window).mean() * 252) ** 0.5

    return df_copy

def real_vol(
    df: pd.DataFrame,
    window: int = 30
) -> pd.DataFrame:
    
    df_copy = df.copy()

    log_returns = np.log(
        df['Close'] / df['Close'].shift(1)
    )
    df_copy['Volatility'] = (log_returns.rolling(window=window).std()*sqrt(252))

    return df_copy

def hist_implied_vol(
    ticker: str,
    api_key: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:

    rows = []
    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="B"
    )

    for date in dates:
        url = (
            f"https://api.polygon.io/v3/snapshot/options/{ticker}"
        )
        params = {
            "apiKey": api_key,
            "as_of": date.strftime("%Y-%m-%d")
        }
        response = requests.get(
            url,
            params=params
        )

        data = response.json()

        if "results" not in data:
            continue

        for contract in data["results"]:
            details = contract.get("details", {})
            greeks = contract.get("greeks", {})
            rows.append({
                "date":date,
                "contract":details.get("ticker"),
                "type":details.get("contract_type"),
                "expiry":details.get("expiration_date"),
                "strike":details.get("strike_price"),
                "implied_volatility":contract.get("implied_volatility"),
                "delta":greeks.get("delta"),
                "gamma":greeks.get("gamma"),
                "theta":greeks.get("theta"),
                "vega":greeks.get("vega")
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.set_index("date")

    return df

def calc_profit(
        strategy: tuple,
        prices_df: pd.DataFrame,
        ticker: str,
        period: str = '1y',
        initial_investment: int = 1000
) -> Dict:
    
    if prices_df.empty:
        return None
    
    profit, next_val, required_capital, pct_up, pct_down = 0, 0, 0, strategy[0], strategy[1]
    investment = initial_investment

    pct_changes = prices_df['Close'].pct_change().fillna(0).to_numpy()

    for pct_change in pct_changes[1:]:

        next_val = investment * (1 + pct_change)
        cum_pct_change = 1 - next_val / initial_investment

        if cum_pct_change <= pct_down or cum_pct_change >= pct_up:
            profit += next_val - initial_investment
            investment = initial_investment
            if profit < required_capital:
                required_capital = profit
        else:
            investment = next_val

    if investment != initial_investment:
        profit += next_val - initial_investment
        if profit < required_capital:
                required_capital = profit
    
    hold_profit = initial_investment * np.cumprod(1 + pct_changes)[-1] - initial_investment
    better = 1 if profit > hold_profit else 0

    to_0x = ceil(log(10**(-4))/log(1+strategy[1]))
    to_2x = ceil(log(2)/log(1+strategy[0]))

    return {
        'better':better,
        'ticker':ticker,
        'period':period,
        'up_pct':strategy[0],
        'down_pct':strategy[1],
        'strat_profit':profit,
        'required_capital':abs(required_capital),
        'hold_profit':hold_profit,
        'price_movements_to_0x':to_0x,
        'price_movements_to_2x':to_2x
    }

def write_data(
    period: str = '1y',
    interval: str = '1d',
    sample_size: int = 5000,
    invest: int = 1000,
    down_max_pct: int = 100,
    up_max_pct: int = 101,
    up_retrieval: bool = True,
    start_date: str = "",
    end_date: str = ""
):
    
    un_period = False
    if start_date != "" and end_date != "":
        un_period = True
        d_s = datetime.strptime(start_date, "%Y-%m-%d")
        d_e = datetime.strptime(end_date, "%Y-%m-%d")
        period = str((d_e - d_s).days) + "d"

    NYSE_FILE = Path("/Users/tristanwinter/Trading/csv_files/nyse.csv")
    NASDAQ_FILE = Path("/Users/tristanwinter/Trading/csv_files/nasdaq.csv")
    RESULTS_FILE = Path(f"/Users/tristanwinter/Trading/csv_files/results_{period}_{interval}_{up_max_pct}_{down_max_pct}_{up_retrieval}.csv")

    if not NYSE_FILE.exists() or not NASDAQ_FILE.exists():
        write_tickers(NYSE_FILE,NASDAQ_FILE)

    sample = pull_tickers_sample(sample_size,NYSE_FILE,NASDAQ_FILE)

    sample = [tick for tick in sample if tick not in list(YAHOO_MAP.values())]

    results_df = pd.DataFrame(columns=[
        'better',
        'ticker',
        'period',
        'up_pct',
        'down_pct',
        'strat_profit',
        'required_capital',
        'hold_profit',
        'price_movements_to_0x',
        'price_movements_to_2x',
        'hist_volatility',
        'gk_volatility'
    ])

    for ticker in tqdm(sample):
        if un_period:
            prices_df = yf.Ticker(ticker).history(start=start_date,end=end_date,interval=interval)[['High', 'Low', 'Open', 'Close', 'Volume']]
        else:
            prices_df = yf.Ticker(ticker).history(period=period,interval=interval)[['High', 'Low', 'Open', 'Close', 'Volume']]
        max_profit = {'strat_profit': -np.inf}
        down_max = min(100,down_max_pct)
        for down in range(1, down_max):
        # can make taking profits only when profits able to recover losses i.e.
        # (1 + up_pct)(1 - |down_pct|) >= 1 => up_pct >= |down_pct|/(1 - |down_pct|)
        # by changing start range for up
            down_pct = -down/100
            up_pct_min = ceil(100*(-down_pct)/(1 + down_pct)) if up_retrieval else 1
            up_pct_min = 100 if up_pct_min > 100 else up_pct_min
            up_max = min(101,up_max_pct) if up_max_pct > up_pct_min else up_pct_min + 1
            for up in range(up_pct_min, up_max): 
            # just reinvest strategy by setting large profit take i.e. range(100,101)
                up_pct = up/100
                result = calc_profit(
                    strategy=(up_pct,down_pct),
                    prices_df=prices_df,
                    ticker=ticker,
                    period=period,
                    initial_investment=invest
                )
                if result != None:
                    if result['strat_profit'] > max_profit['strat_profit']:
                        max_profit = result
        prices_df_copy = prices_df.copy()
        row = {**max_profit}
        gk = garman_klass_vol(prices_df)['Volatility']
        gk_dict = {str(k): v for k, v in gk.dropna().items()}
        row['gk_volatility'] = gk_dict if gk_dict else np.nan
        hist = real_vol(prices_df)['Volatility']
        hist_dict = {str(k): v for k, v in hist.dropna().items()}
        row['hist_volatility'] = hist_dict if hist_dict else np.nan
        results_df = pd.concat([results_df, pd.DataFrame([row])],ignore_index=True)

    results_df = results_df.dropna()
    results_df['diff_in_profits'] = results_df['strat_profit'] - results_df['hold_profit']

    if not results_df.empty:
        results_df.to_csv(RESULTS_FILE, index=False)