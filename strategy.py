from pathlib import Path
import pandas as pd
import yfinance as yf
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import brentq, minimize_scalar
from typing import List, Dict
from math import sqrt, log
from backtests import write_data
from config import State, FileParams, YAHOO_MAP
import logging

def y_ticker(t212_ticker):
    return YAHOO_MAP[t212_ticker]

def plot_relation(
    vars: List[str,str,str],
    df: pd.DataFrame,
    plot: bool = False
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df[vars[0]].values
    y = df[vars[1]].values
    z = df[vars[2]].values

    ax.scatter(x, y, z)

    X = np.column_stack((x, y))

    model = LinearRegression()
    model.fit(X, z)

    x_grid, y_grid = np.meshgrid(
        np.linspace(x.min(), x.max(), 20),
        np.linspace(y.min(), y.max(), 20)
    )

    z_grid = model.predict(
        np.column_stack(
            (x_grid.ravel(), y_grid.ravel())
        )
    ).reshape(x_grid.shape)

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        alpha=0.5
    )

    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel(vars[2])

    if plot:
        plt.show()

    return model

def strategy_line(
    var: float,
    model,
    up_range: tuple = (0.01, 1.0),
    steps: int = 50
) -> pd.DataFrame:
    a, b = model.coef_
    c = model.intercept_

    up_values = np.linspace(
        up_range[0],
        up_range[1],
        steps
    )

    down_values = (
        var - c - a * up_values
    ) / b

    return pd.DataFrame({
        "up_pct": up_values,
        "down_pct": down_values
    })

def closest_point(
    line1: pd.DataFrame,
    line2: pd.DataFrame
) -> Dict:

    def distance(up):
        d1 = np.interp(
            up,
            line1['up_pct'],
            line1['down_pct']
        )

        d2 = np.interp(
            up,
            line2['up_pct'],
            line2['down_pct']
        )

        return abs(d1 - d2)

    lower = max(
        line1['up_pct'].min(),
        line2['up_pct'].min()
    )
    upper = min(
        line1['up_pct'].max(),
        line2['up_pct'].max()
    )
    result = minimize_scalar(
        distance,
        bounds=(lower, upper),
        method='bounded'
    )

    up = result.x

    down1 = np.interp(
        up,
        line1['up_pct'],
        line1['down_pct']
    )
    down2 = np.interp(
        up,
        line2['up_pct'],
        line2['down_pct']
    )

    return {
        "up_pct": up,
        "down_pct": (down1 + down2)/2
    }

def find_intersection(
    line1: pd.DataFrame,
    line2: pd.DataFrame
) -> Dict:

    def difference(up):
        return (
            np.interp(
                up,
                line1['up_pct'],
                line1['down_pct']
            )
            -
            np.interp(
                up,
                line2['up_pct'],
                line2['down_pct']
            )
        )

    ups = np.linspace(
        max(line1['up_pct'].min(), line2['up_pct'].min()),
        min(line1['up_pct'].max(), line2['up_pct'].max()),
        500
    )

    vals = np.array([difference(u) for u in ups])

    idx = np.where(vals[:-1] * vals[1:] < 0)[0]

    if len(idx) == 0:
        return closest_point(line1,line2)

    i = idx[0]

    up = brentq(
        difference,
        ups[i],
        ups[i+1]
    )

    down = np.interp(
        up,
        line1['up_pct'],
        line1['down_pct']
    )

    return {
        'up_pct': up,
        'down_pct': down
    }

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

class Signaller:

    def create_data(self,
                    period: str ='1y',
                    interval: str = '1d',
                    sample_size: int = 5000,
                    invest: int = 1000,
                    down_max: int = 100,
                    up_max: int = 101,
                    up_retrieval: bool = True,
                    start_date: str = "",
                    end_date: str = ""
                    ):
        
        write_data(period=period,
                   interval=interval,
                   sample_size=sample_size,
                   invest=invest,
                   down_max_pct=down_max,
                   up_max_pct=up_max,
                   up_retrieval=up_retrieval,
                   start_date=start_date,
                   end_date=end_date
                   )


    def get_signal(self,
                   ticker: str,
                   file_params: FileParams
                   ) -> Dict:

        RESULTS_FILE = Path(f'/Users/tristanwinter/Trading/csv_files/results_{file_params["period"]}_{file_params["interval"]}_{file_params["up_max"]}_{file_params["down_max"]}_{file_params["up_retrieval"]}.csv')
        results_df = pd.read_csv(RESULTS_FILE)

        results_df['gk_volatility'] = results_df['gk_volatility'].apply(ast.literal_eval)
        results_df['avg_gk_vol'] = results_df['gk_volatility'].apply(lambda d: np.mean(list(d.values())))
        results_df['hist_volatility'] = results_df['hist_volatility'].apply(ast.literal_eval)
        results_df['avg_hist_vol'] = results_df['hist_volatility'].apply(lambda d: np.mean(list(d.values())))

        strat_results_df = results_df[results_df['better'] == 1].copy()

        gk_model = plot_relation([
            'up_pct',
            'down_pct',
            'avg_gk_vol'
        ],
        df = results_df
        )
        gk_model_strat = plot_relation([
            'up_pct',
            'down_pct',
            'avg_gk_vol'
        ],
        df = strat_results_df
        )
        hist_model = plot_relation([
            'up_pct',
            'down_pct',
            'avg_hist_vol'
        ],
        df = results_df
        )
        hist_model_strat = plot_relation([
            'up_pct',
            'down_pct',
            'avg_hist_vol'
        ],
        df = strat_results_df
        )

        data = yf.Ticker(ticker).history(period=file_params["period"],interval=file_params['interval'])
        gk_vol_avg = garman_klass_vol(data)['Volatility'].mean()
        hist_vol_avg = real_vol(data)['Volatility'].mean()

        gk_pred_df = strategy_line(gk_vol_avg,gk_model)
        gk_pred_df = gk_pred_df[gk_pred_df['down_pct'] <= 0]
        hist_pred_df = strategy_line(hist_vol_avg,hist_model)
        hist_pred_df = hist_pred_df[hist_pred_df['down_pct'] <= 0]

        gk_pred_df_strat = strategy_line(gk_vol_avg,gk_model_strat)
        gk_pred_df_strat = gk_pred_df_strat[gk_pred_df_strat['down_pct'] <= 0]
        hist_pred_df_strat = strategy_line(hist_vol_avg,hist_model_strat)
        hist_pred_df_strat = hist_pred_df_strat[hist_pred_df_strat['down_pct'] <= 0]

        if (gk_pred_df.empty or hist_pred_df.empty) and (gk_pred_df_strat.empty or hist_pred_df_strat.empty):
            return {
                'both':{
                    'up_pct':0.0,
                    'down_pct':0.0
                },
                'just_strat':{
                    'up_pct':0.0,
                    'down_pct':0.0
                }
            }
        elif gk_pred_df.empty or hist_pred_df.empty:
            return {
                'both':{
                    'up_pct':0.0,
                    'down_pct':0.0
                },
                'just_strat':find_intersection(gk_pred_df_strat,hist_pred_df_strat)
            }
        elif gk_pred_df_strat.empty or hist_pred_df_strat.empty:
            return {
                'both':find_intersection(gk_pred_df,hist_pred_df),
                'just_strat':{
                    'up_pct':0.0,
                    'down_pct':0.0
                }
            }

        return {
            'both':find_intersection(gk_pred_df,hist_pred_df),
            'just_strat':find_intersection(gk_pred_df_strat,hist_pred_df_strat)
        }
    
    def show_volatility_profit_relation(self,
                                        file_params: FileParams
                                        ):

        RESULTS_FILE = Path(f'/Users/tristanwinter/Trading/csv_files/results_{file_params["period"]}_{file_params["interval"]}_{file_params["up_max"]}_{file_params["down_max"]}_{file_params["up_retrieval"]}.csv')
        results_df = pd.read_csv(RESULTS_FILE)

        results_df['gk_volatility'] = results_df['gk_volatility'].apply(ast.literal_eval)
        results_df['avg_gk_vol'] = results_df['gk_volatility'].apply(lambda d: np.mean(list(d.values())))
        results_df['hist_volatility'] = results_df['hist_volatility'].apply(ast.literal_eval)
        results_df['avg_hist_vol'] = results_df['hist_volatility'].apply(lambda d: np.mean(list(d.values())))

        strat_results_df = results_df[results_df['better'] == 1].copy()

        plot_relation([
            'diff',
            'avg_gk_vol',
            'avg_hist_vol'
        ],
        df = strat_results_df,
        plot = True
        )

    def check(
            self,
            current_price: float,
            position_quantity: float,
            state: State,
            file_params: FileParams,
            just_strat: bool = True
    ) -> Dict:
        
        logging.basicConfig(
            filename=Path("bot.log"),
            level=logging.INFO
        )

        ticker = y_ticker(state['ticker'])
        target_value = state['target_value']
        target_value = state['target_value']

        strat = 'just_strat' if just_strat else 'both'

        up_down_signal = self.get_signal(ticker,file_params)
        threshold_up = float(up_down_signal[strat]['up_pct'])
        threshold_down = float(up_down_signal[strat]['down_pct'])

        if threshold_up == 0.0 and threshold_down == 0.0:
            logging.info(f"No data to build model for strat: {strat} on \n{file_params}")
            print(f'\nNo data to build model for strat: {strat}')
            return None

        position_value = position_quantity*current_price
        exc_def = position_value - target_value
        move = exc_def/target_value

        print('\nChecking for Signal...')
        print(f'\nUpper Percentage: {threshold_up}')
        print(f'\nLower Percentage: {threshold_down}')
        print(f'\nTotal Move: {move*target_value}')
        print(f'\nCurrent Percentage Move: {move}')

        if move >= threshold_up:

            if exc_def <= 0:
                return None
            
            print('\nSell order identified')
            
            return {
                'ticker':ticker,
                'action':'sell',
                'quantity':-exc_def/current_price
            }
        
        if move <= threshold_down:

            if exc_def >= 0:
                return None
            
            print('\nBuy order identified')
            
            return {
                'ticker':ticker,
                'action':'buy',
                'quantity':-exc_def/current_price
            }
        
        return None