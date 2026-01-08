# AlgoTradingBacktester.py

import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
from backtesting import Backtest
import AlgoTradingStrategies
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Config
# -----------------------------

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def build_features(
        df, 
        price_col,
        windows,
        lag_windows,
        vol_windows,
        ema
        ):

    sorted_windows = sorted(set(windows))

    for w in sorted_windows:
        col = f"{'EMA' if ema else 'SMA'}_{w}"
        if ema:
            df[col] = df[price_col].ewm(span=w, adjust=False).mean()
        else:
            df[col] = df[price_col].rolling(window=w, min_periods=1).mean()
    
    for lag in lag_windows:
        df[f"ret_{lag}"] = df[price_col].pct_change(lag)

    # EMA spreads
    for i in range(len(windows) - 1):
        w1, w2 = sorted_windows[i], sorted_windows[i+1]
        df[f"ema_{w1}_{w2}_spread"] = (df[f"EMA_{w1}"] - df[f"EMA_{w2}"]) / df[f"EMA_{w2}"]

    # EMA slopes
    for w in windows:
        df[f"ema_{w}_slope"] = df[f"EMA_{w}"].pct_change(3)

    # Volatility
    for vol in vol_windows:
        df[f"vol_{vol}"] = df[price_col].pct_change().rolling(vol).std()

    return df

def download_and_features(
        ticker,
        days_look_back,
        interval_length,
        windows,
        lag_windows,
        vol_windows,
        ema,
        use_adj_close
        ):
    end_dt = dt.datetime.now()
    start_dt = end_dt - dt.timedelta(days=days_look_back)

    df = yf.download(
        ticker,
        start=start_dt,
        end=end_dt,
        interval=interval_length,
        auto_adjust=False,
        group_by="column",
        progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Try a different interval/lookback.")
    
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing required columns for Backtest: {missing}")

    price_col = "Adj Close" if use_adj_close and "Adj Close" in df.columns else "Close"

    df["Change Tomorrow"] = (df[price_col].shift(-1) / df[price_col] - 1) * 100
    df['Target Up'] = (df['Change Tomorrow'] > 0.01).astype(int)

    df = build_features(
        df,
        price_col,
        windows,
        lag_windows,
        vol_windows,
        ema
        )

    # Drop the last row because Change Tomorrow is NaN there
    df = df.dropna(subset=["Change Tomorrow","Target Up"])

    return df, price_col


def plot_price_and_mas(
        df,
        price_col,
        ticker,
        days_look_back,
        interval_length,
        ema,
        save_plot,
        windows
        ):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[price_col], label=price_col)

    for w in sorted(set(windows)):
        col = f"{'EMA' if ema else 'SMA'}_{w}"
        plt.plot(df.index, df[col], label=col, alpha=0.4)

    plt.title(ticker)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    if save_plot:
        filename = f"{ticker}_{interval_length}_{days_look_back}d_{'EMA' if ema else 'SMA'}.png"
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to: {out_path.resolve()}")

    plt.show()
    plt.close()

# -----------------------------
# Run Class
# -----------------------------
class BuildAndBacktestModel:
    def __init__(
            self,
            tickers: list,
            cash: float,
            max_tries: int,
            optimize_param: str = "Return [%]",
            days_look_back: int = 365,
            interval_length: str = "1d",
            windows: list = [2, 5, 10],
            lag_windows: list = [1, 2, 5, 10],
            vol_windows: list = [5, 10, 20],
            ema: bool = True,
            use_adj_close: bool = True,
            commission: float = 0.002,
            save_plot: bool = False,
            save_model: bool = False,
            plot_strat: bool = False,
            plot_stock_price: bool = False
    ):
        self.tickers = tickers
        self.cash = cash
        self.max_tries = max_tries
        self.optimize_param = optimize_param
        self.days_look_back = days_look_back
        self.interval_length = interval_length
        self.windows = windows
        self.lag_windows = lag_windows
        self.vol_windows = vol_windows
        self.ema = ema
        self.use_adj_close = use_adj_close
        self.commission = commission
        self.save_plot = save_plot
        self.save_model = save_model
        self.plot_strat = plot_strat
        self.plot_stock_price = plot_stock_price

    def build(self):

        buy_grid  = np.round(np.linspace(0.50, 0.70, 21), 3).tolist()
        sell_grid = np.round(np.linspace(0.30, 0.50, 21), 3).tolist()

        results = {}

        for ticker in self.tickers:
            df, price_col = download_and_features(
                ticker,
                self.days_look_back,
                self.interval_length,
                self.windows,
                self.lag_windows,
                self.vol_windows,
                self.ema,
                self.use_adj_close
            )
    
            if self.plot_stock_price:
                plot_price_and_mas(
                    df,
                    price_col,
                    ticker,
                    self.days_look_back,
                    self.interval_length,
                    self.ema,
                    self.save_plot,
                    self.windows
                )

            bt_anchored = Backtest(
                df,
                AlgoTradingStrategies.WalkForwardAnchored,
                cash=self.cash,
                commission=self.commission,
                exclusive_orders=True,
                finalize_trades=True
            )

            stats_anc, heatmap_anc, opt_anc = bt_anchored.optimize(
                buy_prob=buy_grid,
                sell_prob=sell_grid,
                maximize=self.optimize_param,
                max_tries=self.max_tries,
                random_state=42,
                return_heatmap=True,
                return_optimization=True,
                method="sambo",
            )

            bt_unanchored = Backtest(
                df,
                AlgoTradingStrategies.WalkForwardUnanchored,
                cash=self.cash,
                commission=self.commission,
                exclusive_orders=True,
                finalize_trades=True
            )

            stats_unanc, heatmap_unanc, opt_unanc = bt_unanchored.optimize(
                buy_prob=buy_grid,
                sell_prob=sell_grid,
                maximize=self.optimize_param,
                max_tries=self.max_tries,
                random_state=42,
                return_heatmap=True,
                return_optimization=True,
                method="sambo",
            )

            results[ticker] = {
                "price_col": price_col,
                "anchored": {
                    "stats": stats_anc,
                    "heatmap": heatmap_anc,
                    "opt": opt_anc,
                },
                "unanchored": {
                    "stats": stats_unanc,
                    "heatmap": heatmap_unanc,
                    "opt": opt_unanc,
                }
            }

            if self.plot_strat:
                bt_anchored.plot(filename=f"{ticker}_walk_forward_anchored.html", open_browser=False)
                bt_unanchored.plot(filename=f"{ticker}_walk_forward_unanchored.html", open_browser=False)

            print(f"\n===== {ticker} =====")
            print(f"Anchored Return [%]: {stats_anc["Return [%]"]}")
            print(f"Unanchored Return [%]: {stats_unanc["Return [%]"]}")

            if self.save_model:
                best_strategy = None
                run_id = f"{ticker}_model_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"
                rows = []

                if stats_anc["Return [%]"] >= stats_unanc["Return [%]"]:
                    best_strategy = stats_anc._strategy
                    rows.append({
                        "run_id": run_id,
                        "ticker": ticker,
                        "strategy": "anchored",
                        "return_pct": stats_anc["Return [%]"],
                        "sharpe": stats_anc["Sharpe Ratio"],
                        "max_dd": stats_anc["Max. Drawdown [%]"],
                        "trades": stats_anc["# Trades"],
                    })
                else:
                    best_strategy = stats_unanc._strategy
                    rows.append({
                        "run_id": run_id,
                        "ticker": ticker,
                        "strategy": "unanchored",
                        "return_pct": stats_unanc["Return [%]"],
                        "sharpe": stats_unanc["Sharpe Ratio"],
                        "max_dd": stats_unanc["Max. Drawdown [%]"],
                        "trades": stats_unanc["# Trades"],
                    })

                model_payload = {
                    "run_id": run_id,
                    "ticker": ticker,
                    "price_col": price_col,
                    "feature_config": {
                        "windows": self.windows,
                        "lag_windows": self.lag_windows,
                        "vol_windows": self.vol_windows,
                        "ema": self.ema,
                        "use_adj_close": self.use_adj_close,
                        "days_look_back": self.days_look_back,
                        "interval_length": self.interval_length,
                        "drop_cols": AlgoTradingStrategies.DROP_COLS,
                        "target_col": AlgoTradingStrategies.TARGET_COL,
                    },
                    "model": best_strategy.model,
                    "feature_cols": best_strategy._feature_cols,
                    "params": {
                        "buy_prob": best_strategy.buy_prob,
                        "sell_prob": best_strategy.sell_prob,
                        "n_train": best_strategy.n_train,
                        "retrain_every": best_strategy.retrain_every,
                        "strategy_type": "anchored" if stats_anc["Return [%]"] >= stats_unanc["Return [%]"] else "unanchored",
                    }
                }

                model_path = model_dir / f"{run_id}.pkl"
                joblib.dump(model_payload, model_path)

                results_df = pd.DataFrame(rows)
                results_path = results_dir / f"results_{ticker}.csv"
                write_header = not results_path.exists()
                results_df.to_csv(results_path, mode="a", header=write_header, index=False)

class SelectionRule():
    def __init__(
            self,
            metric: str = "return_pct",
            direction: str = "max"
    ):
        self.metric = metric
        self.direction = direction

class BestModelLoader():

    def __init__(
            self,
            ticker
    ):
        self.ticker = ticker

    def list_runs(self, ticker: str) -> pd.DataFrame:
        path = results_dir / f"results_{self.ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No results file found for {ticker}: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Results file is empty for {ticker}: {path}")
        return df
    
    def select_best_run(
            self,
            require_min_trades: int = 0,
            rule = SelectionRule,
            ) -> pd.Series:
        df = self.list_runs(self.ticker).copy()

        if require_min_trades > 0 and "trades" in df.columns:
            df = df[df["trades"] >= require_min_trades]

        if df.empty:
            raise ValueError(f"No candidate runs left for {self.ticker} after filtering.")

        if not rule.metric:
            raise ValueError("SelectionRule must have either metric or score_fn.")

        if rule.metric not in df.columns:
            raise KeyError(f"Metric '{rule.metric}' not found in results columns: {list(df.columns)}")
        
        ascending = (rule.direction == "min")
        best_idx = df[rule.metric].sort_values(ascending=ascending).index[0]
        return df.loc[best_idx]
    
    def load_payload(self, run_id: str) -> dict:
        pkl_path = model_dir / f"{run_id}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Model file not found: {pkl_path}")
        payload = joblib.load(pkl_path)
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected payload type in {pkl_path}: {type(payload)}")
        return payload
    

    def loadBestModel(
            self,
            require_min_trades: int = 0,
            rule = SelectionRule
            ):
        best_row = self.select_best_run(require_min_trades=require_min_trades,rule=rule)
        run_id = str(best_row["run_id"])
        payload = self.load_payload(run_id)

        return payload