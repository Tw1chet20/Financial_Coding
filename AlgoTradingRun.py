# AlgoTradingRun.py

import AlgoTradingBacktester

tickers = ["NVDA","AAPL","MSFT"]
cash = 10_000
max_tries = 100

days_look_back = 365
interval_length = "1d"
windows = [2, 5, 10]
lag_windows = [1, 2, 5, 10]
vol_windows = [5, 10, 20]
ema = True
use_adj_close = True
commission = 0.002
save_plot = False
save_model = True
plot_strat = False
plot_stock_price = False
optimize_param = "Return [%]"

selection_metric = 'return_pct'
selection_direction = 'max'

modelBacktester = AlgoTradingBacktester.BuildAndBacktestModel(
    tickers=tickers,
    cash=cash,
    max_tries=max_tries,
    save_model=save_model
)

modelBacktester.build()

selection_rule = AlgoTradingBacktester.SelectionRule(
    metric=selection_metric,
    direction=selection_direction
    )

best_models = {}

for ticker in tickers:
    modelLoader = AlgoTradingBacktester.BestModelLoader(ticker=ticker)
    best_models[ticker] = modelLoader.loadBestModel(rule=selection_rule)
    print('\n')
    print('\n')
    print(best_models[ticker])
