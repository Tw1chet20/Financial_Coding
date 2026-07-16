import yfinance as yf


class MarketData:

    def get_stock_data(
            self,
            ticker: str,
            period: str,
            interval: str
            ):

        stock = yf.Ticker(ticker)

        return stock.history(
            period=period,
            interval=interval
        ).dropna()
    
    def get_price(self, ticker: str):

        stock = yf.Ticker(ticker)

        return stock.fast_info['last_price']