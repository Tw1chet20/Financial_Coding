import requests
from requests.auth import HTTPBasicAuth
from config import API_KEY, API_SECRET, BASE_URL, T212_MAP
import time
from typing import Dict

class Trading212Client:

    def __init__(self):
        self.auth = HTTPBasicAuth(
            API_KEY,
            API_SECRET
        )


    def _get(self, endpoint: str):

        while True:

            r = requests.get(
                BASE_URL + endpoint,
                auth=self.auth
            )

            if r.status_code == 429:

                print(
                    "\nRate limit reached. Sleeping..."
                )

                time.sleep(10)

                continue

            r.raise_for_status()

            return r.json()


    def _post(self, endpoint: str, payload):

        r = requests.post(
            BASE_URL + endpoint,
            auth=self.auth,
            json=payload
        )

        print("PAYLOAD:", payload)
        print("STATUS:", r.status_code)
        print("RESPONSE:", r.text)

        r.raise_for_status()

        return r.json()


    def get_cash(self):
        return self._get(
            "/equity/account/cash"
        )


    def get_positions(self):
        return self._get(
            "/equity/portfolio"
        )


    def get_orders(self):
        return self._get(
            "/equity/orders"
        )
    

    def get_instruments(self):
        return self._get(
            "/equity/metadata/instruments"
        )
    

    def place_market_order(self, ticker: str, quantity: float):

        payload = {
            "extendedHours":False,
            "quantity": quantity,
            "ticker": ticker
        }

        return self._post(
            "/equity/orders/market",
            payload
        )
    
    def get_position_quantity(self, ticker: str) -> float:

        positions = self.get_positions()

        for p in positions:

            if p['ticker'] == ticker:
                return float(p['quantity'])
            
        return 0.0
    
    def get_order(self, order_id):

        orders = self.get_orders()
        print('\n')
        print(orders)

        for order in orders:
            if order["id"] == order_id:
                return order

        return None
    
    def close_position(self, ticker: str):

        quantity = self.get_position_quantity(ticker)

        payload = {
            "extendedHours":False,
            "quantity": -quantity,
            "ticker": ticker
        }

        return self._post(
            "/equity/orders/market",
            payload
        )
    
    def open_position(self, ticker: str, quantity: float):

        payload = {
            "extendedHours":False,
            "quantity": round(quantity,2),
            "ticker": ticker
        }

        return self._post(
            "/equity/orders/market",
            payload
        )

class Executor:

    def __init__(self, client):
        self.client = client


    def execute(self, signal: Dict):
        
        order = self.client.place_market_order(
            T212_MAP[signal["ticker"]],
            round(signal["quantity"], 2)
        )

        print(
            f"Submitted order {order['id']}"
        )

        return order