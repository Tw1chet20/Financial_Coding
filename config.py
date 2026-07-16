import os
from dotenv import load_dotenv
from typing import TypedDict

class State(TypedDict):
    ticker: str
    target_value: float
    position_quantity: float
    current_position_value: float

class FileParams(TypedDict):
    period: str
    interval: str
    down_max: float
    up_max: float
    up_retrieval: bool

YAHOO_MAP = {
    "AAPL_US_EQ": "AAPL",
    "MSFT_US_EQ": "MSFT",
    "NVDA_US_EQ": "NVDA",
    "AZNl_EQ": "AZN.L",
    "QUBT_US_EQ": "QUBT"
}

T212_MAP = {
    "AAPL": "AAPL_US_EQ",
    "MSFT": "MSFT_US_EQ",
    "NVDA": "NVDA_US_EQ",
    "AZNl.L": "AZN_EQ",
    "QUBT": "QUBT_US_EQ"
}

load_dotenv()

API_KEY = os.getenv("T212_API_KEY")
API_SECRET = os.getenv("T212_API_SECRET")

ENV = os.getenv("T212_ENV", "demo")

if ENV == "demo":
    BASE_URL = "https://demo.trading212.com/api/v0"
else:
    BASE_URL = "https://live.trading212.com/api/v0"