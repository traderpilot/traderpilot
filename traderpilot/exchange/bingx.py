"""Bingx exchange subclass"""

import logging

from traderpilot.exchange import Exchange
from traderpilot.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bingx(Exchange):
    """
    Bingx exchange class. Contains adjustments needed for Traderpilot to work
    with this exchange.
    """

    _tp_has: FtHas = {
        "ohlcv_candle_limit": 1000,
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "order_time_in_force": ["GTC", "IOC", "PO"],
        "trades_has_history": False,  # Endpoint doesn't seem to support pagination
    }
