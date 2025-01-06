"""Bitmart exchange subclass"""

import logging

from traderpilot.exchange import Exchange
from traderpilot.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitmart(Exchange):
    """
    Bitmart exchange class. Contains adjustments needed for Traderpilot to work
    with this exchange.
    """

    _tp_has: FtHas = {
        "stoploss_on_exchange": False,  # Bitmart API does not support stoploss orders
        "ohlcv_candle_limit": 200,
        "trades_has_history": False,  # Endpoint doesn't seem to support pagination
    }
