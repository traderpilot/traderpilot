"""Crypto.com exchange subclass"""

import logging

from traderpilot.exchange import Exchange
from traderpilot.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Cryptocom(Exchange):
    """Crypto.com exchange class.
    Contains adjustments needed for Traderpilot to work with this exchange.
    """

    _tp_has: FtHas = {
        "ohlcv_candle_limit": 300,
    }
