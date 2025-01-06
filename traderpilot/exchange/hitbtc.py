import logging

from traderpilot.exchange import Exchange
from traderpilot.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Hitbtc(Exchange):
    """
    Hitbtc exchange class. Contains adjustments needed for Traderpilot to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Traderpilot development team. So some features
    may still not work as expected.
    """

    _tp_has: FtHas = {
        "ohlcv_candle_limit": 1000,
    }
