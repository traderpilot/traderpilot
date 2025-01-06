import logging

from traderpilot.constants import Config
from traderpilot.enums import TradingMode
from traderpilot.exchange import Exchange


logger = logging.getLogger(__name__)


def migrate_funding_fee_timeframe(config: Config, exchange: Exchange | None):
    from traderpilot.data.history import get_datahandler

    if config.get("trading_mode", TradingMode.SPOT) != TradingMode.FUTURES:
        # only act on futures
        return

    if not exchange:
        from traderpilot.resolvers import ExchangeResolver

        exchange = ExchangeResolver.load_exchange(config, validate=False)

    ff_timeframe = exchange.get_option("funding_fee_timeframe")

    dhc = get_datahandler(config["datadir"], config["dataformat_ohlcv"])
    dhc.fix_funding_fee_timeframe(ff_timeframe)
