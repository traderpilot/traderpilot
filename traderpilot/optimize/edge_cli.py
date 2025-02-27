# pragma pylint: disable=missing-docstring, W0212, too-many-arguments

"""
This module contains the edge backtesting interface
"""

import logging

from traderpilot import constants
from traderpilot.configuration import TimeRange, validate_config_consistency
from traderpilot.constants import Config
from traderpilot.data.dataprovider import DataProvider
from traderpilot.edge import Edge
from traderpilot.optimize.optimize_reports import generate_edge_table
from traderpilot.resolvers import ExchangeResolver, StrategyResolver


logger = logging.getLogger(__name__)


class EdgeCli:
    """
    EdgeCli class, this class contains all the logic to run edge backtesting

    To run a edge backtest:
    edge = EdgeCli(config)
    edge.start()
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # Ensure using dry-run
        self.config["dry_run"] = True
        self.config["stake_amount"] = constants.UNLIMITED_STAKE_AMOUNT
        self.exchange = ExchangeResolver.load_exchange(self.config)
        self.strategy = StrategyResolver.load_strategy(self.config)
        self.strategy.dp = DataProvider(config, self.exchange)

        validate_config_consistency(self.config)

        self.edge = Edge(config, self.exchange, self.strategy)
        # Set refresh_pairs to false for edge-cli (it must be true for edge)
        self.edge._refresh_pairs = False

        self.edge._timerange = TimeRange.parse_timerange(
            None if self.config.get("timerange") is None else str(self.config.get("timerange"))
        )
        self.strategy.tp_bot_start()

    def start(self) -> None:
        result = self.edge.calculate(self.config["exchange"]["pair_whitelist"])
        if result:
            print("")  # blank line for readability
            generate_edge_table(self.edge._cached_pairs)
