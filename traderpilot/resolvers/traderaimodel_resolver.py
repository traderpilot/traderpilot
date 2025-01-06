# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for traderai
"""

import logging
from pathlib import Path

from traderpilot.constants import USERPATH_TRADERAIMODELS, Config
from traderpilot.exceptions import OperationalException
from traderpilot.traderai.traderai_interface import ITraderaiModel
from traderpilot.resolvers import IResolver


logger = logging.getLogger(__name__)


class TraderaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = ITraderaiModel
    object_type_str = "TraderaiModel"
    user_subdir = USERPATH_TRADERAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("traderai/prediction_models").resolve()
    )
    extra_path = "traderaimodel_path"

    @staticmethod
    def load_traderaimodel(config: Config) -> ITraderaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        traderaimodel_name = config.get("traderaimodel")
        if not traderaimodel_name:
            raise OperationalException(
                "No traderaimodel set. Please use `--traderaimodel` to "
                "specify the TraderaiModel class to use.\n"
            )
        if traderaimodel_name in disallowed_models:
            raise OperationalException(
                f"{traderaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        traderaimodel = TraderaiModelResolver.load_object(
            traderaimodel_name,
            config,
            kwargs={"config": config},
        )

        return traderaimodel
