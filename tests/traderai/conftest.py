import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.conftest import get_patched_exchange
from traderpilot.configuration import TimeRange
from traderpilot.data.dataprovider import DataProvider
from traderpilot.resolvers import StrategyResolver
from traderpilot.resolvers.traderaimodel_resolver import TraderaiModelResolver
from traderpilot.traderai.data_drawer import TraderaiDataDrawer
from traderpilot.traderai.data_kitchen import TraderaiDataKitchen


def is_py12() -> bool:
    return sys.version_info >= (3, 12)


@pytest.fixture(scope="function")
def traderai_conf(default_conf, tmp_path):
    traderaiconf = deepcopy(default_conf)
    traderaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "runmode": "backtest",
            "strategy": "traderai_test_strat",
            "user_data_dir": tmp_path,
            "strategy-path": "traderpilot/tests/strategy/strats",
            "traderaimodel": "LightGBMRegressor",
            "traderaimodel_path": "traderai/prediction_models",
            "timerange": "20180110-20180115",
            "traderai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "unique-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0,
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path("config_examples", "config_traderai.example.json")],
        }
    )
    traderaiconf["exchange"].update(
        {"pair_whitelist": ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC"]}
    )
    return traderaiconf


def make_rl_config(conf):
    conf.update({"strategy": "traderai_rl_test_strat"})
    conf["traderai"].update(
        {"model_training_parameters": {"learning_rate": 0.00025, "gamma": 0.9, "verbose": 1}}
    )
    conf["traderai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {"rr": 1, "profit_aim": 0.02, "win_reward_factor": 2},
        "drop_ohlc_from_features": False,
    }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "trainer_kwargs": {
            "n_steps": None,
            "batch_size": 64,
            "n_epochs": 1,
        },
        "model_kwargs": {
            "hidden_dim": 32,
            "dropout_percent": 0.2,
            "n_layer": 1,
        },
    }


def get_patched_data_kitchen(mocker, traderaiconf):
    dk = TraderaiDataKitchen(traderaiconf)
    return dk


def get_patched_data_drawer(mocker, traderaiconf):
    # dd = mocker.patch('traderpilot.traderai.data_drawer', MagicMock())
    dd = TraderaiDataDrawer(traderaiconf)
    return dd


def get_patched_traderai_strategy(mocker, traderaiconf):
    strategy = StrategyResolver.load_strategy(traderaiconf)
    strategy.tp_bot_start()

    return strategy


def get_patched_traderaimodel(mocker, traderaiconf):
    traderaimodel = TraderaiModelResolver.load_traderaimodel(traderaiconf)

    return traderaimodel


def make_unfiltered_dataframe(mocker, traderai_conf):
    traderai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    traderai.dk.live = True
    traderai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(data_load_timerange, traderai.dk)

    traderai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = traderai.dd.get_base_and_corr_dataframes(
        data_load_timerange, traderai.dk.pair, traderai.dk
    )

    unfiltered_dataframe = traderai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, traderai.dk.pair
    )
    for i in range(5):
        unfiltered_dataframe[f"constant_{i}"] = i

    unfiltered_dataframe = traderai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return traderai, unfiltered_dataframe


def make_data_dictionary(mocker, traderai_conf):
    traderai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    traderai.dk.live = True
    traderai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(data_load_timerange, traderai.dk)

    traderai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = traderai.dd.get_base_and_corr_dataframes(
        data_load_timerange, traderai.dk.pair, traderai.dk
    )

    unfiltered_dataframe = traderai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, traderai.dk.pair
    )

    unfiltered_dataframe = traderai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    traderai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = traderai.dk.filter_features(
        unfiltered_dataframe,
        traderai.dk.training_features_list,
        traderai.dk.label_list,
        training_filter=True,
    )

    data_dictionary = traderai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = traderai.dk.normalize_data(data_dictionary)

    return traderai


def get_traderai_live_analyzed_dataframe(mocker, traderaiconf):
    strategy = get_patched_traderai_strategy(mocker, traderaiconf)
    exchange = get_patched_exchange(mocker, traderaiconf)
    strategy.dp = DataProvider(traderaiconf, exchange)
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderaiconf, traderai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    traderai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair("ADA/BTC", "5m")
    return strategy.dp.get_analyzed_dataframe("ADA/BTC", "5m")


def get_traderai_analyzed_dataframe(mocker, traderaiconf):
    strategy = get_patched_traderai_strategy(mocker, traderaiconf)
    exchange = get_patched_exchange(mocker, traderaiconf)
    strategy.dp = DataProvider(traderaiconf, exchange)
    strategy.traderai_info = traderaiconf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderaiconf, traderai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    traderai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = traderai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return traderai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")


def get_ready_to_train(mocker, traderaiconf):
    strategy = get_patched_traderai_strategy(mocker, traderaiconf)
    exchange = get_patched_exchange(mocker, traderaiconf)
    strategy.dp = DataProvider(traderaiconf, exchange)
    strategy.traderai_info = traderaiconf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderaiconf, traderai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    traderai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = traderai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, traderai, strategy
