from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from traderpilot.commands.optimize_commands import setup_optimize_configuration
from traderpilot.configuration.timerange import TimeRange
from traderpilot.data import history
from traderpilot.data.dataprovider import DataProvider
from traderpilot.enums import RunMode
from traderpilot.enums.candletype import CandleType
from traderpilot.exceptions import OperationalException
from traderpilot.traderai.data_kitchen import TraderaiDataKitchen
from traderpilot.optimize.backtesting import Backtesting
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    get_args,
    get_patched_exchange,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)
from tests.traderai.conftest import get_patched_traderai_strategy


def test_traderai_backtest_start_backtest_list(traderai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch(
        "traderpilot.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("traderpilot.optimize.backtesting.history.load_data")
    mocker.patch("traderpilot.optimize.backtesting.history.get_timerange", return_value=(now, now))

    patched_configuration_load_config_file(mocker, traderai_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1m",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re(
        "Using --strategy-list with TraderAI REQUIRES all strategies to have identical", caplog
    )
    Backtesting.cleanup()


@pytest.mark.parametrize(
    "timeframe, expected_startup_candle_count",
    [
        ("5m", 876),
        ("15m", 492),
        ("1d", 302),
    ],
)
def test_traderai_backtest_load_data(
    traderai_conf, mocker, caplog, timeframe, expected_startup_candle_count
):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch(
        "traderpilot.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("traderpilot.optimize.backtesting.history.load_data")
    mocker.patch("traderpilot.optimize.backtesting.history.get_timerange", return_value=(now, now))
    traderai_conf["timeframe"] = timeframe
    traderai_conf.get("traderai", {}).get("feature_parameters", {}).update({"include_timeframes": []})
    backtesting = Backtesting(deepcopy(traderai_conf))
    backtesting.load_bt_data()

    assert log_has_re(
        f"Increasing startup_candle_count for traderai on {timeframe} "
        f"to {expected_startup_candle_count}",
        caplog,
    )
    assert history.load_data.call_args[1]["startup_candles"] == expected_startup_candle_count

    Backtesting.cleanup()


def test_traderai_backtest_live_models_model_not_found(traderai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch(
        "traderpilot.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("traderpilot.optimize.backtesting.history.load_data")
    mocker.patch("traderpilot.optimize.backtesting.history.get_timerange", return_value=(now, now))
    traderai_conf["timerange"] = ""
    traderai_conf.get("traderai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, traderai_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "5m",
        "--traderai-backtest-live-models",
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(
        OperationalException, match=r".* Historic predictions data is required to run backtest .*"
    ):
        Backtesting(bt_config)

    Backtesting.cleanup()


def test_traderai_backtest_consistent_timerange(mocker, traderai_conf):
    traderai_conf["runmode"] = "backtest"
    mocker.patch(
        "traderpilot.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/USDT:USDT"]),
    )

    gbs = mocker.patch("traderpilot.optimize.backtesting.generate_backtest_stats")

    traderai_conf["candle_type_def"] = CandleType.FUTURES
    traderai_conf.get("exchange", {}).update({"pair_whitelist": ["XRP/USDT:USDT"]})
    traderai_conf.get("traderai", {}).get("feature_parameters", {}).update(
        {"include_timeframes": ["5m", "1h"], "include_corr_pairlist": []}
    )
    traderai_conf["timerange"] = "20211120-20211121"

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)

    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.dk = TraderaiDataKitchen(traderai_conf)

    timerange = TimeRange.parse_timerange("20211115-20211122")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)

    backtesting = Backtesting(deepcopy(traderai_conf))
    backtesting.start()

    assert gbs.call_args[1]["min_date"] == datetime(2021, 11, 20, 0, 0, tzinfo=timezone.utc)
    assert gbs.call_args[1]["max_date"] == datetime(2021, 11, 21, 0, 0, tzinfo=timezone.utc)
    Backtesting.cleanup()
