import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, is_arm, is_mac, log_has_re
from tests.traderai.conftest import (
    get_patched_traderai_strategy,
    make_rl_config,
    mock_pytorch_mlp_model_training_parameters,
)
from traderpilot.configuration import TimeRange
from traderpilot.data.dataprovider import DataProvider
from traderpilot.enums import RunMode
from traderpilot.optimize.backtesting import Backtesting
from traderpilot.persistence import Trade
from traderpilot.plugins.pairlistmanager import PairListManager
from traderpilot.traderai.data_kitchen import TraderaiDataKitchen
from traderpilot.traderai.utils import download_all_data_for_training, get_required_data_timerange


def can_run_model(model: str) -> None:
    is_pytorch_model = "Reinforcement" in model or "PyTorch" in model

    if is_arm() and "Catboost" in model:
        pytest.skip("CatBoost is not supported on ARM.")

    if is_pytorch_model and is_mac():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize(
    "model, pca, dbscan, float32, can_short, shuffle, buffer, noise",
    [
        ("LightGBMRegressor", True, False, True, True, False, 0, 0),
        ("XGBoostRegressor", False, True, False, True, False, 10, 0.05),
        ("XGBoostRFRegressor", False, False, False, True, False, 0, 0),
        ("CatboostRegressor", False, False, False, True, True, 0, 0),
        ("PyTorchMLPRegressor", False, False, False, False, False, 0, 0),
        ("PyTorchTransformerRegressor", False, False, False, False, False, 0, 0),
        ("ReinforcementLearner", False, True, False, True, False, 0, 0),
        ("ReinforcementLearner_multiproc", False, False, False, True, False, 0, 0),
        ("ReinforcementLearner_test_3ac", False, False, False, False, False, 0, 0),
        ("ReinforcementLearner_test_3ac", False, False, False, True, False, 0, 0),
        ("ReinforcementLearner_test_4ac", False, False, False, True, False, 0, 0),
    ],
)
def test_extract_data_and_train_model_Standard(
    mocker, traderai_conf, model, pca, dbscan, float32, can_short, shuffle, buffer, noise
):
    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = "joblib"
    traderai_conf.update({"traderaimodel": model})
    traderai_conf.update({"timerange": "20180110-20180130"})
    traderai_conf.update({"strategy": "traderai_test_strat"})
    traderai_conf["traderai"]["feature_parameters"].update({"principal_component_analysis": pca})
    traderai_conf["traderai"]["feature_parameters"].update({"use_DBSCAN_to_remove_outliers": dbscan})
    traderai_conf.update({"reduce_df_footprint": float32})
    traderai_conf["traderai"]["feature_parameters"].update({"shuffle_after_split": shuffle})
    traderai_conf["traderai"]["feature_parameters"].update({"buffer_train_data_candles": buffer})
    traderai_conf["traderai"]["feature_parameters"].update({"noise_standard_deviation": noise})

    if "ReinforcementLearner" in model:
        model_save_ext = "zip"
        traderai_conf = make_rl_config(traderai_conf)
        # test the RL guardrails
        traderai_conf["traderai"]["feature_parameters"].update({"use_SVM_to_remove_outliers": True})
        traderai_conf["traderai"]["feature_parameters"].update({"DI_threshold": 2})
        traderai_conf["traderai"]["data_split_parameters"].update({"shuffle": True})

    if "test_3ac" in model or "test_4ac" in model:
        traderai_conf["traderaimodel_path"] = str(
            Path(__file__).parents[1] / "traderai" / "test_models"
        )
        traderai_conf["traderai"]["rl_config"]["drop_ohlc_from_features"] = True

    if "PyTorch" in model:
        model_save_ext = "zip"
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        traderai_conf["traderai"]["model_training_parameters"].update(pytorch_mlp_mtp)
        if "Transformer" in model:
            # transformer model takes a window, unlike the MLP regressor
            traderai_conf.update({"conv_width": 10})

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.activate_tensorboard = test_tb
    traderai.can_short = can_short
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    traderai.dk.live = True
    traderai.dk.set_paths("ADA/BTC", 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)

    traderai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    traderai.dk.set_paths("ADA/BTC", None)

    traderai.train_timer("start", "ADA/BTC")
    traderai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, traderai.dk, data_load_timerange
    )
    traderai.train_timer("stop", "ADA/BTC")
    traderai.dd.save_metric_tracker_to_disk()
    traderai.dd.save_drawer_to_disk()

    assert Path(traderai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(traderai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(
        traderai.dk.data_path / f"{traderai.dk.model_filename}_model.{model_save_ext}"
    ).is_file()
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_metadata.json").is_file()
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(traderai.dk.full_path))


@pytest.mark.parametrize(
    "model, strat",
    [
        ("LightGBMRegressorMultiTarget", "traderai_test_multimodel_strat"),
        ("XGBoostRegressorMultiTarget", "traderai_test_multimodel_strat"),
        ("CatboostRegressorMultiTarget", "traderai_test_multimodel_strat"),
        ("LightGBMClassifierMultiTarget", "traderai_test_multimodel_classifier_strat"),
        ("CatboostClassifierMultiTarget", "traderai_test_multimodel_classifier_strat"),
    ],
)
def test_extract_data_and_train_model_MultiTargets(mocker, traderai_conf, model, strat):
    can_run_model(model)

    traderai_conf.update({"timerange": "20180110-20180130"})
    traderai_conf.update({"strategy": strat})
    traderai_conf.update({"traderaimodel": model})
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    traderai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)

    traderai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    traderai.dk.set_paths("ADA/BTC", None)

    traderai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, traderai.dk, data_load_timerange
    )

    assert len(traderai.dk.label_list) == 2
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_model.joblib").is_file()
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_metadata.json").is_file()
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(traderai.dk.data["training_features_list"]) == 14

    shutil.rmtree(Path(traderai.dk.full_path))


@pytest.mark.parametrize(
    "model",
    [
        "LightGBMClassifier",
        "CatboostClassifier",
        "XGBoostClassifier",
        "XGBoostRFClassifier",
        "SKLearnRandomForestClassifier",
        "PyTorchMLPClassifier",
    ],
)
def test_extract_data_and_train_model_Classifiers(mocker, traderai_conf, model):
    can_run_model(model)

    traderai_conf.update({"traderaimodel": model})
    traderai_conf.update({"strategy": "traderai_test_classifier"})
    traderai_conf.update({"timerange": "20180110-20180130"})
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)

    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    traderai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)

    traderai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    traderai.dk.set_paths("ADA/BTC", None)

    traderai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, traderai.dk, data_load_timerange
    )

    if "PyTorchMLPClassifier":
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        traderai_conf["traderai"]["model_training_parameters"].update(pytorch_mlp_mtp)

    if traderai.dd.model_type == "joblib":
        model_file_extension = ".joblib"
    elif traderai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(
            f"Unsupported model type: {traderai.dd.model_type}, can't assign model_file_extension"
        )

    assert Path(
        traderai.dk.data_path / f"{traderai.dk.model_filename}_model{model_file_extension}"
    ).exists()
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_metadata.json").exists()
    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(traderai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "traderai_test_strat"),
        ("XGBoostRegressor", 2, "traderai_test_strat"),
        ("CatboostRegressor", 2, "traderai_test_strat"),
        ("PyTorchMLPRegressor", 2, "traderai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "traderai_test_strat"),
        ("ReinforcementLearner", 3, "traderai_rl_test_strat"),
        ("XGBoostClassifier", 2, "traderai_test_classifier"),
        ("LightGBMClassifier", 2, "traderai_test_classifier"),
        ("CatboostClassifier", 2, "traderai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "traderai_test_classifier"),
    ],
)
def test_start_backtesting(mocker, traderai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac() and not is_arm():
        test_tb = False

    traderai_conf.get("traderai", {}).update({"save_backtest_models": True})
    traderai_conf["runmode"] = RunMode.BACKTEST

    Trade.use_db = False

    traderai_conf.update({"traderaimodel": model})
    traderai_conf.update({"timerange": "20180120-20180130"})
    traderai_conf.update({"strategy": strat})

    if "ReinforcementLearner" in model:
        traderai_conf = make_rl_config(traderai_conf)

    if "test_4ac" in model:
        traderai_conf["traderaimodel_path"] = str(
            Path(__file__).parents[1] / "traderai" / "test_models"
        )

    if "PyTorch" in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        traderai_conf["traderai"]["model_training_parameters"].update(pytorch_mlp_mtp)
        if "Transformer" in model:
            # transformer model takes a window, unlike the MLP regressor
            traderai_conf.update({"conv_width": 10})

    traderai_conf.get("traderai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = False
    traderai.activate_tensorboard = test_tb
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = traderai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", traderai.dk)
    df = base_df[traderai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    traderai.dk.set_paths("LTC/BTC", None)
    traderai.start_backtesting(df, metadata, traderai.dk, strategy)
    model_folders = [x for x in traderai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(traderai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, traderai_conf):
    traderai_conf.update({"timerange": "20180120-20180124"})
    traderai_conf["runmode"] = "backtest"
    traderai_conf.get("traderai", {}).update(
        {
            "backtest_period_days": 0.5,
            "save_backtest_models": True,
        }
    )
    traderai_conf.get("traderai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = False
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = traderai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", traderai.dk)
    df = base_df[traderai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    traderai.start_backtesting(df, metadata, traderai.dk, strategy)
    model_folders = [x for x in traderai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(traderai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, traderai_conf, caplog):
    traderai_conf.update({"timerange": "20180120-20180130"})
    traderai_conf["runmode"] = "backtest"
    traderai_conf.get("traderai", {}).update({"save_backtest_models": True})
    traderai_conf.get("traderai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = False
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = traderai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", traderai.dk)
    df = base_df[traderai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    traderai.dk.pair = pair
    traderai.start_backtesting(df, metadata, traderai.dk, strategy)
    model_folders = [x for x in traderai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

    traderai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = False
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = traderai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", traderai.dk)
    df = base_df[traderai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    traderai.dk.pair = pair
    traderai.start_backtesting(df, metadata, traderai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    traderai.dk.pair = pair
    traderai.start_backtesting(df, metadata, traderai.dk, strategy)

    path = traderai.dd.full_path / traderai.dk.backtest_predictions_folder
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(traderai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, traderai_conf, caplog):
    traderai_conf["runmode"] = "backtest"
    traderai_conf.get("traderai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = False
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = traderai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", traderai.dk)
    df = traderai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_traderai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = traderai.dk.remove_special_chars_from_feature_names(df)
    traderai.dk.get_unique_classes_from_labels(df)
    traderai.dk.pair = "ADA/BTC"
    traderai.dk.full_df = df.fillna(0)

    assert "&-s_close_mean" not in traderai.dk.full_df.columns
    assert "&-s_close_std" not in traderai.dk.full_df.columns
    traderai.backtesting_fit_live_predictions(traderai.dk)
    assert "&-s_close_mean" in traderai.dk.full_df.columns
    assert "&-s_close_std" in traderai.dk.full_df.columns
    shutil.rmtree(Path(traderai.dk.full_path))


def test_plot_feature_importance(mocker, traderai_conf):
    from traderpilot.traderai.utils import plot_feature_importance

    traderai_conf.update({"timerange": "20180110-20180130"})
    traderai_conf.get("traderai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"}
    )

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = True
    traderai.dk = TraderaiDataKitchen(traderai_conf)
    traderai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    traderai.dd.load_all_pair_histories(timerange, traderai.dk)

    traderai.dd.pair_dict = {
        "ADA/BTC": {
            "model_filename": "fake_name",
            "trained_timestamp": 1,
            "data_path": "",
            "extras": {},
        }
    }

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    traderai.dk.set_paths("ADA/BTC", None)

    traderai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, traderai.dk, data_load_timerange
    )

    model = traderai.dd.load_data("ADA/BTC", traderai.dk)

    plot_feature_importance(model, "ADA/BTC", traderai.dk)

    assert Path(traderai.dk.data_path / f"{traderai.dk.model_filename}.html")

    shutil.rmtree(Path(traderai.dk.full_path))


@pytest.mark.parametrize(
    "timeframes,corr_pairs",
    [
        (["5m"], ["ADA/BTC", "DASH/BTC"]),
        (["5m"], ["ADA/BTC", "DASH/BTC", "ETH/USDT"]),
        (["5m", "15m"], ["ADA/BTC", "DASH/BTC", "ETH/USDT"]),
    ],
)
def test_traderai_informative_pairs(mocker, traderai_conf, timeframes, corr_pairs):
    traderai_conf["traderai"]["feature_parameters"].update(
        {
            "include_timeframes": timeframes,
            "include_corr_pairlist": corr_pairs,
        }
    )
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    pairlists = PairListManager(exchange, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)


def test_start_set_train_queue(mocker, traderai_conf, caplog):
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    pairlist = PairListManager(exchange, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange, pairlist)
    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.live = False

    traderai.train_queue = traderai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, traderai_conf):
    time_range = get_required_data_timerange(traderai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, traderai_conf, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    pairlist = PairListManager(exchange, traderai_conf)
    strategy.dp = DataProvider(traderai_conf, exchange, pairlist)
    traderai_conf["pairs"] = traderai_conf["exchange"]["pair_whitelist"]
    traderai_conf["datadir"] = tmp_path
    download_all_data_for_training(strategy.dp, traderai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("dp_exists", [(False), (True)])
def test_get_state_info(mocker, traderai_conf, dp_exists, caplog, tickers):
    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")

    traderai_conf.update({"traderaimodel": "ReinforcementLearner"})
    traderai_conf.update({"timerange": "20180110-20180130"})
    traderai_conf.update({"strategy": "traderai_rl_test_strat"})
    traderai_conf = make_rl_config(traderai_conf)
    traderai_conf["entry_pricing"]["price_side"] = "same"
    traderai_conf["exit_pricing"]["price_side"] = "same"

    strategy = get_patched_traderai_strategy(mocker, traderai_conf)
    exchange = get_patched_exchange(mocker, traderai_conf)
    ticker_mock = MagicMock(return_value=tickers()["ETH/BTC"])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(traderai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.traderai_info = traderai_conf.get("traderai", {})
    traderai = strategy.traderai
    traderai.data_provider = strategy.dp
    traderai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    traderai.get_state_info("ADA/BTC")
    traderai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )
