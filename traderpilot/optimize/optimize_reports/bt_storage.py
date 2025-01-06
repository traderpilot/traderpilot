import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from pandas import DataFrame

from traderpilot.constants import LAST_BT_RESULT_FN
from traderpilot.enums.runmode import RunMode
from traderpilot.types import BacktestResultType
from traderpilot.misc import dump_json_to_file, file_dump_json
from traderpilot.optimize.backtest_caching import get_backtest_metadata_filename


logger = logging.getLogger(__name__)


def file_dump_joblib(file_obj: BytesIO, data: Any, log: bool = True) -> None:
    """
    Dump object data into a file
    :param filename: file to create
    :param data: Object data to save
    :return:
    """
    import joblib

    joblib.dump(data, file_obj)


def _generate_filename(recordfilename: Path, appendix: str, suffix: str) -> Path:
    """
    Generates a filename based on the provided parameters.
    :param recordfilename: Path object, which can either be a filename or a directory.
    :param appendix: use for the filename. e.g. backtest-result-<datetime>
    :param suffix: Suffix to use for the file, e.g. .json, .pkl
    :return: Generated filename as a Path object
    """
    if recordfilename.is_dir():
        filename = (recordfilename / f"backtest-result-{appendix}").with_suffix(suffix)
    else:
        filename = Path.joinpath(
            recordfilename.parent, f"{recordfilename.stem}-{appendix}"
        ).with_suffix(suffix)
    return filename


def store_backtest_results(
    config: dict,
    stats: BacktestResultType,
    dtappendix: str,
    *,
    market_change_data: DataFrame | None = None,
    analysis_results: dict[str, dict[str, DataFrame]] | None = None,
) -> Path:
    """
    Stores backtest results and analysis data in a zip file, with metadata stored separately
    for convenience.
    :param config: Configuration dictionary
    :param stats: Dataframe containing the backtesting statistics
    :param dtappendix: Datetime to use for the filename
    :param market_change_data: Dataframe containing market change data
    :param analysis_results: Dictionary containing analysis results
    """
    recordfilename: Path = config["exportfilename"]
    zip_filename = _generate_filename(recordfilename, dtappendix, ".zip")
    base_filename = _generate_filename(recordfilename, dtappendix, "")
    json_filename = _generate_filename(recordfilename, dtappendix, ".json")

    # Store metadata separately with .json extension
    file_dump_json(get_backtest_metadata_filename(json_filename), stats["metadata"])

    # Store latest backtest info separately
    latest_filename = Path.joinpath(zip_filename.parent, LAST_BT_RESULT_FN)
    file_dump_json(latest_filename, {"latest_backtest": str(zip_filename.name)}, log=False)

    # Create zip file and add the files
    with ZipFile(zip_filename, "w", ZIP_DEFLATED) as zipf:
        # Store stats
        stats_copy = {
            "strategy": stats["strategy"],
            "strategy_comparison": stats["strategy_comparison"],
        }
        stats_buf = StringIO()
        dump_json_to_file(stats_buf, stats_copy)
        zipf.writestr(json_filename.name, stats_buf.getvalue())

        # Add market change data if present
        if market_change_data is not None:
            market_change_name = f"{base_filename.stem}_market_change.feather"
            market_change_buf = BytesIO()
            market_change_data.reset_index().to_feather(
                market_change_buf, compression_level=9, compression="lz4"
            )
            market_change_buf.seek(0)
            zipf.writestr(market_change_name, market_change_buf.getvalue())

        # Add analysis results if present and running in backtest mode
        if (
            config.get("export", "none") == "signals"
            and analysis_results is not None
            and config.get("runmode", RunMode.OTHER) == RunMode.BACKTEST
        ):
            for name in ["signals", "rejected", "exited"]:
                if name in analysis_results:
                    analysis_name = f"{base_filename.stem}_{name}.pkl"
                    analysis_buf = BytesIO()
                    file_dump_joblib(analysis_buf, analysis_results[name])
                    analysis_buf.seek(0)
                    zipf.writestr(analysis_name, analysis_buf.getvalue())

    return zip_filename
