# flake8: noqa: F401
"""
Commands module.
Contains all start-commands, subcommands and CLI Interface creation.

Note: Be careful with file-scoped imports in these subfiles.
    as they are parsed on startup, nothing containing optional modules should be loaded.
"""

from traderpilot.commands.analyze_commands import start_analysis_entries_exits
from traderpilot.commands.arguments import Arguments
from traderpilot.commands.build_config_commands import start_new_config, start_show_config
from traderpilot.commands.data_commands import (
    start_convert_data,
    start_convert_trades,
    start_download_data,
    start_list_data,
    start_list_trades_data,
)
from traderpilot.commands.db_commands import start_convert_db
from traderpilot.commands.deploy_commands import (
    start_create_userdir,
    start_install_ui,
    start_new_strategy,
)
from traderpilot.commands.hyperopt_commands import start_hyperopt_list, start_hyperopt_show
from traderpilot.commands.list_commands import (
    start_list_exchanges,
    start_list_hyperopt_loss_functions,
    start_list_markets,
    start_list_strategies,
    start_list_timeframes,
    start_list_traderAI_models,
    start_show_trades,
)
from traderpilot.commands.optimize_commands import (
    start_backtesting,
    start_backtesting_show,
    start_edge,
    start_hyperopt,
    start_lookahead_analysis,
    start_recursive_analysis,
)
from traderpilot.commands.pairlist_commands import start_test_pairlist
from traderpilot.commands.plot_commands import start_plot_dataframe, start_plot_profit
from traderpilot.commands.strategy_utils_commands import start_strategy_update
from traderpilot.commands.trade_commands import start_trading
from traderpilot.commands.webserver_commands import start_webserver
