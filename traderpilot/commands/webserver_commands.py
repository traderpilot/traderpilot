from typing import Any

from traderpilot.enums import RunMode


def start_webserver(args: dict[str, Any]) -> None:
    """
    Main entry point for webserver mode
    """
    from traderpilot.configuration import setup_utils_configuration
    from traderpilot.rpc.api_server import ApiServer

    # Initialize configuration

    config = setup_utils_configuration(args, RunMode.WEBSERVER)
    ApiServer(config, standalone=True)
