# TraderUI

Traderpilot provides a builtin webserver, which can serve [TraderUI](https://github.com/traderpilot/traderui), the traderpilot frontend.

By default, the UI is automatically installed as part of the installation (script, docker).
traderUI can also be manually installed by using the `traderpilot install-ui` command.
This same command can also be used to update traderUI to new new releases.

Once the bot is started in trade / dry-run mode (with `traderpilot trade`) - the UI will be available under the configured API port (by default `http://127.0.0.1:8080`).

??? Note "Looking to contribute to traderUI?"
    Developers should not use this method, but instead clone the corresponding use the method described in the [traderUI repository](https://github.com/traderpilot/traderui) to get the source-code of traderUI. A working installation of node will be required to build the frontend.

!!! tip "traderUI is not required to run traderpilot"
    traderUI is an optional component of traderpilot, and is not required to run the bot.
    It is a frontend that can be used to monitor the bot and to interact with it - but traderpilot itself will work perfectly fine without it.

## Configuration

TraderUI does not have it's own configuration file - but assumes a working setup for the [rest-api](rest-api.md) is available.
Please refer to the corresponding documentation page to get setup with traderUI

## UI

TraderUI is a modern, responsive web application that can be used to monitor and interact with your bot.

TraderUI provides a light, as well as a dark theme.
Themes can be easily switched via a prominent button at the top of the page.
The theme of the screenshots on this page will adapt to the selected documentation Theme, so to see the dark (or light) version, please switch the theme of the Documentation.

### Login

The below screenshot shows the login screen of traderUI.

![TraderUI - login](assets/traderui-login-CORS.png#only-dark)
![TraderUI - login](assets/traderui-login-CORS-light.png#only-light)

!!! Hint "CORS"
    The Cors error shown in this screenshot is due to the fact that the UI is running on a different port than the API, and [CORS](#cors) has not been setup correctly yet.

### Trade view

The trade view allows you to visualize the trades that the bot is making and to interact with the bot.
On this page, you can also interact with the bot by starting and stopping it and - if configured - force trade entries and exits.

![TraderUI - trade view](assets/traderUI-trade-pane-dark.png#only-dark)
![TraderUI - trade view](assets/traderUI-trade-pane-light.png#only-light)

### Plot Configurator

TraderUI Plots can be configured either via a `plot_config` configuration object in the strategy (which can be loaded via "from strategy" button) or via the UI.
Multiple plot configurations can be created and switched at will - allowing for flexible, different views into your charts.

The plot configuration can be accessed via the "Plot Configurator" (Cog icon) button in the top right corner of the trade view.

![TraderUI - plot configuration](assets/traderUI-plot-configurator-dark.png#only-dark)
![TraderUI - plot configuration](assets/traderUI-plot-configurator-light.png#only-light)

### Settings

Several UI related settings can be changed by accessing the settings page.

Things you can change (among others):

* Timezone of the UI
* Visualization of open trades as part of the favicon (browser tab)
* Candle colors (up/down -> red/green)
* Enable / disable in-app notification types

![TraderUI - Settings view](assets/traderui-settings-dark.png#only-dark)
![TraderUI - Settings view](assets/traderui-settings-light.png#only-light)

## Backtesting

When traderpilot is started in [webserver mode](utils.md#webserver-mode) (traderpilot started with `traderpilot webserver`), the backtesting view becomes available.
This view allows you to backtest strategies and visualize the results.

You can also load and visualize previous backtest results, as well as compare the results with each other.

![TraderUI - Backtesting](assets/traderUI-backtesting-dark.png#only-dark)
![TraderUI - Backtesting](assets/traderUI-backtesting-light.png#only-light)


--8<-- "includes/cors.md"
