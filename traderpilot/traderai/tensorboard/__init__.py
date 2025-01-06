# ensure users can still use a non-torch traderai version
try:
    from traderpilot.traderai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger

    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from traderpilot.traderai.tensorboard.base_tensorboard import (
        BaseTensorBoardCallback,
        BaseTensorboardLogger,
    )

    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = ("TBLogger", "TBCallback")
