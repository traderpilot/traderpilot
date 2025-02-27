import logging
from abc import ABC, abstractmethod

import torch

from traderpilot.traderai.torch.PyTorchDataConvertor import PyTorchDataConvertor
from traderpilot.traderai.traderai_interface import ITraderaiModel


logger = logging.getLogger(__name__)


class BasePyTorchModel(ITraderaiModel, ABC):
    """
    Base class for PyTorch type models.
    User *must* inherit from this class and set fit() and predict() and
    data_convertor property.
    """

    def __init__(self, **kwargs):
        super().__init__(config=kwargs["config"])
        self.dd.model_type = "pytorch"
        self.device = (
            "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        test_size = self.traderai_info.get("data_split_parameters", {}).get("test_size")
        self.splits = ["train", "test"] if test_size != 0 else ["train"]
        self.window_size = self.traderai_info.get("conv_width", 1)

    @property
    @abstractmethod
    def data_convertor(self) -> PyTorchDataConvertor:
        """
        a class responsible for converting `*_features` & `*_labels` pandas dataframes
        to pytorch tensors.
        """
        raise NotImplementedError("Abstract property")
