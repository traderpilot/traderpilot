"""
Module to handle data operations for traderpilot
"""

from traderpilot.data import converter


# limit what's imported when using `from traderpilot.data import *`
__all__ = ["converter"]
