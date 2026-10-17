"""Weft: graph-native memory and coordination for multi-agent systems."""

from weft.config import Settings, get_settings
from weft.service import Weft, WeftHandle

__all__ = ["Settings", "Weft", "WeftHandle", "__version__", "get_settings"]
__version__ = "0.1.0"
