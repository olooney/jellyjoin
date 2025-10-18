import importlib.metadata as _imd

from .join import *
from .similarity import *
from .strategy import *

# set the version dynamically
try:
    __version__ = _imd.version("jellyjoin")
except _imd.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

del _imd
