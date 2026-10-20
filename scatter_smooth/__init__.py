from importlib.metadata import version as _version, PackageNotFoundError

from .splines import SplineSmoother
from .loess import LoessSmoother
from .datasets import load_bikeshare

try:
    __version__ = _version('scatter_smooth')
except PackageNotFoundError:          # not installed
    __version__ = '0+unknown'
