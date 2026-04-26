"""RLDS format support for Robodata.

RLDS (Reinforcement Learning Datasets) is a TensorFlow-based format
commonly used for Open-X datasets.
"""

from robodata.formats.rlds.reader import RLDSReader
from robodata.formats.rlds.writer import RLDSWriter

__all__ = ["RLDSReader", "RLDSWriter"]
