"""RoboDM format support for Robodata.

RoboDM is Berkeley's high-performance robotics data format using .vla files.
It provides efficient video compression (up to 70x) with codec flexibility.

See: https://github.com/BerkeleyAutomation/robodm
"""

from robodata.formats.robodm.reader import RoboDMReader
from robodata.formats.robodm.writer import RoboDMWriter

__all__ = ["RoboDMReader", "RoboDMWriter"]
