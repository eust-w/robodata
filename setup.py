#!/usr/bin/env python
"""Setup script for backwards compatibility with pip install -e ."""

from setuptools import setup

# All configuration is in pyproject.toml
# This file exists for editable installs with older pip versions
setup()
