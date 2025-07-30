#!/usr/bin/env python3
"""
Vizor Plugin System
Modular plugin architecture for extensible cybersecurity capabilities
"""

from .base import VizorPlugin
from .plugin_registry import PluginRegistry
from .wrapper_generator import WrapperGenerator

__all__ = ['VizorPlugin', 'PluginRegistry', 'WrapperGenerator'] 