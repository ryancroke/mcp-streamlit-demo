"""
MCP Server Handler Framework.

This module provides a pluggable architecture for handling different types of MCP servers,
eliminating the need for type-specific conditionals in the main orchestrator code.
"""

from .base import MCPServerHandler
from .generic import GenericHandler
from .nodejs import NodeJSHandler
from .python_uv import PythonUVHandler
from .registry import MCPServerHandlerRegistry

__all__ = [
    "GenericHandler",
    "MCPServerHandler",
    "MCPServerHandlerRegistry",
    "NodeJSHandler",
    "PythonUVHandler",
]
