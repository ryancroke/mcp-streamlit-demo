"""
MCP Server Handler Registry.

Manages the collection of MCP server handlers and provides auto-detection
of the appropriate handler for each MCP server type.
"""

from pathlib import Path
from typing import Any

from .base import MCPServerHandler
from .generic import GenericHandler
from .nodejs import NodeJSHandler
from .python_uv import PythonUVHandler


class MCPServerHandlerRegistry:
    """Registry for managing MCP server handlers."""

    def __init__(self):
        """Initialize the registry with default handlers."""
        self.handlers: list[MCPServerHandler] = [
            PythonUVHandler(),
            NodeJSHandler(),
            GenericHandler(),  # Always last as fallback
        ]

        # Sort handlers by priority (lower numbers = higher priority)
        self.handlers.sort(key=lambda h: h.get_priority())

    def register_handler(self, handler: MCPServerHandler) -> None:
        """
        Register a new handler with the registry.

        Args:
            handler: Handler instance to register
        """
        self.handlers.append(handler)
        # Re-sort to maintain priority order
        self.handlers.sort(key=lambda h: h.get_priority())

    def get_handler(self, work_dir: Path, config: dict[str, Any]) -> MCPServerHandler:
        """
        Find the appropriate handler for the given MCP server.

        Args:
            work_dir: Directory containing the MCP server source code
            config: MCP server configuration dictionary

        Returns:
            Handler instance that can manage this MCP server

        Raises:
            RuntimeError: If no handler can be found (should never happen with GenericHandler)
        """
        for handler in self.handlers:
            if handler.can_handle(work_dir, config):
                handler_name = handler.__class__.__name__
                print(f"🔧 Selected {handler_name} for MCP server at {work_dir}")
                return handler

        # This should never happen since GenericHandler always returns True
        raise RuntimeError(f"No handler found for MCP server at {work_dir}")

    def list_handlers(self) -> list[str]:
        """
        Get a list of registered handler names.

        Returns:
            List of handler class names
        """
        return [handler.__class__.__name__ for handler in self.handlers]
