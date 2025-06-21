"""
Generic MCP Server Handler.

Fallback handler for MCP servers that don't match any specific type.
Provides minimal functionality without making assumptions about the project structure.
"""

from pathlib import Path
from typing import Any

from .base import MCPServerHandler


class GenericHandler(MCPServerHandler):
    """Generic fallback handler for unknown MCP server types."""

    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        """
        Always returns True as this is the fallback handler.

        Args:
            work_dir: Directory to check
            config: MCP server configuration

        Returns:
            Always True (fallback handler)
        """
        return True

    def install_dependencies(self, work_dir: Path) -> None:
        """
        No-op for generic handler. Assumes MCP server is ready to run.

        Args:
            work_dir: Directory containing the MCP server
        """
        print(f"⚠️  Using generic handler - no dependency installation for {work_dir}")
        print("   Assuming MCP server is ready to run as-is")

    def transform_command(
        self, config: dict[str, Any], work_dir: Path
    ) -> dict[str, Any]:
        """
        Return configuration unchanged for generic handler.

        Args:
            config: Original MCP server configuration
            work_dir: Directory containing the MCP server

        Returns:
            Unchanged configuration
        """
        print(
            f"⚠️  Using generic handler - no command transformation for {config.get('name', 'unknown')}"
        )
        return config.copy()

    def get_health_check_query(self) -> str:
        """
        Return a generic health check query.

        Returns:
            Query string for health checks
        """
        return "What tools are available?"

    def get_priority(self) -> int:
        """
        Lowest priority - only used as fallback.

        Returns:
            High priority value (lowest priority)
        """
        return 1000
