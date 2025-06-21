"""
Base MCP Server Handler interface.

Defines the contract that all MCP server handlers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MCPServerHandler(ABC):
    """Abstract base class for MCP server handlers."""

    @abstractmethod
    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        """
        Determine if this handler can manage the MCP server.

        Args:
            work_dir: Directory containing the MCP server source code
            config: MCP server configuration dictionary

        Returns:
            True if this handler can manage the MCP server, False otherwise
        """

    @abstractmethod
    def install_dependencies(self, work_dir: Path) -> None:
        """
        Install dependencies and build the MCP server.

        Args:
            work_dir: Directory containing the MCP server source code

        Raises:
            RuntimeError: If installation or build fails
        """

    @abstractmethod
    def transform_command(
        self, config: dict[str, Any], work_dir: Path
    ) -> dict[str, Any]:
        """
        Transform the MCP server configuration for local execution.

        Args:
            config: Original MCP server configuration
            work_dir: Directory containing the built MCP server

        Returns:
            Updated configuration ready for local execution
        """

    @abstractmethod
    def get_health_check_query(self) -> str:
        """
        Return an appropriate health check query for this MCP server type.

        Returns:
            Query string to use for health checks
        """

    def get_priority(self) -> int:
        """
        Return the priority of this handler (lower numbers = higher priority).

        Used by the registry to determine handler order when multiple handlers
        could potentially handle the same MCP server.

        Returns:
            Priority value (default: 100)
        """
        return 100
