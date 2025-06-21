"""
Python UV MCP Server Handler.

Handles Python-based MCP servers that use uv for dependency management.
"""

import subprocess
from pathlib import Path
from typing import Any

from .base import MCPServerHandler


class PythonUVHandler(MCPServerHandler):
    """Handler for Python MCP servers using uv dependency management."""

    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        """
        Detect Python MCP servers by presence of pyproject.toml.

        Args:
            work_dir: Directory to check
            config: MCP server configuration

        Returns:
            True if this is a Python uv project
        """
        return (work_dir / "pyproject.toml").exists()

    def install_dependencies(self, work_dir: Path) -> None:
        """
        Install dependencies using uv sync.

        Args:
            work_dir: Directory containing pyproject.toml

        Raises:
            RuntimeError: If uv sync fails
        """
        try:
            print(f"📦 Installing Python dependencies with uv in {work_dir}")
            subprocess.run(
                ["uv", "sync"], cwd=work_dir, check=True, capture_output=True, text=True
            )
            print("✅ Python dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install Python dependencies: {e.stderr if e.stderr else str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def transform_command(
        self, config: dict[str, Any], work_dir: Path
    ) -> dict[str, Any]:
        """
        Transform command to use uv with --directory flag.

        Args:
            config: Original MCP server configuration
            work_dir: Directory containing the built MCP server

        Returns:
            Updated configuration with --directory flag
        """
        updated_config = config.copy()

        # Insert --directory flag at the beginning of args for uv-based commands
        if updated_config.get("command") == "uv":
            new_args = ["--directory", str(work_dir)]
            new_args.extend(updated_config.get("args", []))
            updated_config["args"] = new_args

        # Resolve any relative data file paths to absolute paths
        updated_config = self._resolve_data_paths(updated_config, work_dir)

        return updated_config

    def get_health_check_query(self) -> str:
        """
        Return a generic health check query.

        Returns:
            Query string for health checks
        """
        return "What tools are available?"

    def get_priority(self) -> int:
        """
        High priority for Python projects.

        Returns:
            Priority value (lower = higher priority)
        """
        return 10

    def _resolve_data_paths(
        self, config: dict[str, Any], work_dir: Path
    ) -> dict[str, Any]:
        """
        Convert relative data file paths to absolute paths from project root.

        Args:
            config: MCP server configuration
            work_dir: MCP server working directory

        Returns:
            Updated configuration with absolute data paths
        """
        updated_config = config.copy()
        updated_args = []

        project_root = Path.cwd()

        for arg in updated_config.get("args", []):
            if isinstance(arg, str) and arg.startswith("data/"):
                # Convert relative data path to absolute path
                absolute_path = project_root / arg
                updated_args.append(str(absolute_path))
            else:
                updated_args.append(arg)

        updated_config["args"] = updated_args
        return updated_config
