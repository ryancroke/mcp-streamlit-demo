"""
Node.js MCP Server Handler.

Handles Node.js/TypeScript-based MCP servers that use npm for dependency management.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

from .base import MCPServerHandler


class NodeJSHandler(MCPServerHandler):
    """Handler for Node.js/TypeScript MCP servers using npm."""

    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        """
        Detect Node.js MCP servers by presence of package.json.

        Args:
            work_dir: Directory to check
            config: MCP server configuration

        Returns:
            True if this is a Node.js project
        """
        return (work_dir / "package.json").exists()

    def install_dependencies(self, work_dir: Path) -> None:
        """
        Install dependencies using npm and build if needed.

        Args:
            work_dir: Directory containing package.json

        Raises:
            RuntimeError: If npm install or build fails
        """
        try:
            print(f"📦 Installing Node.js dependencies with npm in {work_dir}")
            subprocess.run(
                ["npm", "install"],
                cwd=work_dir,
                check=True,
                capture_output=True,
                text=True,
            )

            # Check if build script exists and run it
            package_json_path = work_dir / "package.json"
            with open(package_json_path) as f:
                package_data = json.load(f)

            if "scripts" in package_data and "build" in package_data["scripts"]:
                print(f"🔨 Building Node.js project in {work_dir}")
                subprocess.run(
                    ["npm", "run", "build"],
                    cwd=work_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✅ Node.js project built successfully")
            else:
                print("✅ Node.js dependencies installed (no build step needed)")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to install/build Node.js project: {e.stderr if e.stderr else str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def transform_command(
        self, config: dict[str, Any], work_dir: Path
    ) -> dict[str, Any]:
        """
        Transform npx command to local node execution.

        Args:
            config: Original MCP server configuration (with npx command)
            work_dir: Directory containing the built MCP server

        Returns:
            Updated configuration using node to run local build
        """
        updated_config = config.copy()

        # Convert npx-based command to local node execution
        if updated_config.get("command") == "npx":
            updated_config["command"] = "node"

            # Find the entry point from package.json
            entry_point = self._find_entry_point(work_dir)
            updated_config["args"] = [str(work_dir / entry_point)]

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
        High priority for Node.js projects.

        Returns:
            Priority value (lower = higher priority)
        """
        return 10

    def _find_entry_point(self, work_dir: Path) -> str:
        """
        Find the main entry point for the Node.js MCP server.

        Args:
            work_dir: Directory containing package.json

        Returns:
            Relative path to the entry point file
        """
        package_json_path = work_dir / "package.json"

        if package_json_path.exists():
            with open(package_json_path) as f:
                package_data = json.load(f)

            # Check for bin entry (most common for MCP servers)
            if "bin" in package_data:
                # Get the first (or only) binary entry
                bin_name = list(package_data["bin"].keys())[0]
                entry_point = package_data["bin"][bin_name]
                return entry_point

            # Fallback to main field
            if "main" in package_data:
                return package_data["main"]

        # Default fallback for TypeScript projects
        return "dist/index.js"
