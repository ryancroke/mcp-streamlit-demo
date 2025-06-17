#!/usr/bin/env python3
"""
Quick start script for the MCP Comparison Framework.
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import fastapi
        import uvicorn
        import langchain_core
        import langgraph
        import mcp
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: uv sync")
        return False


def check_files():
    """Check if required files exist."""
    required_files = [
        "comparison_app.py",
        "framework/orchestrator/simple_orchestrator.py",
        "framework/orchestrator/mcp_interface.py",
        "framework/dual_chat_ui/index.html",
        "configs/sqlite/baseline_config.json",
        "configs/sqlite/enhanced_config.json",
        "data/Chinook_Sqlite.db",
        "mcp_servers/mcp_sqlite_baseline/pyproject.toml",
        "mcp_servers/mcp_sqlite_enhanced/pyproject.toml",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True


def main():
    """Main startup function."""
    print("🚀 MCP Comparison Framework Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    print("\n🎵 Starting MCP Comparison Framework...")
    print("📍 Will open at: http://localhost:8001")
    print("🔄 Starting FastAPI server...")
    
    try:
        subprocess.run([
            sys.executable, 
            "comparison_app.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()