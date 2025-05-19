# MCP-streamlit-demo

Brief description of what your project does.

## Prerequisites

- Python 3.11+ (or your specific version requirement)
- [uv](https://github.com/astral-sh/uv) package manager

## Getting Started

### Install uv (if not already installed)

```bash
# macOS/Linux
curl -sSf https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/astral-sh/uv/main/install.ps1 | iex


# Clone the repository
git clone https://github.com/ryancroke/mcp-streamlit-demo
cd your-repo

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies (uses pyproject.toml and uv.lock)
uv pip install .


