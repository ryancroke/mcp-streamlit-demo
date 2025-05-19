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

Create a `.env` file in the root directory with the following variables:
```bash
GITHUB_PAT=your_github_personal_access_token
BRAVE_API_KEY=your_brave_api_key
```
# Installing MCP Servers

## GitHub MCP Server
Goto Ensure you have the GitHub MCP server binary in the `github-mcp-server` directory

https://github.com/github/github-mcp-server

```bash
git clone https://github.com/github/github-mcp-server
```


