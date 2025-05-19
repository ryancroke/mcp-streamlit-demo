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

You can install with Docker or using GO locally
```
If you don't have Docker, you can use go build to build the binary in the cmd/github-mcp-server directory, and use the github-mcp-server stdio command with the GITHUB_PERSONAL_ACCESS_TOKEN environment variable set to your token. 
```

Just point to the executable in the config and add your GITHUB_PERSONAL_TOKEN

# Run Streamlit app

on a mac if you are on a Mac M* chip you need to use Rosetta 2
```
arch -x86_64 streamlit run simple_stremlit_MCP.py
``

otherwise 
```bash 
streamlit run simple_app.py
``` 
# ToDo
1. Abstract out servers. Starting them every time is running into a problem
1. There is information bleed in the app. Hunt this down.
1. Find other improvements. 
1. Add in sidebar to tell you whcih state you are in. 
1. Open telemetry for Burr and get that together. 