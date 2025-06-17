# MCP Comparison Framework

A reusable framework for comparing baseline vs enhanced MCP (Model Context Protocol) servers side-by-side.

## Features

- **Dual-Chat Interface**: Separate conversation threads for baseline and enhanced versions
- **Real-time Comparison**: See how different MCP implementations handle the same queries
- **Easy Configuration**: Swap different MCP servers through configuration files
- **Evaluation Support**: Built for systematic testing and comparison
- **Clean Architecture**: Simplified orchestrator without complex A2A agents

## Quick Start

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run the Application**:
   ```bash
   python comparison_app.py
   ```

3. **Open Browser**:
   Navigate to http://localhost:8001

## Directory Structure

```
mcp-enhanced-compare/
├── framework/              # Reusable comparison framework
│   ├── dual_chat_ui/      # Side-by-side chat interface
│   └── orchestrator/      # Simplified orchestrator
├── configs/               # MCP-specific configurations
│   ├── sqlite/           # SQLite MCP comparison config
│   └── template/         # Template for new MCP comparisons
├── mcp_servers/          # MCP server implementations
│   ├── mcp_sqlite_baseline/   # Official SQLite MCP
│   └── mcp_sqlite_enhanced/   # Enhanced SQLite MCP
├── data/                 # Database files only
│   └── Chinook_Sqlite.db
└── experiments/          # Results and test data
```

## Configuration

Each MCP comparison requires two configuration files:
- `baseline_config.json`: Configuration for the baseline MCP server
- `enhanced_config.json`: Configuration for the enhanced MCP server

See `configs/sqlite/` for examples.

## Current Implementation

- **Target MCP**: SQLite MCP Server
- **Baseline**: Official SQLite MCP implementation
- **Enhanced**: Custom enhanced version (initially identical to baseline)

## Usage

1. Type queries in either the baseline (left) or enhanced (right) chat panel
2. Each panel maintains its own conversation thread
3. Compare responses, SQL queries, and performance
4. Use example queries to test common scenarios

## Adding New MCP Comparisons

1. Create new config directory: `configs/your_mcp/`
2. Add baseline and enhanced configuration files
3. Update the comparison app to load your configs
4. Copy or create your MCP server implementations

## Commands

- **Run**: `python comparison_app.py`
- **Format**: `ruff format .`
- **Lint**: `ruff check .`
- **Type Check**: `mypy .`