# MCP Comparison Framework

A reusable, configuration-driven framework for comparing baseline vs enhanced MCP (Model Context Protocol) servers side-by-side.

## Features

- **✅ Configuration-Driven**: Fully configurable through JSON files - no hardcoded paths or UI text
- **✅ Dynamic UI**: Interface adapts automatically to configuration (titles, colors, icons)
- **✅ Dual-Chat Interface**: Separate conversation threads for baseline and enhanced versions
- **✅ Real-time Comparison**: See how different MCP implementations handle the same queries
- **✅ Environment Variables**: Flexible deployment with `MCP_COMPARISON_CONFIG` support
- **✅ Evaluation Support**: Built for systematic testing and comparison
- **✅ Clean Architecture**: Simplified orchestrator without complex A2A agents

## 🚀 Current Status

**Phase 1 ✅ COMPLETED**: Unified Configuration System
- Single configuration file per comparison type
- Dynamic UI population from API
- Environment variable support
- Backward compatibility maintained

**Phase 2 🔄 NEXT**: GitHub Integration for dynamic MCP sourcing
**Phase 3 📋 PLANNED**: Multi-comparison UI switching  
**Phase 4 📋 PLANNED**: Sequential Thinking MCP example

## Quick Start

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run the Application**:
   ```bash
   uv run comparison_app.py
   ```

3. **Open Browser**:
   Navigate to http://localhost:8001

## Configuration

The framework now uses a unified configuration system. Set the comparison type via environment variable:

```bash
# Use default SQLite comparison
uv run comparison_app.py

# Use custom configuration
MCP_COMPARISON_CONFIG=configs/your_comparison/comparison_config.json uv run comparison_app.py
```

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

## Configuration Format

Each MCP comparison uses a single unified configuration file with this structure:

```json
{
  "comparison_name": "SQLite MCP Comparison",
  "comparison_description": "Comparing baseline vs enhanced versions",
  "data_files": ["data/Chinook_Sqlite.db"],
  "baseline": {
    "mcp_server": { /* MCP server config */ },
    "ui": { "title": "Baseline", "color": "#4A90E2", "icon": "🔵" }
  },
  "enhanced": {
    "mcp_server": { /* MCP server config */ },
    "ui": { "title": "Enhanced", "color": "#E25A4A", "icon": "🔴" }
  }
}
```

See `configs/sqlite/comparison_config.json` for a complete example.

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
2. Create `comparison_config.json` with unified configuration
3. Set `MCP_COMPARISON_CONFIG` environment variable to your config path
4. Copy or create your MCP server implementations

The framework automatically adapts to your configuration - no code changes needed!

## Commands

- **Run**: `uv run comparison_app.py`
- **Format**: `ruff format .`
- **Lint**: `ruff check .`
- **Type Check**: `mypy .`
- **Fix Lint**: `ruff check --fix .`

## Architecture

See `specs/ARCHITECTURE_PLAN.md` for detailed implementation phases and future roadmap.