# MCP Comparison Framework

A reusable, configuration-driven framework for comparing baseline vs enhanced MCP (Model Context Protocol) servers side-by-side. **No local MCP servers required** - automatically clones and runs servers directly from GitHub repositories.

## Features

- **✅ GitHub Integration**: Automatically clones MCP servers from any GitHub repository
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

**Phase 2 ✅ COMPLETED**: GitHub Integration for dynamic MCP sourcing
- Automatic cloning from GitHub repositories
- Dynamic dependency installation
- No local MCP servers required
- Intelligent file watching (disabled for GitHub sources)

**Phase 3 📋 NEXT**: Multi-comparison UI switching  
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
   
   The framework will automatically:
   - Clone SQLite MCP servers from GitHub (`modelcontextprotocol/servers-archived`)
   - Install dependencies in temporary directories
   - Start both baseline and enhanced versions
   - Disable file watching to prevent reload loops

3. **Open Browser**:
   Navigate to http://localhost:8001

## Configuration

The framework uses GitHub-sourced MCP servers by default. All configurations now support GitHub sources:

```bash
# Uses GitHub sources by default (modelcontextprotocol/servers-archived)
uv run comparison_app.py

# Use custom GitHub configuration
MCP_COMPARISON_CONFIG=configs/sqlite-github/comparison_config.json uv run comparison_app.py

# Example: Test different branches or repositories
MCP_COMPARISON_CONFIG=configs/your_custom/comparison_config.json uv run comparison_app.py
```

**No local MCP servers needed!** The framework automatically handles cloning, dependency installation, and cleanup.

## Directory Structure

```
mcp-enhanced-compare/
├── framework/              # Reusable comparison framework
│   ├── dual_chat_ui/      # Side-by-side chat interface
│   └── orchestrator/      # GitHub-enabled orchestrator
├── configs/               # MCP-specific configurations
│   ├── sqlite/           # SQLite MCP comparison config (GitHub sources)
│   ├── sqlite-github/    # Alternative GitHub configuration
│   └── template/         # Template for new MCP comparisons
├── temp/                 # Auto-generated (git-ignored)
│   ├── mcp_baseline_from_git/   # Cloned baseline MCP server
│   └── mcp_enhanced_from_git/   # Cloned enhanced MCP server
├── data/                 # Database files only
│   └── Chinook_Sqlite.db
└── experiments/          # Results and test data
```

## Configuration Format

Each MCP comparison uses a single unified configuration file with GitHub source support:

```json
{
  "comparison_name": "SQLite MCP Comparison",
  "comparison_description": "Comparing SQLite MCP servers sourced from GitHub",
  "data_files": ["data/Chinook_Sqlite.db"],
  "baseline": {
    "source": {
      "type": "github",
      "repo": "modelcontextprotocol/servers-archived",
      "branch": "main",
      "subdirectory": "src/sqlite",
      "install_dir": "temp/mcp_baseline_from_git"
    },
    "mcp_server": { /* MCP server config */ },
    "ui": { "title": "Baseline (GitHub)", "color": "#4A90E2", "icon": "🔵" }
  },
  "enhanced": {
    "source": { /* Same GitHub source structure */ },
    "mcp_server": { /* MCP server config */ },
    "ui": { "title": "Enhanced (GitHub)", "color": "#E25A4A", "icon": "🔴" }
  }
}
```

See `configs/sqlite/comparison_config.json` for the complete GitHub-enabled example.

## Current Implementation

- **Target MCP**: SQLite MCP Server (auto-cloned from GitHub)
- **Source**: `modelcontextprotocol/servers-archived` repository
- **Baseline**: Official SQLite MCP implementation (from GitHub)
- **Enhanced**: Same implementation (demonstrates GitHub workflow)

## Usage

1. Type queries in either the baseline (left) or enhanced (right) chat panel
2. Each panel maintains its own conversation thread
3. Compare responses, SQL queries, and performance
4. Use example queries to test common scenarios

## Adding New MCP Comparisons

1. **Copy the template**: `cp -r configs/template configs/your_mcp`
2. **Edit the configuration**: Update `configs/your_mcp/comparison_config.json` with your:
   - GitHub repository details
   - MCP server command and arguments  
   - UI customization (titles, colors, icons)
3. **Run with your config**: 
   ```bash
   MCP_COMPARISON_CONFIG=configs/your_mcp/comparison_config.json uv run comparison_app.py
   ```

The framework automatically clones, installs, and runs your MCP servers - no local setup needed!

## Commands

- **Run**: `uv run comparison_app.py`
- **Format**: `ruff format .`
- **Lint**: `ruff check .`
- **Type Check**: `mypy .`
- **Fix Lint**: `ruff check --fix .`

## Architecture

See `specs/ARCHITECTURE_PLAN.md` for detailed implementation phases and future roadmap.