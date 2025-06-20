# MCP Comparison Framework

A reusable, configuration-driven framework for comparing baseline vs enhanced MCP (Model Context Protocol) servers side-by-side. **No local MCP servers required** - automatically clones and runs servers directly from GitHub repositories.

## Features

- **✅ Multi-Comparison Switching**: Real-time switching between different MCP types via web UI dropdown
- **✅ GitHub Integration**: Automatically clones MCP servers from any GitHub repository
- **✅ Configuration-Driven**: Fully configurable through JSON files - no hardcoded paths or UI text
- **✅ Dynamic UI**: Interface adapts automatically to configuration (titles, colors, icons)
- **✅ Dual-Chat Interface**: Separate conversation threads for baseline and enhanced versions
- **✅ Real-time Comparison**: See how different MCP implementations handle the same queries
- **✅ Environment Variables**: Flexible deployment with `MCP_COMPARISON_CONFIG` support
- **✅ Evaluation Support**: Built for systematic testing and comparison
- **✅ Clean Architecture**: ComparisonManager handles multiple MCP configurations seamlessly

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

**Phase 3 ✅ COMPLETED**: Multi-comparison UI switching
- Real-time switching between different MCP types via dropdown selector
- ComparisonManager for handling multiple comparison configurations
- Automatic discovery and loading of comparison configurations
- Conversation isolation between different comparison types
- Template directory exclusion for clean configuration management

**Phase 4 📋 NEXT**: Sequential Thinking MCP example

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
   
4. **Switch Between Comparisons**:
   Use the dropdown selector in the header to switch between available MCP comparisons

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
│   ├── comparison_manager.py    # Multi-comparison management
│   ├── dual_chat_ui/      # Side-by-side chat interface with comparison selector
│   └── orchestrator/      # GitHub-enabled orchestrator
├── configs/               # MCP-specific configurations
│   ├── sqlite/           # SQLite MCP comparison config (GitHub sources)
│   └── template/         # Template for new MCP comparisons (excluded from loading)
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

1. **Select a comparison** from the dropdown in the header (if multiple comparisons are available)
2. Type queries in either the baseline (left) or enhanced (right) chat panel
3. Each panel maintains its own conversation thread, isolated per comparison type
4. Compare responses, SQL queries, and performance
5. Switch between comparison types to test different MCP implementations
6. Use example queries to test common scenarios

## Adding New MCP Comparisons

1. **Copy the template**: `cp -r configs/template configs/your_mcp`
2. **Edit the configuration**: Update `configs/your_mcp/comparison_config.json` with your:
   - GitHub repository details
   - MCP server command and arguments  
   - UI customization (titles, colors, icons)
3. **Restart the application**: 
   ```bash
   uv run comparison_app.py
   ```
   
The framework automatically:
- **Discovers** your new comparison configuration
- **Clones** the GitHub repositories
- **Installs** dependencies
- **Adds** your comparison to the dropdown selector
- **No manual configuration** needed!

## Commands

- **Run**: `uv run comparison_app.py`
- **Format**: `ruff format .`
- **Lint**: `ruff check .`
- **Type Check**: `mypy .`
- **Fix Lint**: `ruff check --fix .`

## Architecture

See `specs/ARCHITECTURE_PLAN.md` for detailed implementation phases and future roadmap.