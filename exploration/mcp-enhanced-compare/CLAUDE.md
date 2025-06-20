# CLAUDE.md

This file provides guidance to Claude Code when working with the MCP Comparison Framework.

## Commands
- Run the comparison application: `uv run comparison_app.py`
- Format code: `ruff format .`
- Lint code: `ruff check .`
- Type check: `mypy .`
- Fix lint issues: `ruff check --fix .`

## Architecture
- **Multi-Comparison Management**: ComparisonManager handles multiple MCP comparison configurations
- **Real-time Switching**: Dropdown UI for switching between different MCP types without restart
- **GitHub Integration**: Automatically clones MCP servers from GitHub repositories
- **Comparison Framework**: Reusable system for comparing MCP servers
- **Dual-Chat UI**: Side-by-side interface with separate conversation threads
- **GitHub-Enabled Orchestrator**: Handles repository cloning, dependency installation, and MCP server management
- **Configuration-Driven**: Easy swapping of different MCP servers via GitHub sources
- **Template Exclusion**: Automatically excludes template directories from loading

## Code Style Guidelines
- Python: 3.11+
- Line length: 88 characters (Black default)
- Quotes: Double quotes for strings
- Type hints: Required for all functions and classes
- Error handling: Use explicit exception handling
- Naming: Follow PEP8 conventions

## Project Structure
- `framework/`: Core reusable components with GitHub integration
  - `comparison_manager.py`: Multi-comparison management and orchestrator lifecycle
  - `dual_chat_ui/`: Side-by-side interface with comparison selector dropdown
  - `orchestrator/`: GitHub-enabled MCP server orchestrator
- `configs/`: MCP server configurations with GitHub source support
  - `sqlite/`: Working SQLite MCP comparison
  - `template/`: Template for new comparisons (automatically excluded from loading)
- `temp/`: Auto-generated directories for cloned MCP servers (git-ignored)
- `data/`: Database files only
- `experiments/`: Evaluation results and test data

## Development Notes
- **Multi-comparison architecture** - ComparisonManager handles multiple MCP types simultaneously
- **No local MCP servers required** - everything cloned from GitHub
- Each MCP server runs in isolation from cloned temporary directories
- Baseline and enhanced versions use separate orchestrators with GitHub source handling
- **Real-time comparison switching** via dropdown selector without application restart
- **Conversation isolation** - separate chat histories per comparison type using scoped thread IDs
- UI supports separate input fields for independent conversations
- Health checks ensure MCP servers are running properly
- File watching automatically disabled when using GitHub sources to prevent reload loops
- **Template directories automatically excluded** from configuration loading
- Temp directories cleaned on startup, preserved for debugging