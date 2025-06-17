# CLAUDE.md

This file provides guidance to Claude Code when working with the MCP Comparison Framework.

## Commands
- Run the comparison application: `python comparison_app.py`
- Format code: `ruff format .`
- Lint code: `ruff check .`
- Type check: `mypy .`
- Fix lint issues: `ruff check --fix .`

## Architecture
- **Comparison Framework**: Reusable system for comparing MCP servers
- **Dual-Chat UI**: Side-by-side interface with separate conversation threads
- **Simple Orchestrator**: Streamlined orchestration without A2A agents or multiple MCPs
- **Configuration-Driven**: Easy swapping of different MCP servers

## Code Style Guidelines
- Python: 3.11+
- Line length: 88 characters (Black default)
- Quotes: Double quotes for strings
- Type hints: Required for all functions and classes
- Error handling: Use explicit exception handling
- Naming: Follow PEP8 conventions

## Project Structure
- `framework/`: Core reusable components
- `configs/`: MCP server configurations
- `data/`: Database files and MCP server code
- `experiments/`: Evaluation results and test data

## Development Notes
- Each MCP server runs in isolation
- Baseline and enhanced versions use separate orchestrators
- UI supports separate input fields for independent conversations
- Health checks ensure MCP servers are running properly