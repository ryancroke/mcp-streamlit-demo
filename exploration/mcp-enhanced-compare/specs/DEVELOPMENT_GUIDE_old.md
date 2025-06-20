# MCP Comparison Framework - Development Guide

## Overview

This framework provides a **reusable, configuration-driven system** for comparing baseline vs enhanced MCP (Model Context Protocol) servers. The architecture is designed to be **completely MCP-agnostic**, allowing you to compare any MCP implementation without writing code.

## Table of Contents

1. [Current Architecture](#current-architecture)
2. [Adding a New MCP Comparison](#adding-a-new-mcp-comparison)
3. [Roadmap to Full Abstraction](#roadmap-to-full-abstraction)
4. [Configuration Reference](#configuration-reference)
5. [Development Patterns](#development-patterns)
6. [Testing Guidelines](#testing-guidelines)
7. [Troubleshooting](#troubleshooting)

---

## Current Architecture

### Core Components

```
mcp-enhanced-compare/
├── framework/              # Reusable comparison framework
│   ├── dual_chat_ui/      # MCP-agnostic UI components
│   └── orchestrator/      # LangGraph + mcp-use integration
├── configs/               # MCP-specific configurations
├── mcp_servers/          # MCP server implementations
├── data/                 # Database/data files
└── experiments/          # Evaluation results
```

### Abstraction Levels

**✅ Fully Abstracted (MCP-agnostic):**
- MCP Interface using `mcp-use` pattern
- LangGraph orchestration with memory
- FastAPI backend routing
- Dual-chat UI framework
- Configuration loading system

**⚠️ Partially Abstracted (some hardcoding):**
- Health check queries (MCP-specific)
- UI labels and titles
- Directory naming conventions

**❌ Not Abstracted (SQLite-specific):**
- Example queries in UI
- Health check implementation
- Some error messages

---

## Adding a New MCP Comparison

### Step 1: Prepare MCP Server Code

1. **Create server directories:**
   ```bash
   mkdir -p mcp_servers/mcp_[NAME]_baseline
   mkdir -p mcp_servers/mcp_[NAME]_enhanced
   ```

2. **Copy baseline MCP server:**
   - Place official MCP server code in `mcp_servers/mcp_[NAME]_baseline/`
   - Ensure it has a `pyproject.toml` with proper MCP server configuration
   - Test that `uv run [mcp-command]` works in the directory

3. **Create enhanced version:**
   - Copy baseline to `mcp_servers/mcp_[NAME]_enhanced/`
   - Make your enhancements to the server code
   - Document changes in the enhanced directory

### Step 2: Create Configuration Files

1. **Create config directory:**
   ```bash
   mkdir -p configs/[NAME]
   ```

2. **Baseline configuration** (`configs/[NAME]/baseline_config.json`):
   ```json
   {
     "mcp_server": {
       "name": "[NAME]_baseline",
       "type": "[NAME]",
       "command": "uv",
       "args": [
         "--directory",
         "mcp_servers/mcp_[NAME]_baseline",
         "run",
         "[mcp-command]",
         "--[arg1]",
         "data/[data-file]"
       ],
       "description": "Official [NAME] MCP Server - Baseline Version"
     },
     "ui_config": {
       "title": "[NAME] Baseline",
       "color": "#4A90E2",
       "side": "left"
     }
   }
   ```

3. **Enhanced configuration** (`configs/[NAME]/enhanced_config.json`):
   ```json
   {
     "mcp_server": {
       "name": "[NAME]_enhanced",
       "type": "[NAME]",
       "command": "uv",
       "args": [
         "--directory",
         "mcp_servers/mcp_[NAME]_enhanced",
         "run",
         "[mcp-command]",
         "--[arg1]",
         "data/[data-file]"
       ],
       "description": "Enhanced [NAME] MCP Server - Custom Version"
     },
     "ui_config": {
       "title": "[NAME] Enhanced",
       "color": "#E25A4A",
       "side": "right"
     }
   }
   ```

### Step 3: Update Application Configuration

**Currently requires manual code changes:**

1. **Update `comparison_app.py`:**
   ```python
   # Change config paths
   baseline_config_path = Path("configs/[NAME]/baseline_config.json")
   enhanced_config_path = Path("configs/[NAME]/enhanced_config.json")
   ```

2. **Update health check in `mcp_interface.py`:**
   ```python
   # Update health check query for your MCP type
   if self.server_name == "[NAME]":
       result = await self.agent.run("Your MCP-specific health check query")
   ```

3. **Update UI labels in `index.html`:**
   ```html
   <!-- Update titles and descriptions -->
   <h2>🔵 [NAME] Baseline</h2>
   <h2>🔴 [NAME] Enhanced</h2>
   ```

### Step 4: Add Data Files

1. **Place data files in `data/`:**
   ```bash
   cp your-data-file data/
   ```

2. **Update config paths to point to correct data files**

### Step 5: Test the New MCP

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Test MCP servers individually:**
   ```bash
   cd mcp_servers/mcp_[NAME]_baseline
   uv run [mcp-command] --help
   ```

3. **Run the comparison app:**
   ```bash
   uv run start.py
   ```

4. **Verify both sides work:**
   - Open http://localhost:8001
   - Test queries in both baseline and enhanced panels
   - Check health status indicators

---

## Roadmap to Full Abstraction

### Phase 1: Configuration-Driven UI (High Priority)

**Goal:** Eliminate all hardcoded MCP-specific text and queries.

**Changes needed:**

1. **Enhanced configuration format:**
   ```json
   {
     "comparison_info": {
       "name": "SQLite MCP Comparison",
       "description": "Compare baseline vs enhanced SQLite MCP servers",
       "data_source": "Chinook Music Database"
     },
     "baseline": {
       "mcp_server": { ... },
       "ui_config": {
         "title": "SQLite Baseline",
         "color": "#4A90E2",
         "icon": "🔵"
       },
       "health_check": {
         "query": "List the names of all tables in the database",
         "success_indicator": "contains table names"
       },
       "example_queries": [
         "How many artists are in the database?",
         "What are the top 5 best-selling albums?"
       ]
     },
     "enhanced": { ... }
   }
   ```

2. **Dynamic UI generation:**
   - Load comparison config via API endpoint
   - Generate UI elements from configuration
   - Update titles, colors, and example queries dynamically

3. **MCP registry system:**
   ```json
   {
     "available_comparisons": [
       {
         "id": "sqlite",
         "name": "SQLite MCP",
         "config_path": "configs/sqlite/comparison_config.json"
       },
       {
         "id": "github", 
         "name": "GitHub MCP",
         "config_path": "configs/github/comparison_config.json"
       }
     ]
   }
   ```

### Phase 2: Multi-MCP Support (Medium Priority)

**Goal:** Support multiple MCP comparisons in a single deployment.

**Changes needed:**

1. **MCP selection endpoint:**
   ```python
   @app.get("/api/comparisons")
   async def list_available_comparisons():
       # Return available MCP comparisons
   
   @app.post("/api/select-comparison")
   async def select_comparison(comparison_id: str):
       # Switch active MCP comparison
   ```

2. **Dynamic orchestrator loading:**
   ```python
   # Load orchestrators based on selected comparison
   active_comparison = load_comparison_config(comparison_id)
   baseline_orchestrator = create_orchestrator(active_comparison["baseline"])
   enhanced_orchestrator = create_orchestrator(active_comparison["enhanced"])
   ```

3. **UI MCP selector:**
   ```html
   <select id="mcpSelector">
     <option value="sqlite">SQLite MCP Comparison</option>
     <option value="github">GitHub MCP Comparison</option>
   </select>
   ```

### Phase 3: Auto-Discovery (Low Priority)

**Goal:** Automatically discover available MCP comparisons.

**Changes needed:**

1. **Directory scanning:**
   ```python
   def discover_mcp_comparisons():
       # Scan configs/ directory for comparison configs
       # Validate MCP server directories exist
       # Return available comparisons
   ```

2. **Validation system:**
   ```python
   def validate_mcp_comparison(config_path):
       # Check config file format
       # Verify MCP server directories exist
       # Test MCP server can start
       # Return validation results
   ```

---

## Configuration Reference

### MCP Server Configuration

**Required fields:**
- `name`: Unique identifier for this MCP instance
- `command`: Executable command (usually `uv`)
- `args`: Command arguments including MCP server path and options

**Optional fields:**
- `type`: MCP type for categorization
- `description`: Human-readable description
- `env`: Environment variables for the MCP server

### UI Configuration

**Required fields:**
- `title`: Display name in the UI
- `color`: CSS color for theming
- `side`: "left" or "right" panel placement

**Optional fields:**
- `icon`: Emoji or icon for the panel
- `description`: Subtitle text

### Health Check Configuration

**Current implementation:**
- Hardcoded queries in `mcp_interface.py`
- MCP-specific logic

**Proposed implementation:**
- `query`: Health check query string
- `success_criteria`: How to determine if response indicates health
- `timeout`: Maximum time to wait for response

---

## Development Patterns

### Adding New Features

1. **Always maintain MCP-agnosticism:**
   - Don't hardcode MCP-specific logic
   - Use configuration for MCP-specific values
   - Test with multiple MCP types

2. **Follow the abstraction layers:**
   - **Framework layer**: MCP-agnostic components
   - **Configuration layer**: MCP-specific settings
   - **Implementation layer**: Actual MCP server code

3. **Preserve the comparison contract:**
   - Both baseline and enhanced must use same interface
   - Maintain separate conversation threads
   - Ensure fair comparison conditions

### Configuration Best Practices

1. **Use relative paths** in configs (resolved at runtime):
   ```json
   "args": ["--db-path", "data/database.db"]
   ```

2. **Include validation information:**
   ```json
   "validation": {
     "required_files": ["data/database.db"],
     "health_check_timeout": 30
   }
   ```

3. **Document MCP-specific requirements:**
   ```json
   "requirements": {
     "python_version": ">=3.11",
     "system_dependencies": ["sqlite3"],
     "notes": "Requires SQLite database file"
   }
   ```

### Error Handling Patterns

1. **Graceful degradation:**
   - If one MCP fails, other should continue working
   - Clear error messages for users
   - Automatic retry with exponential backoff

2. **Configuration validation:**
   - Validate configs on startup
   - Check required files exist
   - Test MCP server connectivity

3. **Health monitoring:**
   - Regular health checks
   - Status indicators in UI
   - Automatic recovery attempts

---

## Testing Guidelines

### Unit Testing

1. **Test configuration loading:**
   ```python
   def test_load_config():
       config = load_config("configs/sqlite/baseline_config.json")
       assert config["mcp_server"]["name"] == "sqlite_baseline"
   ```

2. **Test MCP interface abstraction:**
   ```python
   async def test_mcp_interface_abstraction():
       # Should work with any MCP type
       interface = await create_mcp_interface_from_config(config)
       result = await interface.query("test query")
       assert isinstance(result, str)
   ```

3. **Test orchestrator with different MCPs:**
   ```python
   async def test_orchestrator_mcp_agnostic():
       # Test with SQLite config
       sqlite_orchestrator = SimpleMCPOrchestrator("configs/sqlite/baseline_config.json")
       # Test with GitHub config (when available)
       github_orchestrator = SimpleMCPOrchestrator("configs/github/baseline_config.json")
   ```

### Integration Testing

1. **End-to-end MCP comparison:**
   ```bash
   # Test SQLite comparison
   COMPARISON_TYPE=sqlite pytest tests/test_e2e.py
   
   # Test GitHub comparison
   COMPARISON_TYPE=github pytest tests/test_e2e.py
   ```

2. **Conversation memory testing:**
   ```python
   async def test_conversation_memory():
       # Test that conversation context is maintained
       # Test that baseline and enhanced have separate contexts
   ```

3. **Health check testing:**
   ```python
   async def test_health_checks():
       # Test health checks for all configured MCPs
       # Verify proper error handling for unhealthy MCPs
   ```

### Performance Testing

1. **MCP response time comparison:**
   ```python
   async def benchmark_mcp_performance():
       # Compare response times between baseline and enhanced
       # Test with various query types and sizes
   ```

2. **Memory usage monitoring:**
   ```python
   def test_memory_usage():
       # Monitor memory usage over long conversations
       # Check for memory leaks in LangGraph checkpointing
   ```

### Manual Testing Checklist

- [ ] Both MCP panels load successfully
- [ ] Health status indicators work correctly
- [ ] Conversation memory is maintained per panel
- [ ] Example queries work for the specific MCP type
- [ ] Error messages are clear and helpful
- [ ] UI responds appropriately to MCP failures
- [ ] Configuration changes take effect without code changes

---

## Troubleshooting

### Common Issues

**1. MCP Server Won't Start**
```
Error: Failed to initialize [NAME]MCP
```
- Check MCP server directory exists and has correct structure
- Verify `uv run` command works in the MCP server directory
- Check for missing dependencies in MCP server's `pyproject.toml`
- Ensure data files exist at specified paths

**2. Health Check Failures**
```
Error: [NAME]MCP health check failed
```
- Verify health check query is appropriate for the MCP type
- Check if MCP server is responding to queries
- Ensure database/data source is accessible
- Test the health check query manually

**3. Path Resolution Issues**
```
Error: No such file or directory: data/database.db
```
- Check that relative paths in config are correct
- Verify data files are in the expected locations
- Ensure path resolution logic handles your directory structure

**4. Configuration Loading Errors**
```
Error: Server '[NAME]' not found in config
```
- Verify config file JSON syntax is valid
- Check that server name matches between files
- Ensure config file exists at expected path

### Debugging Tips

1. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test MCP servers independently:**
   ```bash
   cd mcp_servers/mcp_[NAME]_baseline
   uv run [mcp-command] --help
   ```

3. **Check MCP communication:**
   ```python
   # Test direct MCP interface
   interface = await create_mcp_interface_from_config(config)
   result = await interface.query("simple test query")
   print(result)
   ```

4. **Validate configurations:**
   ```python
   # Test config loading
   config = load_config("configs/[NAME]/baseline_config.json")
   print(json.dumps(config, indent=2))
   ```

### Performance Optimization

1. **MCP server startup time:**
   - Consider keeping MCP servers running between requests
   - Implement connection pooling for frequently used MCPs
   - Cache MCP interface instances

2. **Memory usage:**
   - Monitor LangGraph checkpointer memory usage
   - Implement conversation history cleanup
   - Set reasonable limits on conversation length

3. **Response times:**
   - Profile MCP query performance
   - Implement query timeouts
   - Consider parallel processing for independent operations

---

## Future Enhancements

### Advanced Features

1. **Batch evaluation system:**
   - Load test queries from files
   - Compare responses automatically
   - Generate comparison reports

2. **A/B testing framework:**
   - Statistical significance testing
   - Performance metrics collection
   - Automated decision making

3. **Custom evaluation metrics:**
   - Accuracy scoring
   - Response time analysis
   - User preference tracking

4. **Multi-user support:**
   - Separate conversation threads per user
   - User authentication and authorization
   - Shared evaluation results

### Scalability Improvements

1. **Distributed MCP servers:**
   - Run MCP servers on separate machines
   - Load balancing for high availability
   - Horizontal scaling support

2. **Cloud deployment:**
   - Docker containerization
   - Kubernetes orchestration
   - Cloud-native configuration management

3. **Real-time collaboration:**
   - WebSocket support for live updates
   - Shared conversation sessions
   - Real-time performance monitoring

---

This development guide provides a comprehensive roadmap for extending the MCP Comparison Framework. The key principle is maintaining **MCP-agnosticism** while providing powerful comparison capabilities. As you add new MCP types, focus on configuration-driven development rather than code changes.