# ARCHITECTURE_PLAN.md

## 1. Overview and Goals

**Objective:** Evolve the MCP Comparison Framework from a static, folder-based system into a dynamic, configuration-driven platform with pluggable architecture for unlimited MCP server types.

The framework has evolved from a proof-of-concept requiring manual code changes to a fully extensible, handler-based system. This plan outlines the architectural journey to achieve the following goals:

1.  **Decouple Development from Testing:** ✅ **COMPLETED** - Enable developers to work on an MCP server in a separate GitHub repository. The comparison framework automatically fetches the latest version from the repository, eliminating manual file copying.
2.  **Achieve Full Configuration-Driven Operation:** ✅ **COMPLETED** - Eliminated all hardcoded paths, titles, and queries. The entire comparison setup (what to test, how to run it, how to display it) is now defined in configuration files.
3.  **Support Multiple Comparison Scenarios:** ✅ **COMPLETED** - Simple switching between different types of MCP comparisons (e.g., "SQLite" vs. "Sequential-Thinking") without changing core application code.
4.  **Eliminate MCP-Specific Conditionals:** ✅ **COMPLETED** - Pluggable handler architecture that scales to unlimited MCP server types without code changes.

## 2. Implementation Status

**✅ Phase 1: COMPLETED** - Unified and centralized configuration
**✅ Phase 2: COMPLETED** - Dynamic MCP sourcing from GitHub  
**✅ Phase 3: COMPLETED** - Multiple, UI-switchable comparisons
**✅ Phase 4: COMPLETED** - Pluggable Handler Architecture & Sequential Thinking MCP

## 3. Phased Implementation Details

### Phase 1: Unify and Centralize Configuration ✅ COMPLETED

**Goal:** Remove all hardcoded paths and UI text by consolidating configuration into a single, comprehensive file for each comparison type.

**✅ Completed Implementation:**

1.  **✅ Created Unified `comparison_config.json`:**
    *   Uses a single unified `configs/sqlite/comparison_config.json` with GitHub source support.
    *   This file will describe the entire comparison: the name, the data files, and the configurations for *both* the baseline and enhanced orchestrators.

    **Example: `configs/sqlite/comparison_config.json`**
    ```json
    {
      "comparison_name": "SQLite MCP Comparison",
      "comparison_description": "Comparing the official SQLite MCP with an enhanced version.",
      "data_files": ["data/Chinook_Sqlite.db"],
      "baseline": {
        "mcp_server": {
          "name": "sqlite_baseline",
          "command": "uv",
          "args": [
            "run",
            "--directory", "mcp_servers/mcp_sqlite_baseline",
            "mcp", "serve",
            "--db", "data/Chinook_Sqlite.db"
          ]
        },
        "ui": {
          "title": "SQLite Baseline",
          "color": "#4A90E2",
          "icon": "🔵"
        }
      },
      "enhanced": {
        "mcp_server": {
          "name": "sqlite_enhanced",
          "command": "uv",
          "args": [
            "run",
            "--directory", "mcp_servers/mcp_sqlite_enhanced",
            "mcp", "serve",
            "--db", "data/Chinook_Sqlite.db"
          ]
        },
        "ui": {
          "title": "SQLite Enhanced",
          "color": "#E25A4A",
          "icon": "🔴"
        }
      }
    }
    ```

2.  **✅ Refactored `comparison_app.py` to be Config-Driven:**
    *   ✅ Modified the `lifespan` manager to load comparison configuration from `MCP_COMPARISON_CONFIG` environment variable (defaults to `configs/sqlite/comparison_config.json`)
    *   ✅ Updated `SimpleMCPOrchestrator` to accept configuration dictionaries directly
    *   ✅ Created `/api/config` endpoint that serves UI-specific configuration data

3.  **✅ Made Frontend UI Dynamic:**
    *   ✅ Updated `framework/dual_chat_ui/app.js` to fetch configuration from `/api/config` on page load
    *   ✅ Implemented dynamic UI population for titles, colors, icons, and placeholders
    *   ✅ Added CSS custom properties for dynamic theming
    *   ✅ Removed all hardcoded UI labels and text

**✅ Phase 1 Results:**
- Configuration-driven operation achieved
- Single source of truth for comparison setup
- Dynamic UI that adapts to configuration  
- Environment variable support for flexible deployment
- Backward compatibility maintained for MCP server operations

---

### Phase 2: Implement Dynamic MCP Sourcing from GitHub ✅ COMPLETED

**Goal:** Enable both baseline and enhanced servers to be sourced directly from Git repositories at startup, allowing for easy testing of the GitHub integration functionality.

**Understanding & Approach:**
- **Current command**: `uv --directory mcp_servers/mcp_sqlite_baseline run mcp-server-sqlite --db-path data/Chinook_Sqlite.db`
- **GitHub approach**: `uv --directory temp/mcp_baseline_from_git/src/sqlite run mcp-server-sqlite --db-path ../../data/Chinook_Sqlite.db`
- **Strategy**: Always re-clone for clean state, fail completely if GitHub operations fail

**Actionable Steps:**

1. **Extend Configuration Schema:**
   - Add optional `source` object to baseline/enhanced config blocks
   - Support `type: "github"` with repo, branch, subdirectory, install_dir
   - Keep existing local configs working (no source = local mode)

   **Example: Updated `comparison_config.json` with GitHub sources**
   ```json
   {
     "comparison_name": "SQLite MCP Comparison (GitHub-sourced)",
     "comparison_description": "Comparing SQLite MCP servers sourced directly from GitHub.",
     "data_files": ["data/Chinook_Sqlite.db"],
     "baseline": {
       "source": {
         "type": "github",
         "repo": "modelcontextprotocol/servers-archived",
         "branch": "main",
         "subdirectory": "src/sqlite",
         "install_dir": "temp/mcp_baseline_from_git"
       },
       "mcp_server": {
         "name": "sqlite_baseline_git",
         "command": "uv",
         "args": [
           "run",
           "mcp-server-sqlite",
           "--db-path", "data/Chinook_Sqlite.db"
         ]
       },
       "ui": {
         "title": "SQLite Baseline (Git)",
         "color": "#4A90E2",
         "icon": "🔵"
       }
     },
     "enhanced": {
       "source": {
         "type": "github",
         "repo": "modelcontextprotocol/servers-archived",
         "branch": "main",
         "subdirectory": "src/sqlite",
         "install_dir": "temp/mcp_enhanced_from_git"
       },
       "mcp_server": {
         "name": "sqlite_enhanced_git",
         "command": "uv",
         "args": [
           "run",
           "mcp-server-sqlite",
           "--db-path", "data/Chinook_Sqlite.db"
         ]
       },
       "ui": {
         "title": "SQLite Enhanced (Git)",
         "color": "#E25A4A",
         "icon": "🔴"
       }
     }
   }
   ```

2. **Update SimpleMCPOrchestrator:**
   - Modify `initialize()` method to check for `source` configuration
   - If GitHub source found:
     1. Delete install_dir if exists (clean slate)
     2. `git clone --branch {branch} {repo} {install_dir}`
     3. Dynamically insert `--directory {install_dir}/{subdirectory}` into args
     4. Adjust data file paths to be relative from cloned directory
     5. Run `uv sync` in cloned directory to install dependencies
     6. Proceed with normal MCP server startup

3. **Path Resolution Strategy:**
   - Convert relative data paths to absolute paths from project root
   - This ensures `--db-path ../../data/Chinook_Sqlite.db` works from cloned subdirectory

4. **Error Handling:**
   - Git clone failures → fail completely with clear error message
   - Missing subdirectories → fail completely
   - Dependency installation failures → fail completely
   - No fallback to local servers

5. **Testing Configuration:**
   - Create GitHub-sourced config using same repo for both baseline/enhanced
   - Use `modelcontextprotocol/servers-archived` repo, `src/sqlite` subdirectory
   - This validates the complete GitHub workflow before introducing actual differences

**Key Technical Details:**
- **Command structure**: Insert `--directory` at position 0 in args array
- **Dependencies**: Run `uv sync` in cloned directory before starting MCP server
- **Cleanup**: Always delete and re-clone (no git pull optimization)
- **Paths**: Make data file paths absolute to work from any working directory

**✅ Phase 2 Results:**
- GitHub-driven architecture achieved - no local MCP servers required
- Automatic repository cloning and dependency installation
- Dynamic configuration switching between local and GitHub sources
- Robust error handling for git operations and dependency failures
- Default configuration now uses GitHub sources (`modelcontextprotocol/servers-archived`)
- File watcher issues resolved (auto-disables reload for GitHub sources)
- Temp directory management: clean on startup, preserve for debugging
- Framework successfully decouples development from testing workflow

**Example Usage:**
```bash
# Uses GitHub sources by default
uv run comparison_app.py

# Custom GitHub configuration
MCP_COMPARISON_CONFIG=configs/custom/comparison_config.json uv run comparison_app.py
```

---

### Phase 3: Support Multiple, UI-Switchable Comparisons ✅ COMPLETED

**Goal:** Allow users to switch between entirely different MCP comparisons (e.g., SQLite vs. Sequential-Thinking MCP) through the web UI in real-time without restarting the application.

**Key Architectural Changes:**

This phase transforms the framework from a single-comparison tool into a true multi-comparison platform with seamless UI-based switching.

**✅ Completed Implementation:**

1.  **✅ Created ComparisonManager Class:**
    *   Added `framework/comparison_manager.py` to handle multiple comparison configurations
    *   Scans `configs/` directory for `comparison_config.json` files (excluding template directories)
    *   Validates and initializes orchestrators for each valid comparison
    *   Manages orchestrator lifecycle with proper cleanup
    *   Supports switching between active comparisons

2.  **✅ Added Backend API Endpoints:**
    *   `/api/comparisons/list` - Returns available comparisons with UI metadata
    *   `/api/comparisons/current` - Returns active comparison configuration  
    *   `/api/comparisons/switch` - Switches active comparison via POST request
    *   Modified existing query endpoints to use active comparison's orchestrators

3.  **✅ Refactored comparison_app.py:**
    *   Replaced global orchestrator variables with ComparisonManager instance
    *   Updated `lifespan` function to initialize ComparisonManager
    *   Modified query processing to route to active comparison's orchestrators
    *   Added proper error handling for missing comparisons

4.  **✅ Updated Conversation Thread Scoping:**
    *   Changed thread ID format to: `{comparison_id}_{orchestrator_type}_{thread_id}`
    *   Examples: `sqlite_baseline_abc123`, `sequential_enhanced_def456`  
    *   Prevents cross-contamination between comparison types
    *   Maintains separate conversation histories per comparison type

5.  **✅ Added Frontend Comparison Selector:**
    *   Added dropdown selector in header for comparison selection
    *   Fetches available comparisons on page load via `/api/comparisons/list`
    *   Updates UI titles, colors, and icons dynamically when switching
    *   Handles loading states during comparison switches
    *   Clears chat histories when switching to prevent confusion

**✅ Phase 3 Results:**
- ✅ Real-time switching between MCP types without application restart
- ✅ Template directories automatically excluded from loading
- ✅ Graceful fallback when comparisons fail to initialize
- ✅ Complete conversation isolation between comparison types
- ✅ Dynamic UI that adapts to available comparisons
- ✅ Framework successfully scales to multiple MCP comparison scenarios

---

### Phase 4: Pluggable Handler Architecture & Sequential Thinking MCP ✅ COMPLETED

**Goal:** Eliminate MCP-specific conditionals from the orchestrator and create a pluggable architecture that scales to unlimited MCP server types, demonstrated with a Sequential Thinking MCP comparison.

**✅ Completed Implementation:**

1.  **✅ Created Handler Interface (`framework/handlers/base.py`):**
    *   Abstract `MCPServerHandler` class with methods:
      - `can_handle()` - Auto-detect MCP server type
      - `install_dependencies()` - Handle installation/building  
      - `transform_command()` - Transform commands for local execution
      - `get_health_check_query()` - Return appropriate health check
      - `get_priority()` - Handler selection priority

2.  **✅ Implemented Concrete Handlers:**
    *   `PythonUVHandler` (`framework/handlers/python_uv.py`) - For Python/uv-based MCP servers
      - Detects `pyproject.toml` presence
      - Installs with `uv sync`
      - Transforms commands with `--directory` flag
    *   `NodeJSHandler` (`framework/handlers/nodejs.py`) - For Node.js/npm-based MCP servers  
      - Detects `package.json` presence
      - Installs with `npm install`, builds with `npm run build`
      - Transforms `npx` commands to local `node` execution
    *   `GenericHandler` (`framework/handlers/generic.py`) - Fallback for unknown types
      - Always returns True for `can_handle()`
      - No-op installation and transformation

3.  **✅ Created Handler Registry (`framework/handlers/registry.py`):**
    *   Auto-detects appropriate handler for each MCP server
    *   Extensible registry for adding new handlers
    *   Priority-based handler selection

4.  **✅ Refactored Simple Orchestrator:**
    *   Removed **all MCP-specific conditionals** (eliminated ~100 lines of conditional logic)
    *   Uses handler registry for all MCP server operations
    *   Clean separation of concerns

5.  **✅ Added Sequential Thinking MCP Comparison:**
    *   Created `configs/sequential/comparison_config.json` 
    *   Working Node.js/TypeScript-based MCP server
    *   Auto-detected and built by `NodeJSHandler`
    *   Demonstrates step-by-step reasoning functionality

**✅ Phase 4 Results:**
- ✅ **Zero conditionals** in main orchestrator code  
- ✅ **Auto-detection** working for Python vs Node.js projects
- ✅ **Sequential Thinking MCP** fully functional with step-by-step reasoning
- ✅ **Easy extensibility** - just add handlers to registry
- ✅ **Future-proof** architecture ready for Docker, Go, Rust handlers
- ✅ **Maintainable** - each handler owns its MCP type's logic

**Example Handler Usage Logs:**
```
🔧 Selected NodeJSHandler for MCP server at temp/sequential_baseline_from_git/src/sequentialthinking
🔧 Selected PythonUVHandler for MCP server at temp/mcp_baseline_from_git/src/sqlite
```

---

## 4. New Workflow (Fully Implemented)

The development and testing workflow is now dramatically simplified with the pluggable handler architecture:

1.  **Develop:** A developer makes changes to any MCP server (Python, Node.js, Docker, etc.) and pushes them to GitHub.
2.  **Configure:** Create a simple `comparison_config.json` pointing to the GitHub repository - **no MCP type specification needed**.
3.  **Auto-Detect:** The framework automatically detects the MCP server type and selects the appropriate handler.
4.  **Test:** Run `uv run comparison_app.py` - the framework handles everything automatically.
5.  **Automate:** The system automatically:
   - Clones the latest code from GitHub
   - Detects MCP server type (Python/Node.js/Docker/etc.)
   - Installs dependencies using the correct tools
   - Builds projects as needed
   - Starts the comparison UI
6.  **Evaluate:** Immediately compare different MCP implementations with zero manual intervention.

## 5. Future Extensions (Easily Achievable)

The pluggable architecture makes these extensions trivial:

### Docker Handler
```python
class DockerHandler(MCPServerHandler):
    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        return (work_dir / "Dockerfile").exists()
    
    def install_dependencies(self, work_dir: Path) -> None:
        subprocess.run(["docker", "build", ".", "-t", "mcp-server"], cwd=work_dir)
    
    def transform_command(self, config: dict[str, Any], work_dir: Path) -> dict[str, Any]:
        return {"command": "docker", "args": ["run", "-i", "mcp-server"]}
```

### Go Handler
```python  
class GoHandler(MCPServerHandler):
    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        return (work_dir / "go.mod").exists()
    
    def install_dependencies(self, work_dir: Path) -> None:
        subprocess.run(["go", "mod", "download"], cwd=work_dir)
        subprocess.run(["go", "build", "."], cwd=work_dir)
```

### Rust Handler
```python
class RustHandler(MCPServerHandler):
    def can_handle(self, work_dir: Path, config: dict[str, Any]) -> bool:
        return (work_dir / "Cargo.toml").exists()
    
    def install_dependencies(self, work_dir: Path) -> None:
        subprocess.run(["cargo", "build", "--release"], cwd=work_dir)
```

**Just add to registry - zero core code changes!** 🚀