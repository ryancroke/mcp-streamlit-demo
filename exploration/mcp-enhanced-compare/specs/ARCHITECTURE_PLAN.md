# ARCHITECTURE_PLAN.md

## 1. Overview and Goals

**Objective:** Evolve the MCP Comparison Framework from a static, folder-based system into a dynamic, configuration-driven platform.

The framework has evolved from a proof-of-concept requiring manual code changes to a configuration-driven system. This plan outlines the architectural changes to achieve the following goals:

1.  **Decouple Development from Testing:** Enable developers to work on an MCP server in a separate GitHub repository. The comparison framework should be able to test the latest version simply by fetching it from the repository, eliminating the need for manual file copying.
2.  **Achieve Full Configuration-Driven Operation:** ✅ **COMPLETED** - Eliminated all hardcoded paths, titles, and queries. The entire comparison setup (what to test, how to run it, how to display it) is now defined in configuration files.
3.  **Support Multiple Comparison Scenarios:** Make it simple to switch between different types of MCP comparisons (e.g., "SQLite" vs. "Sequential-Thinking") without changing the core application code.

## 2. Implementation Status

**✅ Phase 1: COMPLETED** - Unified and centralized configuration
**✅ Phase 2: COMPLETED** - Dynamic MCP sourcing from GitHub  
**✅ Phase 3: COMPLETED** - Multiple, UI-switchable comparisons
**📋 Phase 4: NEXT** - Sequential Thinking MCP example

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

### Phase 4: Add Sequential Thinking MCP Comparison

**Goal:** Create a second MCP comparison configuration to demonstrate the multi-comparison capability using the Sequential Thinking MCP server.

**Actionable Steps:**

1.  **Create Sequential Thinking Configuration:**
    *   Create `configs/sequential/comparison_config.json` with GitHub sources pointing to the Sequential Thinking MCP.

    **Example: `configs/sequential/comparison_config.json`**
    ```json
    {
      "comparison_name": "Sequential Thinking MCP Comparison",
      "comparison_description": "Comparing Sequential Thinking MCP baseline vs enhanced versions.",
      "data_files": [],
      "baseline": {
        "source": {
          "type": "github",
          "repo": "modelcontextprotocol/servers",
          "branch": "main",
          "subdirectory": "src/sequentialthinking",
          "install_dir": "temp/sequential_baseline_from_git"
        },
        "mcp_server": {
          "name": "sequential_baseline_git",
          "command": "uv",
          "args": [
            "run",
            "mcp-server-sequentialthinking"
          ]
        },
        "ui": {
          "title": "Sequential Thinking Baseline",
          "color": "#9B59B6",
          "icon": "🧠"
        }
      },
      "enhanced": {
        "source": {
          "type": "github",
          "repo": "modelcontextprotocol/servers",
          "branch": "main",
          "subdirectory": "src/sequentialthinking",
          "install_dir": "temp/sequential_enhanced_from_git"
        },
        "mcp_server": {
          "name": "sequential_enhanced_git",
          "command": "uv",
          "args": [
            "run",
            "mcp-server-sequentialthinking"
          ]
        },
        "ui": {
          "title": "Sequential Thinking Enhanced",
          "color": "#E74C3C",
          "icon": "🚀"
        }
      }
    }
    ```

2.  **Test Multi-Comparison Switching:**
    *   Verify that users can switch between SQLite and Sequential Thinking comparisons via UI
    *   Ensure UI updates appropriately for different MCP types (titles, colors, icons)
    *   Test that GitHub sourcing works for both repository structures
    *   Validate conversation history isolation between comparison types

    **Example User Workflow:**
    ```
    1. Open http://localhost:8001
    2. Select "SQLite MCP Comparison" from dropdown
    3. Test database queries in both panels
    4. Switch to "Sequential Thinking MCP" from dropdown
    5. Test reasoning tasks in both panels
    6. Switch back to SQLite - previous conversation history preserved
    ```

**Benefits of This Approach:**
- Demonstrates the framework's MCP-agnostic design
- Tests GitHub integration with two different repository structures
- Provides a concrete example of how to add new MCP comparisons
- Validates that the UI can handle different types of MCP servers
- Shows the power of configuration-driven architecture

## 3. New Workflow (Post-Implementation)

Once these changes are complete, the development and testing workflow will be dramatically simplified:

1.  **Develop:** A developer makes changes to the enhanced MCP server and pushes them to its dedicated GitHub repository.
2.  **Configure:** The `comparison_config.json` is set up once to point to this GitHub repository.
3.  **Test:** The user runs `uv run comparison_app.py`.
4.  **Automate:** The framework automatically fetches the latest code from GitHub, sets up the environment, and starts the comparison UI.
5.  **Evaluate:** The user can immediately begin comparing the stable baseline against the latest development version, with zero manual intervention.