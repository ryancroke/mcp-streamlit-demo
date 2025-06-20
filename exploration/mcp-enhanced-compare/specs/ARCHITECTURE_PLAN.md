# ARCHITECTURE_PLAN.md

## 1. Overview and Goals

**Objective:** Evolve the MCP Comparison Framework from a static, folder-based system into a dynamic, configuration-driven platform.

The framework has evolved from a proof-of-concept requiring manual code changes to a configuration-driven system. This plan outlines the architectural changes to achieve the following goals:

1.  **Decouple Development from Testing:** Enable developers to work on an MCP server in a separate GitHub repository. The comparison framework should be able to test the latest version simply by fetching it from the repository, eliminating the need for manual file copying.
2.  **Achieve Full Configuration-Driven Operation:** ✅ **COMPLETED** - Eliminated all hardcoded paths, titles, and queries. The entire comparison setup (what to test, how to run it, how to display it) is now defined in configuration files.
3.  **Support Multiple Comparison Scenarios:** Make it simple to switch between different types of MCP comparisons (e.g., "SQLite" vs. "Sequential-Thinking") without changing the core application code.

## 2. Implementation Status

**✅ Phase 1: COMPLETED** - Unified and centralized configuration
**🔄 Phase 2: NEXT** - Dynamic MCP sourcing from GitHub  
**📋 Phase 3: PLANNED** - Multiple, UI-switchable comparisons
**📋 Phase 4: PLANNED** - Sequential Thinking MCP example

## 3. Phased Implementation Details

### Phase 1: Unify and Centralize Configuration ✅ COMPLETED

**Goal:** Remove all hardcoded paths and UI text by consolidating configuration into a single, comprehensive file for each comparison type.

**✅ Completed Implementation:**

1.  **✅ Created Unified `comparison_config.json`:**
    *   Instead of separate `baseline_config.json` and `enhanced_config.json`, create a single `configs/sqlite/comparison_config.json`.
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

### Phase 2: Implement Dynamic MCP Sourcing from GitHub

**Goal:** Enable both baseline and enhanced servers to be sourced directly from Git repositories at startup, allowing for easy testing of the GitHub integration functionality.

**Testing Approach:** To validate the GitHub integration works properly, we'll configure both baseline and enhanced to use the same public SQLite MCP repository (`https://github.com/modelcontextprotocol/servers-archived`) but potentially different branches or subdirectories. This allows us to test the complete GitHub sourcing workflow.

**Actionable Steps:**

1.  **Extend the Configuration Schema:**
    *   Add an optional `source` object to both `baseline` and `enhanced` configuration blocks. This object will specify the source type and location.

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
    *Note: The `--directory` argument is intentionally removed from `mcp_server.args`. It will be added dynamically. Also added `subdirectory` field to handle repositories where the MCP server is in a subdirectory.*

2.  **Update Orchestrator Initialization Logic:**
    *   In `framework/orchestrator/simple_orchestrator.py`, modify the `initialize` method.
    *   Before starting the MCP server, it should check for the `source` object in its configuration.
    *   If a `github` source is found, the orchestrator must:
        1.  Delete the `install_dir` if it exists to ensure a clean slate.
        2.  Use a `subprocess` call to `git clone` the specified repository and branch into the `install_dir`.
        3.  If `subdirectory` is specified, navigate to that subdirectory within the cloned repo.
        4.  Dynamically modify its own `mcp_server.args` list to insert the `--directory` flag and the path to the correct directory (either `install_dir` or `install_dir/subdirectory`).
        5.  Proceed with starting the server process as usual.

**Benefits of This Testing Approach:**
- Validates GitHub integration with a real, public repository
- Tests the complete workflow without requiring custom repositories
- Both baseline and enhanced use the same source, making it easy to verify they work identically
- Later, enhanced can be pointed to a fork or different branch for actual comparison testing

---

### Phase 3: Support Multiple, UI-Switchable Comparisons

**Goal:** Allow users to switch between entirely different MCP comparisons (e.g., SQLite vs. Sequential-Thinking MCP) through the web UI in real-time without restarting the application.

**Key Architectural Changes:**

This phase transforms the framework from a single-comparison tool into a true multi-comparison platform with seamless UI-based switching.

**Actionable Steps:**

1.  **Create ComparisonManager Class:**
    *   Add `framework/comparison_manager.py` to handle multiple comparison configurations
    *   Responsible for discovering, loading, and validating all available comparisons
    *   Manages orchestrator lifecycle (keeping multiple orchestrators running)
    
    ```python
    class ComparisonManager:
        def __init__(self):
            self.comparisons = {}  # comparison_id -> config
            self.orchestrators = {}  # comparison_id -> {baseline, enhanced}
            self.active_comparison = None
        
        async def load_all_comparisons(self):
            # Scan configs/ directory for comparison_config.json files
            # Validate each configuration
            # Initialize orchestrators for valid comparisons
        
        async def switch_comparison(self, comparison_id: str):
            # Update active comparison
            # Return comparison metadata for UI updates
    ```

2.  **Update Backend API Endpoints:**
    *   Add `/api/comparisons/list` - Returns available comparisons with UI metadata
    *   Add `/api/comparisons/current` - Returns active comparison configuration
    *   Add `/api/comparisons/switch/{comparison_id}` - Switches active comparison
    *   Modify existing query endpoints to use active comparison's orchestrators
    
3.  **Refactor comparison_app.py:**
    *   Replace global orchestrator variables with ComparisonManager instance
    *   Update `lifespan` function to initialize ComparisonManager
    *   Modify query processing to route to active comparison's orchestrators
    
4.  **Update Conversation Thread Scoping:**
    *   Change thread ID format to: `{comparison_id}_{orchestrator_type}_{thread_id}`
    *   Examples: `sqlite_baseline_abc123`, `sequential_enhanced_def456`
    *   This prevents cross-contamination between comparison types
    *   Maintain separate conversation histories per comparison type

5.  **Add Frontend Comparison Selector:**
    *   Add dropdown or radio button UI for comparison selection
    *   Fetch available comparisons on page load via `/api/comparisons/list`
    *   Update UI titles, colors, and icons dynamically when switching
    *   Handle loading states during comparison switches
    
    **Example Frontend Changes:**
    ```html
    <div class="comparison-selector">
        <label>Select Comparison:</label>
        <select id="comparisonSelector">
            <option value="sqlite">SQLite MCP Comparison</option>
            <option value="sequential">Sequential Thinking MCP</option>
        </select>
    </div>
    ```

**Development Considerations:**

1.  **Resource Management:** Keep all orchestrators running for better UX (accept memory cost vs. startup delays)
2.  **State Isolation:** Each comparison maintains separate conversation histories
3.  **Configuration Validation:** All configurations validated at startup; invalid ones excluded from UI
4.  **Graceful Switching:** Handle active conversations appropriately when user switches comparisons

**Benefits:**
- Real-time switching between MCP types without application restart
- Better user experience with immediate visual feedback  
- True demonstration of framework's MCP-agnostic design
- Single running instance handles multiple MCP comparison scenarios

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
3.  **Test:** The user runs `python start.py`.
4.  **Automate:** The framework automatically fetches the latest code from GitHub, sets up the environment, and starts the comparison UI.
5.  **Evaluate:** The user can immediately begin comparing the stable baseline against the latest development version, with zero manual intervention.