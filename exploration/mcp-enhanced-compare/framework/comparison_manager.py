"""Comparison Manager for handling multiple MCP comparison configurations."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .orchestrator.simple_orchestrator import SimpleMCPOrchestrator

logger = logging.getLogger(__name__)


class ComparisonManager:
    """Manages multiple MCP comparison configurations and their orchestrators."""
    
    def __init__(self, config_root: str = "configs"):
        self.config_root = Path(config_root)
        self.comparisons: Dict[str, Dict[str, Any]] = {}
        self.orchestrators: Dict[str, Dict[str, SimpleMCPOrchestrator]] = {}
        self.active_comparison: Optional[str] = None
        
    async def load_all_comparisons(self) -> None:
        """Scan configs directory and load all valid comparison configurations."""
        logger.info(f"Scanning for comparison configurations in {self.config_root}")
        
        if not self.config_root.exists():
            logger.warning(f"Config directory {self.config_root} does not exist")
            return
            
        # Find all comparison_config.json files (excluding template directories)
        config_files = list(self.config_root.glob("*/comparison_config.json"))
        
        for config_file in config_files:
            comparison_id = config_file.parent.name
            
            # Skip template directories
            if comparison_id.lower() in ['template', 'templates']:
                logger.info(f"Skipping template directory: {comparison_id}")
                continue
            try:
                logger.info(f"Loading comparison configuration: {comparison_id}")
                
                # Load and validate configuration
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Validate required fields
                self._validate_comparison_config(config)
                
                # Store configuration
                self.comparisons[comparison_id] = config
                
                # Initialize orchestrators for this comparison
                await self._initialize_orchestrators(comparison_id, config)
                
                logger.info(f"Successfully loaded comparison: {comparison_id}")
                
            except Exception as e:
                logger.error(f"Failed to load comparison {comparison_id}: {e}")
                # Remove failed comparison from list
                if comparison_id in self.comparisons:
                    del self.comparisons[comparison_id]
                continue
        
        # Set first available comparison as active if none set
        if not self.active_comparison and self.comparisons:
            self.active_comparison = list(self.comparisons.keys())[0]
            logger.info(f"Set active comparison to: {self.active_comparison}")
    
    def _validate_comparison_config(self, config: Dict[str, Any]) -> None:
        """Validate that a comparison configuration has all required fields."""
        required_fields = ["comparison_name", "baseline", "enhanced"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate baseline and enhanced configs
        for variant in ["baseline", "enhanced"]:
            variant_config = config[variant]
            if "mcp_server" not in variant_config:
                raise ValueError(f"Missing mcp_server config in {variant}")
            if "ui" not in variant_config:
                raise ValueError(f"Missing ui config in {variant}")
    
    async def _initialize_orchestrators(self, comparison_id: str, config: Dict[str, Any]) -> None:
        """Initialize baseline and enhanced orchestrators for a comparison."""
        orchestrators = {}
        
        try:
            # Initialize baseline orchestrator
            baseline_config = config["baseline"]
            baseline_orchestrator = SimpleMCPOrchestrator(baseline_config)
            await baseline_orchestrator.initialize()
            orchestrators["baseline"] = baseline_orchestrator
            
            # Initialize enhanced orchestrator
            enhanced_config = config["enhanced"]
            enhanced_orchestrator = SimpleMCPOrchestrator(enhanced_config)
            await enhanced_orchestrator.initialize()
            orchestrators["enhanced"] = enhanced_orchestrator
            
            self.orchestrators[comparison_id] = orchestrators
            logger.info(f"Initialized orchestrators for comparison: {comparison_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrators for {comparison_id}: {e}")
            # Clean up any partially initialized orchestrators
            for orchestrator in orchestrators.values():
                try:
                    await orchestrator.cleanup()
                except Exception:
                    pass
            raise
    
    async def switch_comparison(self, comparison_id: str) -> Dict[str, Any]:
        """Switch to a different comparison and return its configuration."""
        if comparison_id not in self.comparisons:
            raise ValueError(f"Unknown comparison: {comparison_id}")
        
        self.active_comparison = comparison_id
        logger.info(f"Switched to comparison: {comparison_id}")
        
        return self.get_current_comparison_config()
    
    def get_available_comparisons(self) -> List[Dict[str, Any]]:
        """Get list of available comparisons with their metadata."""
        comparisons = []
        for comparison_id, config in self.comparisons.items():
            comparisons.append({
                "id": comparison_id,
                "name": config.get("comparison_name", comparison_id),
                "description": config.get("comparison_description", ""),
                "baseline_ui": config["baseline"]["ui"],
                "enhanced_ui": config["enhanced"]["ui"]
            })
        return comparisons
    
    def get_current_comparison_config(self) -> Dict[str, Any]:
        """Get the configuration for the currently active comparison."""
        if not self.active_comparison:
            raise ValueError("No active comparison")
        
        config = self.comparisons[self.active_comparison].copy()
        config["comparison_id"] = self.active_comparison
        return config
    
    def get_active_orchestrators(self) -> Dict[str, SimpleMCPOrchestrator]:
        """Get the orchestrators for the currently active comparison."""
        if not self.active_comparison:
            raise ValueError("No active comparison")
        
        return self.orchestrators[self.active_comparison]
    
    async def cleanup(self) -> None:
        """Clean up all orchestrators."""
        logger.info("Cleaning up all orchestrators")
        
        for comparison_id, orchestrators in self.orchestrators.items():
            for variant, orchestrator in orchestrators.items():
                try:
                    await orchestrator.cleanup()
                    logger.info(f"Cleaned up {variant} orchestrator for {comparison_id}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {variant} orchestrator for {comparison_id}: {e}")
        
        self.orchestrators.clear()
        self.comparisons.clear()
        self.active_comparison = None