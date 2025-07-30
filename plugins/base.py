#!/usr/bin/env python3
"""
Vizor Plugin Base Class
Base class for all Vizor plugins
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time

@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str  # threat_intel, scanner, enrichment, etc.
    methods: List[str]
    dependencies: List[str]
    config_schema: Optional[Dict] = None
    created_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()

class VizorPlugin(ABC):
    """
    Base class for all Vizor plugins
    
    Plugins should inherit from this class and implement
    the required methods for their specific functionality.
    """
    
    def __init__(self, config, plugin_path: Optional[Path] = None):
        self.config = config
        self.plugin_path = plugin_path
        self.metadata = self._load_metadata()
        self._initialized = False
    
    def _load_metadata(self) -> PluginMetadata:
        """Load plugin metadata from plugin.json or use defaults"""
        if self.plugin_path and (self.plugin_path / "plugin.json").exists():
            with open(self.plugin_path / "plugin.json", 'r') as f:
                data = json.load(f)
                return PluginMetadata(**data)
        else:
            # Default metadata - should be overridden by subclasses
            return PluginMetadata(
                name=self.__class__.__name__,
                version="1.0.0",
                description="Base plugin",
                author="Unknown",
                plugin_type="base",
                methods=[],
                dependencies=[]
            )
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_methods(self) -> List[str]:
        """
        Get list of available methods
        
        Returns:
            List of method names that can be called
        """
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return self.metadata
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized"""
        return self._initialized
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate plugin configuration
        
        Returns:
            Dict with validation results
        """
        return {
            'valid': True,
            'errors': [],
            'warnings': []
        }
    
    def cleanup(self):
        """Cleanup plugin resources"""
        pass

class ThreatIntelPlugin(VizorPlugin):
    """Base class for threat intelligence plugins"""
    
    def __init__(self, config, plugin_path: Optional[Path] = None):
        super().__init__(config, plugin_path)
        self.metadata.plugin_type = "threat_intel"
    
    @abstractmethod
    def enrich_ioc(self, ioc: str, ioc_type: str) -> Dict[str, Any]:
        """
        Enrich an indicator of compromise
        
        Args:
            ioc: The IOC value (hash, IP, domain, etc.)
            ioc_type: Type of IOC (hash, ip, domain, url)
            
        Returns:
            Enrichment data
        """
        pass
    
    @abstractmethod
    def search_threats(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for threat information
        
        Args:
            query: Search query
            
        Returns:
            List of threat information
        """
        pass

class ScannerPlugin(VizorPlugin):
    """Base class for scanning plugins"""
    
    def __init__(self, config, plugin_path: Optional[Path] = None):
        super().__init__(config, plugin_path)
        self.metadata.plugin_type = "scanner"
    
    @abstractmethod
    def scan_target(self, target: str, scan_type: str) -> Dict[str, Any]:
        """
        Scan a target for security issues
        
        Args:
            target: Target to scan
            scan_type: Type of scan to perform
            
        Returns:
            Scan results
        """
        pass

class EnrichmentPlugin(VizorPlugin):
    """Base class for data enrichment plugins"""
    
    def __init__(self, config, plugin_path: Optional[Path] = None):
        super().__init__(config, plugin_path)
        self.metadata.plugin_type = "enrichment"
    
    @abstractmethod
    def enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich data with additional information
        
        Args:
            data: Data to enrich
            
        Returns:
            Enriched data
        """
        pass 