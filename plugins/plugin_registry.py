#!/usr/bin/env python3
"""
Vizor Plugin Registry
Manages plugin discovery, loading, and lifecycle
"""

import importlib.util
import json
import time
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import sys

from .base import VizorPlugin, PluginMetadata

class PluginRegistry:
    """
    Registry for managing Vizor plugins
    
    Handles plugin discovery, loading, validation, and lifecycle management.
    """
    
    def __init__(self, config):
        self.config = config
        self.plugins_dir = config.plugins_dir
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.plugins_dir / "registry.json"
        self.loaded_plugins: Dict[str, VizorPlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load plugin registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    for plugin_name, metadata_dict in registry_data.items():
                        self.plugin_metadata[plugin_name] = PluginMetadata(**metadata_dict)
            except Exception as e:
                print(f"Warning: Could not load plugin registry: {e}")
    
    def _save_registry(self):
        """Save plugin registry to file"""
        registry_data = {}
        for name, metadata in self.plugin_metadata.items():
            registry_data[name] = {
                'name': metadata.name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'plugin_type': metadata.plugin_type,
                'methods': metadata.methods,
                'dependencies': metadata.dependencies,
                'config_schema': metadata.config_schema,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at
            }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def discover_plugins(self) -> List[Path]:
        """Discover available plugins in plugins directory"""
        plugins = []
        
        if not self.plugins_dir.exists():
            return plugins
        
        for item in self.plugins_dir.iterdir():
            if item.is_dir() and (item / "plugin.json").exists():
                plugins.append(item)
            elif item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                plugins.append(item)
        
        return plugins
    
    def load_plugin(self, plugin_path: Path) -> Optional[VizorPlugin]:
        """
        Load a plugin from path
        
        Args:
            plugin_path: Path to plugin directory or file
            
        Returns:
            Loaded plugin instance or None if failed
        """
        try:
            if plugin_path.is_dir():
                return self._load_directory_plugin(plugin_path)
            elif plugin_path.is_file():
                return self._load_file_plugin(plugin_path)
            else:
                raise ValueError(f"Invalid plugin path: {plugin_path}")
                
        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {e}")
            return None
    
    def _load_directory_plugin(self, plugin_dir: Path) -> Optional[VizorPlugin]:
        """Load plugin from directory structure"""
        plugin_json = plugin_dir / "plugin.json"
        main_py = plugin_dir / "main.py"
        
        if not plugin_json.exists() or not main_py.exists():
            return None
        
        # Load metadata
        with open(plugin_json, 'r') as f:
            metadata_dict = json.load(f)
        
        # Load plugin module
        spec = importlib.util.spec_from_file_location(
            f"plugin_{metadata_dict['name']}", 
            main_py
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin class
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, VizorPlugin) and 
                attr != VizorPlugin):
                plugin_class = attr
                break
        
        if not plugin_class:
            return None
        
        # Create plugin instance
        plugin = plugin_class(self.config, plugin_dir)
        plugin.metadata = PluginMetadata(**metadata_dict)
        
        return plugin
    
    def _load_file_plugin(self, plugin_file: Path) -> Optional[VizorPlugin]:
        """Load plugin from single file"""
        spec = importlib.util.spec_from_file_location(
            f"plugin_{plugin_file.stem}", 
            plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin class
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, VizorPlugin) and 
                attr != VizorPlugin):
                plugin_class = attr
                break
        
        if not plugin_class:
            return None
        
        # Create plugin instance
        plugin = plugin_class(self.config, plugin_file)
        
        return plugin
    
    def register_plugin(self, plugin_path: Path) -> Dict[str, Any]:
        """
        Register a plugin in the registry
        
        Args:
            plugin_path: Path to plugin
            
        Returns:
            Registration result
        """
        try:
            # Load plugin
            plugin = self.load_plugin(plugin_path)
            if not plugin:
                return {
                    'success': False,
                    'error': 'Failed to load plugin'
                }
            
            # Initialize plugin
            if not plugin.initialize():
                return {
                    'success': False,
                    'error': 'Plugin initialization failed'
                }
            
            # Validate plugin
            validation = plugin.validate_config()
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Plugin validation failed: {validation['errors']}"
                }
            
            # Register plugin
            plugin_name = plugin.metadata.name
            self.loaded_plugins[plugin_name] = plugin
            self.plugin_metadata[plugin_name] = plugin.metadata
            
            # Update metadata
            self.plugin_metadata[plugin_name].updated_at = time.time()
            
            # Save registry
            self._save_registry()
            
            return {
                'success': True,
                'name': plugin_name,
                'version': plugin.metadata.version,
                'methods': plugin.get_methods(),
                'plugin_type': plugin.metadata.plugin_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin"""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            plugin.cleanup()
            del self.loaded_plugins[plugin_name]
            
            if plugin_name in self.plugin_metadata:
                del self.plugin_metadata[plugin_name]
            
            self._save_registry()
            return True
        
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[VizorPlugin]:
        """Get a loaded plugin by name"""
        return self.loaded_plugins.get(plugin_name)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins"""
        plugins = []
        
        for name, metadata in self.plugin_metadata.items():
            plugin_info = {
                'name': name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'plugin_type': metadata.plugin_type,
                'methods': metadata.methods,
                'status': 'active' if name in self.loaded_plugins else 'inactive',
                'path': str(self.plugins_dir / name) if name in self.loaded_plugins else None
            }
            plugins.append(plugin_info)
        
        return plugins
    
    def test_plugin(self, plugin_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """
        Test a plugin
        
        Args:
            plugin_path: Path to plugin
            verbose: Verbose output
            
        Returns:
            Test results
        """
        try:
            # Load plugin
            plugin = self.load_plugin(plugin_path)
            if not plugin:
                return {
                    'success': False,
                    'error': 'Failed to load plugin'
                }
            
            # Test initialization
            if not plugin.initialize():
                return {
                    'success': False,
                    'error': 'Plugin initialization failed'
                }
            
            # Test validation
            validation = plugin.validate_config()
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Plugin validation failed: {validation['errors']}"
                }
            
            # Test methods
            methods = plugin.get_methods()
            method_tests = {}
            
            for method in methods:
                try:
                    # Basic method existence test
                    if hasattr(plugin, method):
                        method_tests[method] = 'available'
                    else:
                        method_tests[method] = 'missing'
                except Exception as e:
                    method_tests[method] = f'error: {str(e)}'
            
            # Cleanup
            plugin.cleanup()
            
            return {
                'success': True,
                'tests_run': len(methods),
                'method_tests': method_tests,
                'warnings': validation.get('warnings', []),
                'errors': validation.get('errors', [])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tests_run': 0,
                'method_tests': {},
                'warnings': [],
                'errors': [str(e)]
            }
    
    def check_updates(self, plugin_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Check for plugin updates
        
        Args:
            plugin_name: Specific plugin to check (check all if None)
            
        Returns:
            Update information
        """
        # This would implement update checking logic
        # For now, return empty result
        return {
            'updates_available': False,
            'updates': []
        }
    
    def apply_updates(self, updates: List[Dict]) -> Dict[str, Any]:
        """
        Apply plugin updates
        
        Args:
            updates: List of updates to apply
            
        Returns:
            Update results
        """
        # This would implement update application logic
        # For now, return empty result
        return {
            'successful': 0,
            'failed': 0,
            'errors': []
        } 