#!/usr/bin/env python3
"""
Vizor Configuration Settings
Manages configuration, privacy settings, and environment variables
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import json

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    default_model: str = "mistral"
    fallback_models: List[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = ["phi3", "deepseek-coder", "wizardcoder"]

@dataclass
class VectorConfig:
    """Configuration for vector store"""
    provider: str = "chromadb"
    collection_name: str = "vizor_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_documents: int = 10000
    similarity_threshold: float = 0.7

@dataclass
class PrivacyConfig:
    """Privacy and security configuration"""
    telemetry_enabled: bool = False
    cloud_fallback_enabled: bool = False
    data_retention_days: int = 90
    encrypt_local_data: bool = True
    log_queries: bool = True
    log_responses: bool = False


@dataclass
class LearningConfig:
    """Learning and adaptation configuration"""
    auto_learn_enabled: bool = True
    learning_sources: List[str] = None
    update_interval_hours: int = 24
    confidence_threshold: float = 0.7
    max_gap_memory: int = 1000
    
    def __post_init__(self):
        if self.learning_sources is None:
            self.learning_sources = [
                "cisa.gov",
                "mitre.org", 
                "nvd.nist.gov",
                "threatpost.com"
            ]

class VizorConfig:
    """Main configuration manager for Vizor"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.data_dir = self.config_path.parent / "data"
        self.plugins_dir = self.config_path.parent / "plugins"
        self.logs_dir = self.config_path.parent / "logs"
        
        # Load environment variables
        self._load_env()
        
        # Initialize configuration
        self.model_config = ModelConfig()
        self.vector_config = VectorConfig()
        self.privacy_config = PrivacyConfig()
        self.learning_config = LearningConfig()
        
        # Runtime settings
        self.dry_run = False
        self.verbose = False
        
        # Load existing config if it exists
        if self.config_path.exists():
            self.load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration path"""
        home = Path.home()
        config_dir = home / ".vizor"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "vizor.yaml"
    
    def _load_env(self):
        """Load environment variables from .env file"""
        env_file = self.config_path.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                # Update configurations from file
                if 'model' in config_data:
                    self.model_config = ModelConfig(**config_data['model'])
                
                if 'vector' in config_data:
                    self.vector_config = VectorConfig(**config_data['vector'])
                
                if 'privacy' in config_data:
                    self.privacy_config = PrivacyConfig(**config_data['privacy'])
                
                if 'learning' in config_data:
                    self.learning_config = LearningConfig(**config_data['learning'])
                    
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    def save_config(self):
        """Save current configuration to YAML file"""
        config_data = {
            'model': asdict(self.model_config),
            'vector': asdict(self.vector_config),
            'privacy': asdict(self.privacy_config),
            'learning': asdict(self.learning_config),
            'version': '1.0.0',
            'last_updated': str(Path(__file__).stat().st_mtime)
        }
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def load_custom_config(self, config_path: str):
        """Load configuration from custom path"""
        self.config_path = Path(config_path)
        self.load_config()
    
    def initialize(self, force: bool = False):
        """Initialize Vizor configuration and directories"""
        if self.config_path.exists() and not force:
            raise Exception(f"Configuration already exists at {self.config_path}. Use --force to overwrite.")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default .env file
        env_file = self.config_path.parent / ".env"
        if not env_file.exists() or force:
            env_content = """# Vizor Environment Variables
# API Keys (optional - only if using cloud fallback)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Threat Intelligence API Keys (optional)
# VIRUSTOTAL_API_KEY=your_vt_key_here
# GREYNOISE_API_KEY=your_greynoise_key_here

# Email Configuration (optional - for briefing delivery)
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your_email@gmail.com
# SMTP_PASSWORD=your_app_password

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
        
        # Save default configuration
        self.save_config()
        
        # Create initial gap memory file
        gap_file = self.data_dir / "gap_memory.json"
        if not gap_file.exists():
            with open(gap_file, 'w') as f:
                json.dump({}, f)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            # Check Ollama connection
            import ollama
            try:
                models = ollama.list()
                llm_status = f"Connected ({len(models['models'])} models)"
                available_models = [m['name'] for m in models['models']]
            except:
                llm_status = "Disconnected"
                available_models = []
            
            # Check vector store
            vector_status = "Not initialized"
            try:
                if (self.data_dir / "chroma.db").exists():
                    vector_status = "Initialized"
            except:
                pass
            
            # Check plugins
            plugin_count = 0
            try:
                if self.plugins_dir.exists():
                    plugin_count = len([f for f in self.plugins_dir.iterdir() if f.is_dir()])
            except:
                pass
            
            # Check memory
            memory_status = "Empty"
            try:
                gap_file = self.data_dir / "gap_memory.json"
                if gap_file.exists():
                    with open(gap_file, 'r') as f:
                        gaps = json.load(f)
                        if gaps:
                            memory_status = f"{len(gaps)} knowledge gaps tracked"
                        else:
                            memory_status = "No gaps tracked"
            except:
                pass
            
            # Overall health check
            healthy = (
                llm_status != "Disconnected" and
                len(available_models) > 0 and
                self.config_path.exists()
            )
            
            return {
                'config_status': 'Loaded' if self.config_path.exists() else 'Not found',
                'llm_status': llm_status,
                'available_models': available_models,
                'vector_status': vector_status,
                'plugin_count': plugin_count,
                'memory_status': memory_status,
                'healthy': healthy,
                'data_dir': str(self.data_dir),
                'config_path': str(self.config_path)
            }
            
        except Exception as e:
            return {
                'config_status': 'Error',
                'llm_status': 'Error',
                'vector_status': 'Error',
                'plugin_count': 0,
                'memory_status': 'Error',
                'healthy': False,
                'error': str(e)
            }
    
    def set_dry_run(self, enabled: bool):
        """Enable or disable dry run mode"""
        self.dry_run = enabled
    
    def set_verbose(self, enabled: bool):
        """Enable or disable verbose logging"""
        self.verbose = enabled
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service from environment variables"""
        key_mapping = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'virustotal': 'VIRUSTOTAL_API_KEY',
            'greynoise': 'GREYNOISE_API_KEY'
        }
        
        env_var = key_mapping.get(service.lower())
        if env_var:
            return os.getenv(env_var)
        return None
    
    def get_ollama_host(self) -> str:
        """Get Ollama host URL"""
        return os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    def is_cloud_fallback_enabled(self) -> bool:
        """Check if cloud fallback is enabled and configured"""
        return (
            self.privacy_config.cloud_fallback_enabled and
            (self.get_api_key('openai') or self.get_api_key('anthropic'))
        )
    

