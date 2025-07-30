#!/usr/bin/env python3
"""
Vizor API Wrapper Generator
Generates Python wrappers from OpenAPI specifications
"""

import json
import re
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import time
from dataclasses import dataclass

@dataclass
class APIEndpoint:
    """Represents an API endpoint"""
    path: str
    method: str
    operation_id: str
    summary: str
    parameters: List[Dict]
    responses: Dict
    tags: List[str]

@dataclass
class APISpec:
    """Represents an OpenAPI specification"""
    info: Dict
    servers: List[Dict]
    paths: Dict
    components: Dict
    security: List[Dict]

class WrapperGenerator:
    """
    Generates Python API wrappers from OpenAPI specifications
    
    Supports both OpenAPI 3.0 and Swagger 2.0 specifications
    with automatic authentication handling and error management.
    """
    
    def __init__(self, config):
        self.config = config
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
    
    def generate_from_url(
        self, 
        url: str, 
        wrapper_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        auth_type: str = "none",
        include_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Generate API wrapper from URL
        
        Args:
            url: URL to OpenAPI spec or API documentation
            wrapper_name: Custom name for the wrapper
            output_dir: Output directory for generated files
            auth_type: Authentication type (none, api_key, bearer, basic)
            include_tests: Whether to generate test cases
            
        Returns:
            Generation results
        """
        try:
            # Fetch OpenAPI specification
            spec = self._fetch_openapi_spec(url)
            if not spec:
                raise Exception("Could not fetch OpenAPI specification")
            
            # Parse specification
            api_spec = self._parse_openapi_spec(spec)
            
            # Generate wrapper name if not provided
            if not wrapper_name:
                wrapper_name = self._generate_wrapper_name(api_spec.info.get('title', 'api'))
            
            # Determine output directory
            if not output_dir:
                output_dir = self.config.plugins_dir / wrapper_name
            else:
                output_dir = Path(output_dir)
            
            # Generate wrapper files
            result = self._generate_wrapper_files(
                api_spec=api_spec,
                wrapper_name=wrapper_name,
                output_dir=output_dir,
                auth_type=auth_type,
                include_tests=include_tests
            )
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'wrapper_path': None,
                'endpoint_count': 0,
                'method_count': 0
            }
    
    def _fetch_openapi_spec(self, url: str) -> Optional[Dict]:
        """Fetch OpenAPI specification from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to parse as JSON first
            try:
                return response.json()
            except:
                pass
            
            # Try to parse as YAML
            try:
                return yaml.safe_load(response.text)
            except:
                pass
            
            # Try to extract from HTML (for documentation pages)
            return self._extract_spec_from_html(response.text)
            
        except Exception as e:
            print(f"Error fetching OpenAPI spec: {e}")
            return None
    
    def _extract_spec_from_html(self, html_content: str) -> Optional[Dict]:
        """Extract OpenAPI spec from HTML documentation"""
        # Look for common patterns in documentation pages
        patterns = [
            r'<script[^>]*id="swagger-data"[^>]*>(.*?)</script>',
            r'window\.swaggerSpec\s*=\s*({.*?});',
            r'window\.openapiSpec\s*=\s*({.*?});'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue
        
        return None
    
    def _parse_openapi_spec(self, spec: Dict) -> APISpec:
        """Parse OpenAPI specification"""
        return APISpec(
            info=spec.get('info', {}),
            servers=spec.get('servers', []),
            paths=spec.get('paths', {}),
            components=spec.get('components', {}),
            security=spec.get('security', [])
        )
    
    def _generate_wrapper_name(self, title: str) -> str:
        """Generate wrapper name from API title"""
        # Clean and convert title to valid Python identifier
        name = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        name = re.sub(r'\s+', '_', name).lower()
        name = re.sub(r'^[0-9]', 'api_', name)
        
        if not name:
            name = 'api_wrapper'
        
        return name
    
    def _generate_wrapper_files(
        self,
        api_spec: APISpec,
        wrapper_name: str,
        output_dir: Path,
        auth_type: str,
        include_tests: bool
    ) -> Dict[str, Any]:
        """Generate wrapper files"""
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract endpoints
        endpoints = self._extract_endpoints(api_spec.paths)
        
        # Generate main wrapper file
        wrapper_content = self._generate_wrapper_class(
            api_spec=api_spec,
            wrapper_name=wrapper_name,
            endpoints=endpoints,
            auth_type=auth_type
        )
        
        wrapper_file = output_dir / "main.py"
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_content)
        
        # Generate plugin.json
        plugin_json = self._generate_plugin_json(
            api_spec=api_spec,
            wrapper_name=wrapper_name,
            endpoints=endpoints
        )
        
        plugin_file = output_dir / "plugin.json"
        with open(plugin_file, 'w') as f:
            json.dump(plugin_json, f, indent=2)
        
        # Generate README
        readme_content = self._generate_readme(
            api_spec=api_spec,
            wrapper_name=wrapper_name,
            endpoints=endpoints
        )
        
        readme_file = output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Generate tests if requested
        test_files = []
        if include_tests:
            test_content = self._generate_tests(
                wrapper_name=wrapper_name,
                endpoints=endpoints
            )
            
            test_file = output_dir / "test_wrapper.py"
            with open(test_file, 'w') as f:
                f.write(test_content)
            test_files.append(str(test_file))
        
        return {
            'success': True,
            'wrapper_path': str(output_dir),
            'endpoint_count': len(endpoints),
            'method_count': len(endpoints),
            'files_created': [
                str(wrapper_file),
                str(plugin_file),
                str(readme_file)
            ] + test_files
        }
    
    def _extract_endpoints(self, paths: Dict) -> List[APIEndpoint]:
        """Extract endpoints from OpenAPI paths"""
        endpoints = []
        
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        operation_id=operation.get('operationId', f"{method}_{path.replace('/', '_').strip('_')}"),
                        summary=operation.get('summary', ''),
                        parameters=operation.get('parameters', []),
                        responses=operation.get('responses', {}),
                        tags=operation.get('tags', [])
                    )
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _generate_wrapper_class(
        self,
        api_spec: APISpec,
        wrapper_name: str,
        endpoints: List[APIEndpoint],
        auth_type: str
    ) -> str:
        """Generate the main wrapper class"""
        
        class_name = ''.join(word.capitalize() for word in wrapper_name.split('_'))
        
        # Generate imports
        imports = [
            "import requests",
            "import json",
            "from typing import Dict, List, Any, Optional",
            "from pathlib import Path",
            "import time",
            "from .base import VizorPlugin"
        ]
        
        # Generate authentication setup
        auth_setup = self._generate_auth_setup(auth_type)
        
        # Generate methods
        methods = []
        for endpoint in endpoints:
            method = self._generate_endpoint_method(endpoint)
            methods.append(method)
        
        # Generate class template
        template = f'''#!/usr/bin/env python3
"""
{api_spec.info.get('title', 'API')} Wrapper
Generated by Vizor Wrapper Generator

{api_spec.info.get('description', '')}
"""

{chr(10).join(imports)}

class {class_name}Wrapper(VizorPlugin):
    """
    {api_spec.info.get('title', 'API')} Wrapper
    
    {api_spec.info.get('description', 'Generated API wrapper')}
    """
    
    def __init__(self, config, plugin_path=None):
        super().__init__(config, plugin_path)
        self.base_url = "{api_spec.servers[0].get('url', 'https://api.example.com') if api_spec.servers else 'https://api.example.com'}"
        self.session = requests.Session()
        {auth_setup}
    
    def initialize(self) -> bool:
        """Initialize the wrapper"""
        try:
            # Test connection
            response = self.session.get(f"{{self.base_url}}/health", timeout=10)
            return response.status_code == 200
        except:
            # If health endpoint doesn't exist, assume it's working
            return True
    
    def get_methods(self) -> List[str]:
        """Get available methods"""
        return [
{chr(10).join(f'            "{endpoint.operation_id}",' for endpoint in endpoints)}
        ]
    
    {chr(10).join(methods)}
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        try:
            url = f"{{self.base_url}}{{endpoint}}"
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {{"error": str(e), "status_code": getattr(e.response, 'status_code', None)}}
        except json.JSONDecodeError:
            return {{"error": "Invalid JSON response", "raw_response": response.text}}
'''
        
        return template
    
    def _generate_auth_setup(self, auth_type: str) -> str:
        """Generate authentication setup code"""
        if auth_type == "api_key":
            return '''self.api_key = config.get_api_key('api')
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})'''
        elif auth_type == "bearer":
            return '''self.bearer_token = config.get_api_key('bearer')
        if self.bearer_token:
            self.session.headers.update({'Authorization': f'Bearer {self.bearer_token}'})'''
        elif auth_type == "basic":
            return '''self.username = config.get_api_key('username')
        self.password = config.get_api_key('password')
        if self.username and self.password:
            self.session.auth = (self.username, self.password)'''
        else:
            return "# No authentication required"
    
    def _generate_endpoint_method(self, endpoint: APIEndpoint) -> str:
        """Generate method for an endpoint"""
        method_name = endpoint.operation_id
        docstring = endpoint.summary or f"Call {endpoint.method} {endpoint.path}"
        
        # Generate parameters
        params = []
        for param in endpoint.parameters:
            if param.get('in') == 'path':
                params.append(param['name'])
            elif param.get('in') == 'query':
                params.append(f"{param['name']}: Optional[str] = None")
        
        param_str = ', '.join(params) if params else ''
        
        # Generate method body
        body = f'''        """{docstring}"""
        endpoint = "{endpoint.path}"
        
        # Build query parameters
        params = {{}}
        {chr(10).join(f'        if {param["name"]} is not None:' for param in endpoint.parameters if param.get('in') == 'query')}
        {chr(10).join(f'            params["{param["name"]}"] = {param["name"]}' for param in endpoint.parameters if param.get('in') == 'query')}
        
        return self._make_request("{endpoint.method}", endpoint, params=params)'''
        
        return f'''    def {method_name}(self, {param_str}) -> Dict[str, Any]:
{body}'''
    
    def _generate_plugin_json(
        self,
        api_spec: APISpec,
        wrapper_name: str,
        endpoints: List[APIEndpoint]
    ) -> Dict[str, Any]:
        """Generate plugin.json metadata"""
        return {
            "name": wrapper_name,
            "version": "1.0.0",
            "description": api_spec.info.get('description', f"Generated wrapper for {api_spec.info.get('title', 'API')}"),
            "author": "Vizor Wrapper Generator",
            "plugin_type": "api_wrapper",
            "methods": [endpoint.operation_id for endpoint in endpoints],
            "dependencies": ["requests"],
            "config_schema": {
                "api_key": {"type": "string", "required": False},
                "base_url": {"type": "string", "required": False}
            },
            "created_at": time.time(),
            "updated_at": time.time()
        }
    
    def _generate_readme(
        self,
        api_spec: APISpec,
        wrapper_name: str,
        endpoints: List[APIEndpoint]
    ) -> str:
        """Generate README for the wrapper"""
        return f'''# {api_spec.info.get('title', 'API')} Wrapper

Generated by Vizor Wrapper Generator

## Description

{api_spec.info.get('description', 'No description available')}

## Installation

This wrapper is automatically installed when registered with Vizor.

## Usage

```python
from vizor import Vizor

# The wrapper is automatically available after registration
# Use it through Vizor's plugin system
```

## Available Methods

{chr(10).join(f'- `{endpoint.operation_id}` - {endpoint.summary or f"{endpoint.method} {endpoint.path}"}' for endpoint in endpoints)}

## Configuration

Configure API credentials in your Vizor configuration file.
'''
    
    def _generate_tests(
        self,
        wrapper_name: str,
        endpoints: List[APIEndpoint]
    ) -> str:
        """Generate test file for the wrapper"""
        class_name = ''.join(word.capitalize() for word in wrapper_name.split('_'))
        
        return f'''#!/usr/bin/env python3
"""
Tests for {class_name}Wrapper
"""

import pytest
from unittest.mock import Mock, patch
from .main import {class_name}Wrapper

class Test{class_name}Wrapper:
    """Test cases for {class_name}Wrapper"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = Mock()
        self.wrapper = {class_name}Wrapper(self.config)
    
    def test_initialization(self):
        """Test wrapper initialization"""
        assert self.wrapper is not None
        assert hasattr(self.wrapper, 'base_url')
    
    def test_get_methods(self):
        """Test get_methods returns expected methods"""
        methods = self.wrapper.get_methods()
        assert isinstance(methods, list)
        {chr(10).join(f'        assert "{endpoint.operation_id}" in methods' for endpoint in endpoints)}
    
    # Add more specific tests for each endpoint as needed
'''
    
    def generate_plugin_template(
        self,
        template_type: str,
        plugin_name: str,
        description: Optional[str] = None,
        author: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate plugin template
        
        Args:
            template_type: Type of template (basic, threat_intel, scanner, enrichment)
            plugin_name: Name for the plugin
            description: Plugin description
            author: Plugin author
            
        Returns:
            Generation results
        """
        try:
            output_dir = self.config.plugins_dir / plugin_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate template based on type
            if template_type == "basic":
                result = self._generate_basic_template(plugin_name, description, author, output_dir)
            elif template_type == "threat_intel":
                result = self._generate_threat_intel_template(plugin_name, description, author, output_dir)
            elif template_type == "scanner":
                result = self._generate_scanner_template(plugin_name, description, author, output_dir)
            elif template_type == "enrichment":
                result = self._generate_enrichment_template(plugin_name, description, author, output_dir)
            else:
                raise ValueError(f"Unknown template type: {template_type}")
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'plugin_path': None,
                'files': []
            }
    
    def _generate_basic_template(
        self,
        plugin_name: str,
        description: Optional[str],
        author: Optional[str],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate basic plugin template"""
        files = []
        
        # Generate main.py
        main_content = f'''#!/usr/bin/env python3
"""
{plugin_name} Plugin
{description or "Basic Vizor plugin template"}
"""

from typing import Dict, List, Any
from .base import VizorPlugin

class {plugin_name.replace('_', '').title()}Plugin(VizorPlugin):
    """
    {description or "Basic plugin template"}
    """
    
    def __init__(self, config, plugin_path=None):
        super().__init__(config, plugin_path)
        self.metadata.plugin_type = "basic"
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        self._initialized = True
        return True
    
    def get_methods(self) -> List[str]:
        """Get available methods"""
        return ["example_method"]
    
    def example_method(self, input_data: str) -> Dict[str, Any]:
        """
        Example method
        
        Args:
            input_data: Input data
            
        Returns:
            Processed result
        """
        return {{
            "input": input_data,
            "processed": True,
            "result": f"Processed: {{input_data}}"
        }}
'''
        
        main_file = output_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(main_content)
        files.append(str(main_file))
        
        # Generate plugin.json
        plugin_json = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": description or "Basic plugin template",
            "author": author or "Vizor Plugin Generator",
            "plugin_type": "basic",
            "methods": ["example_method"],
            "dependencies": [],
            "config_schema": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        plugin_file = output_dir / "plugin.json"
        with open(plugin_file, 'w') as f:
            json.dump(plugin_json, f, indent=2)
        files.append(str(plugin_file))
        
        # Generate README
        readme_content = f'''# {plugin_name} Plugin

{description or "Basic plugin template for Vizor"}

## Installation

This plugin is automatically installed when registered with Vizor.

## Usage

```python
# Use through Vizor's plugin system
```

## Methods

- `example_method(input_data)` - Example method that processes input data

## Configuration

No configuration required.
'''
        
        readme_file = output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        files.append(str(readme_file))
        
        return {
            'success': True,
            'plugin_path': str(output_dir),
            'files': files
        }
    
    def _generate_threat_intel_template(
        self,
        plugin_name: str,
        description: Optional[str],
        author: Optional[str],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate threat intelligence plugin template"""
        files = []
        
        # Generate main.py
        main_content = f'''#!/usr/bin/env python3
"""
{plugin_name} Threat Intelligence Plugin
{description or "Threat intelligence plugin template"}
"""

from typing import Dict, List, Any
from .base import ThreatIntelPlugin

class {plugin_name.replace('_', '').title()}ThreatIntelPlugin(ThreatIntelPlugin):
    """
    {description or "Threat intelligence plugin template"}
    """
    
    def __init__(self, config, plugin_path=None):
        super().__init__(config, plugin_path)
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        # Add your initialization logic here
        self._initialized = True
        return True
    
    def get_methods(self) -> List[str]:
        """Get available methods"""
        return ["enrich_ioc", "search_threats"]
    
    def enrich_ioc(self, ioc: str, ioc_type: str) -> Dict[str, Any]:
        """
        Enrich an indicator of compromise
        
        Args:
            ioc: The IOC value
            ioc_type: Type of IOC (hash, ip, domain, url)
            
        Returns:
            Enrichment data
        """
        # Add your IOC enrichment logic here
        return {{
            "ioc": ioc,
            "ioc_type": ioc_type,
            "enriched": True,
            "reputation": "unknown",
            "confidence": 0.0,
            "sources": []
        }}
    
    def search_threats(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for threat information
        
        Args:
            query: Search query
            
        Returns:
            List of threat information
        """
        # Add your threat search logic here
        return []
'''
        
        main_file = output_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(main_content)
        files.append(str(main_file))
        
        # Generate plugin.json
        plugin_json = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": description or "Threat intelligence plugin template",
            "author": author or "Vizor Plugin Generator",
            "plugin_type": "threat_intel",
            "methods": ["enrich_ioc", "search_threats"],
            "dependencies": [],
            "config_schema": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        plugin_file = output_dir / "plugin.json"
        with open(plugin_file, 'w') as f:
            json.dump(plugin_json, f, indent=2)
        files.append(str(plugin_file))
        
        return {
            'success': True,
            'plugin_path': str(output_dir),
            'files': files
        }
    
    def _generate_scanner_template(
        self,
        plugin_name: str,
        description: Optional[str],
        author: Optional[str],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate scanner plugin template"""
        files = []
        
        # Generate main.py
        main_content = f'''#!/usr/bin/env python3
"""
{plugin_name} Scanner Plugin
{description or "Scanner plugin template"}
"""

from typing import Dict, List, Any
from .base import ScannerPlugin

class {plugin_name.replace('_', '').title()}ScannerPlugin(ScannerPlugin):
    """
    {description or "Scanner plugin template"}
    """
    
    def __init__(self, config, plugin_path=None):
        super().__init__(config, plugin_path)
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        # Add your initialization logic here
        self._initialized = True
        return True
    
    def get_methods(self) -> List[str]:
        """Get available methods"""
        return ["scan_target"]
    
    def scan_target(self, target: str, scan_type: str) -> Dict[str, Any]:
        """
        Scan a target for security issues
        
        Args:
            target: Target to scan
            scan_type: Type of scan to perform
            
        Returns:
            Scan results
        """
        # Add your scanning logic here
        return {{
            "target": target,
            "scan_type": scan_type,
            "scanned": True,
            "threat_level": "unknown",
            "findings": [],
            "confidence": 0.0
        }}
'''
        
        main_file = output_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(main_content)
        files.append(str(main_file))
        
        # Generate plugin.json
        plugin_json = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": description or "Scanner plugin template",
            "author": author or "Vizor Plugin Generator",
            "plugin_type": "scanner",
            "methods": ["scan_target"],
            "dependencies": [],
            "config_schema": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        plugin_file = output_dir / "plugin.json"
        with open(plugin_file, 'w') as f:
            json.dump(plugin_json, f, indent=2)
        files.append(str(plugin_file))
        
        return {
            'success': True,
            'plugin_path': str(output_dir),
            'files': files
        }
    
    def _generate_enrichment_template(
        self,
        plugin_name: str,
        description: Optional[str],
        author: Optional[str],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate enrichment plugin template"""
        files = []
        
        # Generate main.py
        main_content = f'''#!/usr/bin/env python3
"""
{plugin_name} Enrichment Plugin
{description or "Enrichment plugin template"}
"""

from typing import Dict, List, Any
from .base import EnrichmentPlugin

class {plugin_name.replace('_', '').title()}EnrichmentPlugin(EnrichmentPlugin):
    """
    {description or "Enrichment plugin template"}
    """
    
    def __init__(self, config, plugin_path=None):
        super().__init__(config, plugin_path)
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        # Add your initialization logic here
        self._initialized = True
        return True
    
    def get_methods(self) -> List[str]:
        """Get available methods"""
        return ["enrich_data"]
    
    def enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich data with additional information
        
        Args:
            data: Data to enrich
            
        Returns:
            Enriched data
        """
        # Add your enrichment logic here
        enriched_data = data.copy()
        enriched_data["enriched"] = True
        enriched_data["enrichment_source"] = "{plugin_name}"
        return enriched_data
'''
        
        main_file = output_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(main_content)
        files.append(str(main_file))
        
        # Generate plugin.json
        plugin_json = {
            "name": plugin_name,
            "version": "1.0.0",
            "description": description or "Enrichment plugin template",
            "author": author or "Vizor Plugin Generator",
            "plugin_type": "enrichment",
            "methods": ["enrich_data"],
            "dependencies": [],
            "config_schema": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        plugin_file = output_dir / "plugin.json"
        with open(plugin_file, 'w') as f:
            json.dump(plugin_json, f, indent=2)
        files.append(str(plugin_file))
        
        return {
            'success': True,
            'plugin_path': str(output_dir),
            'files': files
        } 