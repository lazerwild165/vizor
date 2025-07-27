#!/usr/bin/env python3
"""
Vizor Output Formatters
Ensures proper formatting, indentation, and code quality
"""

import re
import ast
import json
from typing import Dict, Any, Optional, List
from rich.syntax import Syntax
from rich.console import Console
from rich.panel import Panel
import black
import autopep8

console = Console()

class CodeFormatter:
    """
    Ensures proper code formatting and indentation
    Validates syntax and applies best practices
    """
    
    def __init__(self, config):
        self.config = config
        
    def format_code_response(self, response: str, language: str = "python") -> Dict[str, Any]:
        """
        Format code response with proper indentation and validation
        
        Args:
            response: Raw response from model
            language: Programming language
            
        Returns:
            Formatted response with validation results
        """
        
        # Extract code blocks from response
        code_blocks = self._extract_code_blocks(response)
        
        formatted_response = response
        validation_results = []
        
        for i, (lang, code) in enumerate(code_blocks):
            if not lang:
                lang = language
            
            # Format the code
            formatted_code, validation = self._format_single_code_block(code, lang)
            
            # Replace in response
            formatted_response = formatted_response.replace(
                f"```{lang}\n{code}\n```",
                f"```{lang}\n{formatted_code}\n```"
            )
            
            validation_results.append({
                'block_index': i,
                'language': lang,
                'original_lines': len(code.split('\n')),
                'formatted_lines': len(formatted_code.split('\n')),
                'syntax_valid': validation['syntax_valid'],
                'issues_fixed': validation['issues_fixed']
            })
        
        return {
            'formatted_response': formatted_response,
            'code_blocks_found': len(code_blocks),
            'validation_results': validation_results,
            'all_syntax_valid': all(v['syntax_valid'] for v in validation_results)
        }
    
    def _extract_code_blocks(self, text: str) -> List[tuple]:
        """Extract code blocks from markdown-style text"""
        
        # Pattern for code blocks with language specification
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        return [(lang.strip() if lang else None, code.strip()) for lang, code in matches]
    
    def _format_single_code_block(self, code: str, language: str) -> tuple:
        """
        Format a single code block with proper indentation
        
        Args:
            code: Raw code string
            language: Programming language
            
        Returns:
            Tuple of (formatted_code, validation_info)
        """
        
        validation_info = {
            'syntax_valid': True,
            'issues_fixed': [],
            'errors': []
        }
        
        if language.lower() == 'python':
            return self._format_python_code(code, validation_info)
        elif language.lower() in ['bash', 'shell', 'sh']:
            return self._format_shell_code(code, validation_info)
        elif language.lower() in ['javascript', 'js']:
            return self._format_javascript_code(code, validation_info)
        elif language.lower() in ['powershell', 'ps1']:
            return self._format_powershell_code(code, validation_info)
        else:
            # Generic formatting
            return self._format_generic_code(code, validation_info)
    
    def _format_python_code(self, code: str, validation_info: Dict) -> tuple:
        """Format Python code with proper indentation and style"""
        
        try:
            # First, validate syntax
            ast.parse(code)
            validation_info['syntax_valid'] = True
        except SyntaxError as e:
            validation_info['syntax_valid'] = False
            validation_info['errors'].append(f"Syntax error: {str(e)}")
            # Try to fix common issues
            code = self._fix_common_python_issues(code, validation_info)
        
        try:
            # Apply Black formatting for consistent style
            formatted_code = black.format_str(code, mode=black.FileMode(
                line_length=88,
                string_normalization=True,
                is_pyi=False
            ))
            validation_info['issues_fixed'].append("Applied Black formatting")
        except Exception as e:
            try:
                # Fallback to autopep8
                formatted_code = autopep8.fix_code(code, options={
                    'max_line_length': 88,
                    'indent_size': 4,
                    'aggressive': 1
                })
                validation_info['issues_fixed'].append("Applied autopep8 formatting")
            except Exception as e2:
                # Last resort: basic indentation fix
                formatted_code = self._fix_basic_indentation(code)
                validation_info['issues_fixed'].append("Applied basic indentation fix")
                validation_info['errors'].append(f"Advanced formatting failed: {str(e)}")
        
        return formatted_code.strip(), validation_info
    
    def _fix_common_python_issues(self, code: str, validation_info: Dict) -> str:
        """Fix common Python syntax issues"""
        
        fixes_applied = []
        
        # Fix missing colons
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Add missing colons for control structures
            if (stripped.startswith(('if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try', 'except', 'finally', 'with ')) 
                and not stripped.endswith(':') and not stripped.endswith('\\') and stripped != 'else'):
                line = line + ':'
                fixes_applied.append("Added missing colon")
            
            fixed_lines.append(line)
        
        if fixes_applied:
            validation_info['issues_fixed'].extend(fixes_applied)
        
        return '\n'.join(fixed_lines)
    
    def _fix_basic_indentation(self, code: str) -> str:
        """Apply basic indentation fixing"""
        
        lines = code.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Decrease indent for certain keywords
            if stripped.startswith(('except', 'elif', 'else', 'finally')):
                current_indent = max(0, indent_level - 1)
            elif stripped.startswith(('def ', 'class ')) and indent_level > 0:
                current_indent = 0
                indent_level = 0
            else:
                current_indent = indent_level
            
            # Apply indentation
            fixed_line = '    ' * current_indent + stripped
            fixed_lines.append(fixed_line)
            
            # Increase indent for control structures
            if stripped.endswith(':'):
                indent_level = current_indent + 1
        
        return '\n'.join(fixed_lines)
    
    def _format_shell_code(self, code: str, validation_info: Dict) -> tuple:
        """Format shell/bash code"""
        
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Basic shell formatting
                if stripped.startswith('#'):
                    formatted_lines.append(stripped)  # Comments
                elif stripped.startswith(('if', 'for', 'while', 'case')):
                    formatted_lines.append(stripped)
                elif stripped.startswith(('then', 'do', 'else')):
                    formatted_lines.append('  ' + stripped)  # Indent control blocks
                elif stripped in ('fi', 'done', 'esac'):
                    formatted_lines.append(stripped)  # End blocks
                else:
                    formatted_lines.append(stripped)
            else:
                formatted_lines.append('')
        
        validation_info['issues_fixed'].append("Applied shell formatting")
        return '\n'.join(formatted_lines), validation_info
    
    def _format_javascript_code(self, code: str, validation_info: Dict) -> tuple:
        """Format JavaScript code"""
        
        # Basic JavaScript formatting
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Handle closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Apply indentation
            formatted_line = '  ' * indent_level + stripped
            formatted_lines.append(formatted_line)
            
            # Handle opening braces
            if stripped.endswith('{'):
                indent_level += 1
        
        validation_info['issues_fixed'].append("Applied JavaScript formatting")
        return '\n'.join(formatted_lines), validation_info
    
    def _format_powershell_code(self, code: str, validation_info: Dict) -> tuple:
        """Format PowerShell code"""
        
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Basic PowerShell formatting
                formatted_lines.append(stripped)
            else:
                formatted_lines.append('')
        
        validation_info['issues_fixed'].append("Applied PowerShell formatting")
        return '\n'.join(formatted_lines), validation_info
    
    def _format_generic_code(self, code: str, validation_info: Dict) -> tuple:
        """Generic code formatting"""
        
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Remove excessive whitespace but preserve structure
            if line.strip():
                formatted_lines.append(line.rstrip())
            else:
                formatted_lines.append('')
        
        return '\n'.join(formatted_lines), validation_info

class ResponseFormatter:
    """
    Formats different types of responses for optimal display
    """
    
    def __init__(self, config):
        self.config = config
        self.code_formatter = CodeFormatter(config)
    
    def format_response(self, response: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Format response based on task type
        
        Args:
            response: Raw response from model
            task_type: Type of task for appropriate formatting
            
        Returns:
            Formatted response with metadata
        """
        
        content = response.get('content', '')
        
        if task_type in ['code_generation', 'code_analysis']:
            return self._format_code_response(response, content)
        elif task_type == 'briefing':
            return self._format_briefing_response(response, content)
        elif task_type == 'threat_analysis':
            return self._format_threat_response(response, content)
        else:
            return self._format_general_response(response, content)
    
    def _format_code_response(self, response: Dict, content: str) -> Dict[str, Any]:
        """Format code-related responses"""
        
        # Apply code formatting
        formatting_result = self.code_formatter.format_code_response(content)
        
        # Create formatted response
        formatted_response = response.copy()
        formatted_response['content'] = formatting_result['formatted_response']
        formatted_response['formatting_metadata'] = {
            'code_blocks_found': formatting_result['code_blocks_found'],
            'all_syntax_valid': formatting_result['all_syntax_valid'],
            'validation_results': formatting_result['validation_results']
        }
        
        return formatted_response
    
    def _format_briefing_response(self, response: Dict, content: str) -> Dict[str, Any]:
        """Format briefing responses with structure"""
        
        # Add structure to briefings
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                # Format headers
                if stripped.startswith('#'):
                    formatted_lines.append(f"\n{stripped}")
                # Format bullet points
                elif stripped.startswith(('- ', '* ', '• ')):
                    formatted_lines.append(f"  {stripped}")
                # Format numbered lists
                elif re.match(r'^\d+\.', stripped):
                    formatted_lines.append(f"  {stripped}")
                else:
                    formatted_lines.append(stripped)
            else:
                formatted_lines.append('')
        
        formatted_response = response.copy()
        formatted_response['content'] = '\n'.join(formatted_lines)
        formatted_response['formatting_metadata'] = {
            'structure_applied': True,
            'type': 'briefing'
        }
        
        return formatted_response
    
    def _format_threat_response(self, response: Dict, content: str) -> Dict[str, Any]:
        """Format threat analysis responses"""
        
        # Structure threat analysis
        formatted_response = response.copy()
        formatted_response['content'] = content
        formatted_response['formatting_metadata'] = {
            'type': 'threat_analysis',
            'structured': True
        }
        
        return formatted_response
    
    def _format_general_response(self, response: Dict, content: str) -> Dict[str, Any]:
        """Format general responses"""
        
        formatted_response = response.copy()
        formatted_response['content'] = content.strip()
        formatted_response['formatting_metadata'] = {
            'type': 'general'
        }
        
        return formatted_response

class EnhancedPromptTemplates:
    """
    Enhanced prompt templates that enforce proper formatting
    """
    
    @staticmethod
    def get_code_generation_prompt(requirements: str, language: str = "python") -> str:
        """Get enhanced code generation prompt that enforces formatting"""
        
        return f"""You are an expert cybersecurity code generator. Generate secure, well-formatted code that follows best practices.

FORMATTING REQUIREMENTS:
- Use proper indentation (4 spaces for Python, 2 for JavaScript)
- Include comprehensive error handling
- Add security considerations as comments
- Follow language-specific style guides
- Ensure all code blocks are properly formatted with ```{language}

SECURITY REQUIREMENTS:
- Validate all inputs
- Use secure coding practices
- Include appropriate error handling
- Add security comments where relevant

Requirements:
{requirements}

Please provide:
1. Clean, well-formatted code
2. Explanation of security considerations
3. Usage examples
4. Error handling details

Format your response with proper markdown code blocks using ```{language}
"""
    
    @staticmethod
    def get_code_analysis_prompt(code: str, language: str = "python") -> str:
        """Get enhanced code analysis prompt"""
        
        return f"""You are a cybersecurity code analyst. Analyze the provided code for security vulnerabilities, formatting issues, and best practices.

ANALYSIS REQUIREMENTS:
- Identify security vulnerabilities
- Check code formatting and style
- Suggest improvements
- Provide corrected code if needed

Code to analyze:
```{language}
{code}
```

Please provide:
1. Security vulnerability assessment
2. Code quality analysis
3. Formatting recommendations
4. Corrected/improved version (if needed)
5. Best practice suggestions

Format any code examples with proper ```{language} blocks.
"""

# Integration with LLM Manager
def enhance_llm_manager_with_formatting():
    """Enhancement to integrate formatting with LLM Manager"""
    
    enhanced_templates = {
        'CODE_GENERATION': EnhancedPromptTemplates.get_code_generation_prompt,
        'CODE_ANALYSIS': EnhancedPromptTemplates.get_code_analysis_prompt
    }
    
    return enhanced_templates
