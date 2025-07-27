#!/usr/bin/env python3
"""
Vizor Streaming Response Handler
Provides real-time output for better user experience
"""

import asyncio
import ollama
from typing import AsyncGenerator, Dict, Any, Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
import re

console = Console()

class StreamingResponseHandler:
    """
    Handles streaming responses from models for real-time output
    Provides immediate feedback while models are processing
    """
    
    def __init__(self, config):
        self.config = config
        
    async def stream_response(
        self, 
        prompt: str,
        model_name: str,
        task_type: str,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from model in real-time
        
        Args:
            prompt: Input prompt
            model_name: Model to use
            task_type: Type of task for formatting
            system_prompt: Optional system prompt
            
        Yields:
            Chunks of response as they arrive
        """
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Use Ollama's streaming API
            stream = await asyncio.to_thread(
                ollama.chat,
                model=model_name,
                messages=messages,
                stream=True
            )
            
            accumulated_response = ""
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    accumulated_response += content
                    yield content
                    
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
    
    async def display_streaming_response(
        self,
        prompt: str,
        model_name: str,
        task_type: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Display streaming response with rich formatting
        
        Returns:
            Complete response text
        """
        
        accumulated_response = ""
        
        with Live(console=console, refresh_per_second=10) as live:
            live.update(Panel(
                "[yellow]🤔 Thinking...[/yellow]",
                title=f"🤖 {model_name.title()}",
                border_style="blue"
            ))
            
            async for chunk in self.stream_response(prompt, model_name, task_type, system_prompt):
                accumulated_response += chunk
                
                # Format the response based on task type
                formatted_response = self._format_streaming_content(
                    accumulated_response, 
                    task_type
                )
                
                live.update(Panel(
                    formatted_response,
                    title=f"🤖 {model_name.title()} - [green]Responding...[/green]",
                    border_style="green"
                ))
        
        return accumulated_response
    
    def _format_streaming_content(self, content: str, task_type: str) -> str:
        """
        Format streaming content based on task type
        
        Args:
            content: Accumulated content so far
            task_type: Type of task for appropriate formatting
            
        Returns:
            Formatted content for display
        """
        
        if task_type in ["code_generation", "code_analysis"]:
            return self._format_code_content(content)
        elif task_type == "briefing":
            return self._format_briefing_content(content)
        else:
            return content
    
    def _format_code_content(self, content: str) -> str:
        """Format code content with syntax highlighting"""
        
        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
        
        if code_blocks:
            formatted_content = content
            for lang, code in code_blocks:
                if lang and code.strip():
                    try:
                        # Create syntax-highlighted version
                        syntax = Syntax(
                            code.strip(), 
                            lang or "python", 
                            theme="monokai",
                            line_numbers=True,
                            word_wrap=True
                        )
                        # Note: In actual implementation, we'd need to handle this differently
                        # as Live display has limitations with Syntax objects
                        formatted_content = content  # Simplified for now
                    except:
                        pass
            
            return formatted_content
        
        return content
    
    def _format_briefing_content(self, content: str) -> str:
        """Format briefing content with structure"""
        
        # Add structure indicators for briefings
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip().startswith('#'):
                formatted_lines.append(f"[bold blue]{line}[/bold blue]")
            elif line.strip().startswith('- '):
                formatted_lines.append(f"[green]{line}[/green]")
            elif line.strip().startswith('*'):
                formatted_lines.append(f"[yellow]{line}[/yellow]")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

class ProgressiveResponseManager:
    """
    Manages progressive response display with immediate feedback
    """
    
    def __init__(self, config):
        self.config = config
        self.streaming_handler = StreamingResponseHandler(config)
    
    async def get_response_with_progress(
        self,
        prompt: str,
        model_name: str,
        task_type: str,
        show_thinking: bool = True
    ) -> Dict[str, Any]:
        """
        Get response with progressive display and immediate feedback
        
        Args:
            prompt: Input prompt
            model_name: Model to use
            task_type: Task type for formatting
            show_thinking: Whether to show thinking process
            
        Returns:
            Complete response with metadata
        """
        
        start_time = asyncio.get_event_loop().time()
        
        if show_thinking:
            # Show immediate feedback
            console.print(f"[blue]🧠 Routing to {model_name.title()} for {task_type}[/blue]")
            console.print(f"[dim]Estimated response time: {self._estimate_response_time(model_name, len(prompt))}s[/dim]")
        
        # Get streaming response
        response_content = await self.streaming_handler.display_streaming_response(
            prompt=prompt,
            model_name=model_name,
            task_type=task_type
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return {
            'content': response_content,
            'model': model_name,
            'task_type': task_type,
            'processing_time': processing_time,
            'success': True
        }
    
    def _estimate_response_time(self, model_name: str, prompt_length: int) -> float:
        """Estimate response time based on model and prompt length"""
        
        # Base times (in seconds) for different models
        base_times = {
            'phi3': 2.0,        # Fastest
            'mistral': 4.0,     # Medium
            'deepseek-coder': 6.0,  # Slower for complex code
            'wizardcoder': 5.0  # Medium-slow
        }
        
        base_time = base_times.get(model_name, 4.0)
        
        # Adjust for prompt length
        length_factor = min(prompt_length / 1000.0, 2.0)  # Cap at 2x
        
        return base_time * (1 + length_factor)

# Integration with CLI commands
class FastResponseCLI:
    """CLI integration for fast, progressive responses"""
    
    def __init__(self, config):
        self.config = config
        self.progress_manager = ProgressiveResponseManager(config)
    
    async def handle_fast_query(self, question: str, model: Optional[str] = None):
        """Handle query with fast, progressive response"""
        
        # Quick model selection for speed
        if not model:
            if len(question) < 100:
                model = "phi3"  # Use fastest model for short queries
            elif "code" in question.lower():
                model = "deepseek-coder"
            else:
                model = "mistral"
        
        # Determine task type quickly
        task_type = self._quick_task_detection(question)
        
        # Get progressive response
        response = await self.progress_manager.get_response_with_progress(
            prompt=question,
            model_name=model,
            task_type=task_type
        )
        
        return response
    
    def _quick_task_detection(self, question: str) -> str:
        """Quick task type detection for speed"""
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['code', 'script', 'function']):
            if any(word in question_lower for word in ['generate', 'create', 'write']):
                return "code_generation"
            else:
                return "code_analysis"
        elif any(word in question_lower for word in ['brief', 'summary', 'report']):
            return "briefing"
        elif any(word in question_lower for word in ['threat', 'attack', 'malware']):
            return "threat_analysis"
        else:
            return "general_query"
