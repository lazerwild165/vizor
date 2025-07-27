#!/usr/bin/env python3
"""
Vizor LLM Manager
Manages local language models and intelligent routing
"""

import ollama
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import time
from pathlib import Path

class TaskType(Enum):
    """Types of tasks for model routing"""
    GENERAL_QUERY = "general_query"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    THREAT_ANALYSIS = "threat_analysis"
    REASONING = "reasoning"
    BRIEFING = "briefing"
    LEARNING = "learning"

@dataclass
class ModelInfo:
    """Information about a local model"""
    name: str
    display_name: str
    description: str
    strengths: List[TaskType]
    context_length: int
    parameters: str
    available: bool = False
    performance_score: float = 0.0

class LLMManager:
    """Manages local LLM models and intelligent task routing"""
    
    def __init__(self, config):
        self.config = config
        self.models = self._initialize_models()
        self.model_cache = {}
        self.performance_history = {}
        
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize available models with their capabilities"""
        models = {
            "phi3": ModelInfo(
                name="phi3",
                display_name="Phi-3",
                description="Microsoft's efficient small language model, optimized for general tasks",
                strengths=[TaskType.GENERAL_QUERY, TaskType.REASONING, TaskType.BRIEFING],
                context_length=4096,
                parameters="3.8B"
            ),
            "mistral": ModelInfo(
                name="mistral",
                display_name="Mistral",
                description="High-performance open-source model with excellent reasoning capabilities",
                strengths=[TaskType.REASONING, TaskType.THREAT_ANALYSIS, TaskType.LEARNING],
                context_length=8192,
                parameters="7B"
            ),
            "deepseek-coder": ModelInfo(
                name="deepseek-coder",
                display_name="DeepSeek Coder",
                description="Specialized coding model optimized for programming tasks",
                strengths=[TaskType.CODE_ANALYSIS, TaskType.CODE_GENERATION],
                context_length=16384,
                parameters="6.7B"
            ),
            "wizardcoder": ModelInfo(
                name="wizardcoder",
                display_name="WizardCoder",
                description="Code-focused model with strong code generation and analysis",
                strengths=[TaskType.CODE_ANALYSIS, TaskType.CODE_GENERATION],
                context_length=8192,
                parameters="15B"
            )
        }
        
        # Check availability of each model
        self._check_model_availability(models)
        return models
    
    def _check_model_availability(self, models: Dict[str, ModelInfo]):
        """Check which models are actually available via Ollama"""
        try:
            available_models = ollama.list()
            available_names = [model['name'].split(':')[0] for model in available_models['models']]
            
            for model_key, model_info in models.items():
                # Check for exact match or partial match
                model_info.available = any(
                    model_info.name in available_name or available_name in model_info.name
                    for available_name in available_names
                )
                
                if model_info.available:
                    print(f"✅ {model_info.display_name} is available")
                else:
                    print(f"❌ {model_info.display_name} not found")
                    
        except Exception as e:
            print(f"⚠️ Could not check model availability: {e}")
            # Assume all models are available if we can't check
            for model_info in models.values():
                model_info.available = True
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        return [model for model in self.models.values() if model.available]
    
    def select_best_model(self, task_type: TaskType, context_length_needed: int = 0) -> Optional[str]:
        """
        Intelligently select the best model for a given task type
        
        Args:
            task_type: The type of task to perform
            context_length_needed: Minimum context length required
            
        Returns:
            Model name or None if no suitable model found
        """
        available_models = self.get_available_models()
        
        if not available_models:
            return None
        
        # Filter models by task strength and context length
        suitable_models = []
        for model in available_models:
            if (task_type in model.strengths and 
                model.context_length >= context_length_needed):
                suitable_models.append(model)
        
        if not suitable_models:
            # Fallback to any available model if no perfect match
            suitable_models = [model for model in available_models 
                             if model.context_length >= context_length_needed]
        
        if not suitable_models:
            # Last resort: use any available model
            suitable_models = available_models
        
        # Select based on performance history and model capabilities
        best_model = self._rank_models(suitable_models, task_type)
        return best_model.name if best_model else None
    
    def _rank_models(self, models: List[ModelInfo], task_type: TaskType) -> Optional[ModelInfo]:
        """Rank models based on performance history and capabilities"""
        if not models:
            return None
        
        scored_models = []
        for model in models:
            score = 0.0
            
            # Base score from model strengths
            if task_type in model.strengths:
                score += 10.0
            
            # Performance history bonus
            if model.name in self.performance_history:
                history = self.performance_history[model.name]
                avg_performance = sum(history) / len(history)
                score += avg_performance * 5.0
            
            # Context length bonus (more is better)
            score += model.context_length / 1000.0
            
            # Parameter count consideration (bigger models generally better)
            param_count = float(model.parameters.replace('B', ''))
            score += param_count * 0.5
            
            scored_models.append((model, score))
        
        # Sort by score and return best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0]
    
    async def generate_response(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        task_type: TaskType = TaskType.GENERAL_QUERY,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using specified or auto-selected model
        
        Args:
            prompt: The input prompt
            model_name: Specific model to use (auto-select if None)
            task_type: Type of task for model selection
            system_prompt: System prompt to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary with content, model used, and metadata
        """
        start_time = time.time()
        
        # Auto-select model if not specified
        if not model_name:
            model_name = self.select_best_model(task_type, len(prompt) // 4)
            
        if not model_name:
            raise Exception("No suitable model available")
        
        if model_name not in self.models or not self.models[model_name].available:
            raise Exception(f"Model {model_name} not available")
        
        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = await asyncio.to_thread(
                ollama.chat,
                model=model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1
                }
            )
            
            processing_time = time.time() - start_time
            
            # Update performance history
            self._update_performance_history(model_name, processing_time, True)
            
            return {
                "content": response['message']['content'],
                "model": model_name,
                "model_display_name": self.models[model_name].display_name,
                "task_type": task_type.value,
                "processing_time": processing_time,
                "success": True,
                "tokens_used": response.get('eval_count', 0),
                "prompt_tokens": response.get('prompt_eval_count', 0)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_history(model_name, processing_time, False)
            
            return {
                "content": f"Error generating response: {str(e)}",
                "model": model_name,
                "model_display_name": self.models.get(model_name, {}).display_name,
                "task_type": task_type.value,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
    
    def _update_performance_history(self, model_name: str, processing_time: float, success: bool):
        """Update performance history for a model"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        # Score based on speed and success
        score = 10.0 if success else 0.0
        if success and processing_time > 0:
            # Faster responses get higher scores
            score += max(0, 10.0 - processing_time)
        
        self.performance_history[model_name].append(score)
        
        # Keep only last 50 entries
        if len(self.performance_history[model_name]) > 50:
            self.performance_history[model_name] = self.performance_history[model_name][-50:]
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            "total_models": len(self.models),
            "available_models": len(self.get_available_models()),
            "models": {}
        }
        
        for name, model in self.models.items():
            model_status = {
                "display_name": model.display_name,
                "available": model.available,
                "description": model.description,
                "parameters": model.parameters,
                "context_length": model.context_length,
                "strengths": [strength.value for strength in model.strengths]
            }
            
            if name in self.performance_history:
                history = self.performance_history[name]
                model_status["avg_performance"] = sum(history) / len(history)
                model_status["total_uses"] = len(history)
            
            status["models"][name] = model_status
        
        return status
    
    def get_prompt_templates(self) -> Dict[TaskType, str]:
        """Get optimized prompt templates for different task types"""
        return {
            TaskType.GENERAL_QUERY: """You are Vizor, a cybersecurity expert assistant. Provide clear, accurate, and helpful responses to security-related questions. If you're unsure about something, acknowledge the uncertainty and suggest ways to get more information.

User Query: {prompt}""",

            TaskType.CODE_ANALYSIS: """You are a cybersecurity code analyst. Analyze the provided code for security vulnerabilities, potential threats, and best practices. Be thorough and provide specific recommendations.

Code to analyze:
{prompt}""",

            TaskType.CODE_GENERATION: """You are a security-focused code generator. Generate secure, well-documented code that follows cybersecurity best practices. Include error handling and security considerations.

Requirements:
{prompt}""",

            TaskType.THREAT_ANALYSIS: """You are a threat intelligence analyst. Analyze the provided information for security threats, indicators of compromise, and potential risks. Provide actionable intelligence and recommendations.

Threat data:
{prompt}""",

            TaskType.REASONING: """You are a cybersecurity strategist. Think through the problem step by step, consider multiple perspectives, and provide well-reasoned analysis with clear justification for your conclusions.

Problem to analyze:
{prompt}""",

            TaskType.BRIEFING: """You are a cybersecurity briefing specialist. Create clear, concise, and actionable briefings that highlight key threats, trends, and recommendations for decision-makers.

Briefing topic:
{prompt}""",

            TaskType.LEARNING: """You are a cybersecurity educator. Explain complex security concepts clearly, provide context, and help build understanding of cybersecurity topics.

Topic to explain:
{prompt}"""
        }
    
    def format_prompt(self, prompt: str, task_type: TaskType) -> str:
        """Format prompt using appropriate template"""
        templates = self.get_prompt_templates()
        template = templates.get(task_type, templates[TaskType.GENERAL_QUERY])
        return template.format(prompt=prompt)
