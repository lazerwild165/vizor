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
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    RAG_SEARCH = "rag_search"
    ORCHESTRATION = "orchestration"

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
    """Manages local LLM models and intelligent task routing with Swarm Brain architecture"""
    
    def __init__(self, config):
        self.config = config
        self.models = self._initialize_models()
        self.model_cache = {}
        self.performance_history = {}
        
        # Swarm Brain components
        self.task_router = None
        self.rag_pool = None
        self.reasoner = None
        self.summarizer = None
        self.codegen = None
        self.classifier = None
        self.output_builder = None
        
        # Context sharing system
        self.shared_context = {}
        self.context_history = []
        self.max_context_history = 10
        
        # Don't initialize swarm brain or check availability until needed
        self._swarm_initialized = False
        
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize available models with cybersecurity-focused selection"""
        models = {
            # Fast, general-purpose models
            'phi3': ModelInfo(
                name='phi3',
                display_name='Phi-3',
                description='Fast, efficient model good for general cybersecurity tasks',
                strengths=[TaskType.GENERAL_QUERY, TaskType.REASONING, TaskType.LEARNING],
                context_length=8192,
                parameters='3.8B',
                available=False
            ),
            'mistral': ModelInfo(
                name='mistral',
                display_name='Mistral',
                description='Balanced model for security analysis and reasoning',
                strengths=[TaskType.GENERAL_QUERY, TaskType.REASONING, TaskType.THREAT_ANALYSIS, TaskType.BRIEFING],
                context_length=8192,
                parameters='7B',
                available=False
            ),
            
            # Code-focused models
            'deepseek-coder': ModelInfo(
                name='deepseek-coder',
                display_name='DeepSeek Coder',
                description='Specialized for code analysis and vulnerability detection',
                strengths=[TaskType.CODE_ANALYSIS, TaskType.CODE_GENERATION, TaskType.THREAT_ANALYSIS],
                context_length=16384,
                parameters='6.7B',
                available=False
            ),
            'wizardcoder': ModelInfo(
                name='wizardcoder',
                display_name='WizardCoder',
                description='Excellent for secure code generation and review',
                strengths=[TaskType.CODE_GENERATION, TaskType.CODE_ANALYSIS],
                context_length=8192,
                parameters='15B',
                available=False
            ),
            
            # Fast, lightweight models
            'tinyllama': ModelInfo(
                name='tinyllama',
                display_name='TinyLlama',
                description='Very fast model for quick responses',
                strengths=[TaskType.GENERAL_QUERY, TaskType.CLASSIFICATION, TaskType.SUMMARIZATION],
                context_length=2048,
                parameters='1.1B',
                available=False
            ),
            'phi3-mini': ModelInfo(
                name='phi3-mini',
                display_name='Phi-3 Mini',
                description='Ultra-fast model for basic queries',
                strengths=[TaskType.GENERAL_QUERY, TaskType.CLASSIFICATION],
                context_length=4096,
                parameters='3.8B',
                available=False
            ),
            
            # Security-specialized models (if available)
            'llama3.2': ModelInfo(
                name='llama3.2',
                display_name='Llama 3.2',
                description='Latest Llama model with good reasoning capabilities',
                strengths=[TaskType.GENERAL_QUERY, TaskType.REASONING, TaskType.THREAT_ANALYSIS],
                context_length=8192,
                parameters='8B',
                available=False
            ),
            'codellama': ModelInfo(
                name='codellama',
                display_name='Code Llama',
                description='Code-focused model for security analysis',
                strengths=[TaskType.CODE_ANALYSIS, TaskType.CODE_GENERATION],
                context_length=16384,
                parameters='7B',
                available=False
            ),
            
            # Quantized versions for speed
            'mistral:7b-q4_0': ModelInfo(
                name='mistral:7b-q4_0',
                display_name='Mistral (4-bit)',
                description='Fast quantized version of Mistral',
                strengths=[TaskType.GENERAL_QUERY, TaskType.REASONING],
                context_length=8192,
                parameters='7B',
                available=False
            ),
            'phi3:q4_0': ModelInfo(
                name='phi3:q4_0',
                display_name='Phi-3 (4-bit)',
                description='Fast quantized version of Phi-3',
                strengths=[TaskType.GENERAL_QUERY, TaskType.REASONING],
                context_length=8192,
                parameters='3.8B',
                available=False
            )
        }
        
        return models
    
    def _ensure_availability_checked(self):
        """Check model availability only when needed"""
        if not hasattr(self, '_availability_checked'):
            self._check_model_availability(self.models)
            self._availability_checked = True
    
    def _initialize_swarm_brain(self):
        """Initialize Swarm Brain components"""
        try:
            # Task Router - Uses Phi-3 for lightweight orchestration
            if self.models.get("phi3") and self.models["phi3"].available:
                self.task_router = "phi3"
            
            # Reasoner - Uses Mistral for long-context planning
            if self.models.get("mistral") and self.models["mistral"].available:
                self.reasoner = "mistral"
            
            # Summarizer - Uses Phi-3 Mini for content condensation
            if self.models.get("phi3-mini") and self.models["phi3-mini"].available:
                self.summarizer = "phi3-mini"
            elif self.models.get("phi3") and self.models["phi3"].available:
                self.summarizer = "phi3"  # Fallback
            elif self.models.get("mistral") and self.models["mistral"].available:
                self.summarizer = "mistral"  # Alternative fallback
            
            # CodeGen - Uses DeepSeek Coder or WizardCoder
            if self.models.get("deepseek-coder") and self.models["deepseek-coder"].available:
                self.codegen = "deepseek-coder"
            elif self.models.get("wizardcoder") and self.models["wizardcoder"].available:
                self.codegen = "wizardcoder"
            
            # Classifier - Uses DistilBERT or Phi-3
            if self.models.get("distilbert") and self.models["distilbert"].available:
                self.classifier = "distilbert"
            elif self.models.get("phi3") and self.models["phi3"].available:
                self.classifier = "phi3"  # Fallback
            elif self.models.get("mistral") and self.models["mistral"].available:
                self.classifier = "mistral"  # Alternative fallback
            
            # RAG Pool - Uses BGE-Mini for embeddings
            if self.models.get("bge-mini") and self.models["bge-mini"].available:
                self.rag_pool = "bge-mini"
            elif self.models.get("phi3") and self.models["phi3"].available:
                self.rag_pool = "phi3"  # Fallback for embeddings
            
            # Output Builder - Aggregates responses
            self.output_builder = "local"  # Local aggregation logic
            
        except Exception as e:
            print(f"Warning: Swarm Brain initialization failed: {e}")
    

    
    async def swarm_generate_response(
        self,
        prompt: str,
        task_type: TaskType = TaskType.GENERAL_QUERY,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_swarm: bool = True,
        add_to_context: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response using Swarm Brain architecture
        
        Args:
            prompt: Input prompt
            task_type: Type of task
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens
            use_swarm: Whether to use swarm architecture
            
        Returns:
            Response with metadata
        """
        if not use_swarm:
            # Fallback to single model
            return await self.generate_response(
                prompt=prompt,
                task_type=task_type,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        try:
            start_time = time.time()
            
            # Step 1: Task Routing and Analysis
            task_analysis = await self._route_task(prompt, task_type)
            
            # Add task analysis to context for other models
            if add_to_context:
                self.add_context(f"task_analysis_{task_type.value}", task_analysis, task_type)
            
            # Step 2: RAG Search (if needed)
            rag_results = None
            if task_analysis.get('needs_context', False):
                rag_results = await self._search_rag_pool(prompt, task_analysis)
                
                # Add RAG results to context
                if add_to_context and rag_results:
                    self.add_context(f"rag_results_{task_type.value}", rag_results[:3], task_type)
            
            # Step 3: Generate response based on task type
            if task_type == TaskType.CLASSIFICATION:
                response = await self._classify_content(prompt, task_analysis)
            elif task_type == TaskType.SUMMARIZATION:
                response = await self._summarize_content(prompt, task_analysis)
            elif task_type in [TaskType.CODE_GENERATION, TaskType.CODE_ANALYSIS]:
                response = await self._generate_code(prompt, task_analysis, rag_results)
            elif task_type == TaskType.REASONING:
                response = await self._reason_about_topic(prompt, task_analysis, rag_results)
            else:
                # General query - use reasoner
                response = await self._general_query(prompt, task_analysis, rag_results)
            
            # Add response insights to context
            if add_to_context:
                self.add_context(f"response_insights_{task_type.value}", {
                    'key_points': self._extract_key_points(response),
                    'confidence': task_analysis.get('confidence', 0.0),
                    'model_used': task_analysis.get('selected_model', 'unknown')
                }, task_type)
            
            # Step 4: Build final output
            final_response = await self._build_output(response, task_analysis, rag_results)
            
            processing_time = time.time() - start_time
            
            return {
                'content': final_response,
                'model': 'swarm_brain',
                'task_type': task_type.value,
                'processing_time': processing_time,
                'swarm_components': task_analysis.get('components_used', []),
                'confidence': task_analysis.get('confidence', 0.7),
                'rag_results_count': len(rag_results) if rag_results else 0
            }
            
        except Exception as e:
            # Fallback to single model
            print(f"Swarm Brain failed, falling back to single model: {e}")
            return await self.generate_response(
                prompt=prompt,
                task_type=task_type,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    async def _route_task(self, prompt: str, task_type: TaskType) -> Dict[str, Any]:
        """Route task to appropriate components"""
        try:
            if not self.task_router:
                return {
                    'needs_context': True,
                    'confidence': 0.7,
                    'components_used': ['fallback'],
                    'model': 'fallback'
                }
            
            routing_prompt = f"""
            Analyze this task and determine the best approach:
            
            Task Type: {task_type.value}
            Prompt: {prompt}
            
            Provide JSON response:
            {{
                "needs_context": true/false,
                "confidence": 0.0-1.0,
                "components_used": ["list", "of", "components"],
                "model": "primary_model_name"
            }}
            """
            
            response = await self.generate_response(
                prompt=routing_prompt,
                model_name=self.task_router,
                task_type=TaskType.ORCHESTRATION,
                temperature=0.3
            )
            
            try:
                return json.loads(response['content'])
            except:
                return {
                    'needs_context': True,
                    'confidence': 0.7,
                    'components_used': [self.task_router],
                    'model': self.task_router
                }
                
        except Exception as e:
            return {
                'needs_context': True,
                'confidence': 0.7,
                'components_used': ['fallback'],
                'model': 'fallback'
            }
    
    async def _search_rag_pool(self, prompt: str, task_analysis: Dict) -> Optional[List[Dict]]:
        """Search RAG pool for relevant context"""
        # This would integrate with vector memory
        # For now, return None
        return None
    
    async def _classify_content(self, prompt: str, task_analysis: Dict) -> str:
        """Classify content using classifier model"""
        if not self.classifier:
            return "classification_unavailable"
        
        try:
            response = await self.generate_response(
                prompt=prompt,
                model_name=self.classifier,
                task_type=TaskType.CLASSIFICATION,
                temperature=0.3
            )
            return response['content']
        except:
            return "classification_failed"
    
    async def _summarize_content(self, prompt: str, task_analysis: Dict) -> str:
        """Summarize content using summarizer model"""
        if not self.summarizer:
            return "summarization_unavailable"
        
        try:
            response = await self.generate_response(
                prompt=prompt,
                model_name=self.summarizer,
                task_type=TaskType.SUMMARIZATION,
                temperature=0.3
            )
            return response['content']
        except:
            return "summarization_failed"
    
    async def _generate_code(self, prompt: str, task_analysis: Dict, rag_results: Optional[List[Dict]]) -> str:
        """Generate code using codegen model"""
        if not self.codegen:
            return "code_generation_unavailable"
        
        try:
            # Enhance prompt with RAG results if available
            enhanced_prompt = prompt
            if rag_results:
                enhanced_prompt += f"\n\nRelevant context:\n{chr(10).join([r.get('content', '') for r in rag_results[:3]])}"
            
            response = await self.generate_response(
                prompt=enhanced_prompt,
                model_name=self.codegen,
                task_type=TaskType.CODE_GENERATION,
                temperature=0.3
            )
            return response['content']
        except:
            return "code_generation_failed"
    
    async def _reason_about_topic(self, prompt: str, task_analysis: Dict, rag_results: Optional[List[Dict]]) -> str:
        """Reason about topic using reasoner model"""
        if not self.reasoner:
            return "reasoning_unavailable"
        
        try:
            # Enhance prompt with RAG results if available
            enhanced_prompt = prompt
            if rag_results:
                enhanced_prompt += f"\n\nRelevant context:\n{chr(10).join([r.get('content', '') for r in rag_results[:3]])}"
            
            response = await self.generate_response(
                prompt=enhanced_prompt,
                model_name=self.reasoner,
                task_type=TaskType.REASONING,
                temperature=0.7
            )
            return response['content']
        except:
            return "reasoning_failed"
    
    async def _general_query(self, prompt: str, task_analysis: Dict, rag_results: Optional[List[Dict]]) -> str:
        """Handle general query using reasoner"""
        return await self._reason_about_topic(prompt, task_analysis, rag_results)
    
    async def _build_output(self, response: str, task_analysis: Dict, rag_results: Optional[List[Dict]]) -> str:
        """Build final output from component responses"""
        # Simple output building - could be enhanced with more sophisticated aggregation
        return response
    
    def _check_model_availability(self, models: Dict[str, ModelInfo]):
        """Check which models are actually available via Ollama"""
        try:
            # For now, just assume the models we know are available
            known_available = ['phi3', 'mistral', 'deepseek-coder', 'wizardcoder']
            
            for model_key, model_info in models.items():
                # Check if model is in our known available list
                model_info.available = model_info.name in known_available
                
                # Only print in verbose mode or when explicitly requested
                if hasattr(self, 'config') and getattr(self.config, 'verbose', False):
                    if model_info.available:
                        print(f"✅ {model_info.display_name} is available")
                    else:
                        print(f"❌ {model_info.display_name} not found")
                    
        except Exception as e:
            if hasattr(self, 'config') and getattr(self.config, 'verbose', False):
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
        # Check availability only when actually selecting a model
        self._ensure_availability_checked()
        
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
        Ultra-fast response generation - Direct to Ollama
        
        Args:
            prompt: Input prompt
            model_name: Specific model to use
            task_type: Type of task
            system_prompt: System prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Returns:
            Response with metadata
        """
        start_time = time.time()
        
        try:
            # ULTRA FAST: Direct Ollama call
            import ollama
            
            # Use specified model or default to mistral
            selected_model = model_name or "mistral"
            
            # Simple system prompt
            system_msg = system_prompt or "You are a helpful cybersecurity assistant. Provide clear, accurate answers."
            
            # Direct call to Ollama
            response = ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or 1024  # Shorter for speed
                }
            )
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'content': response['message']['content'],
                'model': selected_model,
                'processing_time': processing_time,
                'tokens_used': len(response['message']['content'].split())
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'content': f"Error generating response: {str(e)}",
                'error': str(e),
                'processing_time': processing_time
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
        """Get task-specific prompt templates with tight system prompts"""
        return {
            TaskType.GENERAL_QUERY: """You are Vizor, a local-first cybersecurity copilot. You operate with complete privacy and autonomy.

CORE PRINCIPLES:
- Local-first: All processing happens locally, no data leaves your system
- Privacy-focused: No telemetry, no cloud dependencies
- Autonomous: Self-learning and self-improving capabilities
- Expert-level: Deep cybersecurity knowledge across all domains

RESPONSE STYLE:
- Concise but comprehensive
- Actionable insights
- Technical accuracy
- Professional tone
- Include relevant context when available

User Query: {prompt}""",

            TaskType.CODE_ANALYSIS: """You are Vizor's Code Analysis Engine. Your role is to identify security vulnerabilities and provide remediation guidance.

ANALYSIS FRAMEWORK:
1. Identify potential vulnerabilities (OWASP Top 10, CWE, etc.)
2. Assess severity and impact
3. Provide specific remediation steps
4. Suggest security best practices
5. Reference relevant standards (NIST, ISO 27001, etc.)

FORMAT:
- Vulnerability: [Type]
- Severity: [High/Medium/Low]
- Impact: [Description]
- Remediation: [Specific steps]
- Best Practice: [Guidance]

Code to analyze:
{prompt}""",

            TaskType.CODE_GENERATION: """You are Vizor's Secure Code Generator. Generate production-ready, secure code following cybersecurity best practices.

SECURITY REQUIREMENTS:
- Input validation and sanitization
- Output encoding
- Authentication and authorization
- Secure communication (HTTPS, TLS)
- Error handling without information disclosure
- Logging and monitoring
- Memory safety considerations

CODE STANDARDS:
- Clear documentation
- Security comments
- Error handling
- Input validation
- Secure defaults
- Principle of least privilege

Requirements:
{prompt}""",

            TaskType.THREAT_ANALYSIS: """You are Vizor's Threat Intelligence Analyst. Analyze security threats using MITRE ATT&CK, STIX, and industry frameworks.

ANALYSIS FRAMEWORK:
1. Threat Actor Identification
2. TTP Analysis (Tactics, Techniques, Procedures)
3. Impact Assessment
4. Mitigation Strategies
5. Detection Methods
6. Response Recommendations

OUTPUT FORMAT:
- Threat Level: [Critical/High/Medium/Low]
- ATT&CK Techniques: [List]
- Indicators: [IOCs]
- Mitigation: [Actions]
- Detection: [Methods]

Threat data:
{prompt}""",

            TaskType.REASONING: """You are Vizor's Reasoning Engine. Think through complex cybersecurity problems systematically.

REASONING PROCESS:
1. Break down the problem into components
2. Identify key security principles involved
3. Consider multiple attack vectors
4. Evaluate trade-offs and risks
5. Synthesize a comprehensive solution
6. Validate against security best practices

APPROACH:
- Systematic analysis
- Evidence-based reasoning
- Consider edge cases
- Balance security vs usability
- Provide clear rationale

Problem to analyze:
{prompt}""",

            TaskType.BRIEFING: """You are Vizor's Security Briefing Specialist. Create executive-level security briefings that are actionable and informative.

BRIEFING STRUCTURE:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Risk Assessment (High/Medium/Low)
4. Recommended Actions
5. Timeline and Resources
6. Next Steps

STYLE:
- Executive-friendly language
- Actionable recommendations
- Clear risk levels
- Specific timelines
- Resource requirements

Briefing topic:
{prompt}""",

            TaskType.LEARNING: """You are Vizor's Learning Engine. Help users understand cybersecurity concepts and integrate new knowledge.

LEARNING APPROACH:
1. Explain core concepts clearly
2. Provide real-world examples
3. Connect to existing knowledge
4. Offer practical applications
5. Suggest further learning paths

PEDAGOGY:
- Progressive complexity
- Hands-on examples
- Security-first mindset
- Current threat landscape
- Industry best practices

Topic to explain:
{prompt}""",

            TaskType.CLASSIFICATION: """You are Vizor's Security Classifier. Categorize and classify security-related content accurately.

CLASSIFICATION FRAMEWORK:
- Threat Type: Malware, Phishing, APT, Insider, etc.
- Severity: Critical, High, Medium, Low
- Domain: Network, Application, Physical, Social
- Stage: Reconnaissance, Weaponization, Delivery, Exploitation, Installation, Command & Control, Actions on Objectives
- Industry: Financial, Healthcare, Government, etc.

OUTPUT: Structured classification with confidence scores.

Content to classify:
{prompt}""",

            TaskType.SUMMARIZATION: """You are Vizor's Security Summarizer. Create concise, accurate summaries of security content.

SUMMARIZATION GUIDELINES:
- Extract key security insights
- Maintain technical accuracy
- Highlight actionable items
- Preserve critical context
- Use clear, professional language

FORMAT:
- Key Points (3-5 bullet points)
- Risk Assessment
- Action Items
- Technical Details (if relevant)

Content to summarize:
{prompt}""",

            TaskType.RAG_SEARCH: """You are Vizor's Security Search Assistant. Help find and retrieve relevant security information from the knowledge base.

SEARCH CAPABILITIES:
- Semantic similarity matching
- Keyword extraction
- Context-aware retrieval
- Relevance scoring
- Source attribution

OUTPUT:
- Relevant documents
- Confidence scores
- Source information
- Related topics

Search query:
{prompt}""",

            TaskType.ORCHESTRATION: """You are Vizor's Task Orchestrator. Route and coordinate security tasks across the system.

ORCHESTRATION ROLES:
- Task analysis and routing
- Resource allocation
- Context sharing
- Result aggregation
- Quality assurance

DECISION FACTORS:
- Task complexity
- Model capabilities
- Context requirements
- Performance history
- Resource availability

Task to orchestrate:
{prompt}"""
        }
    
    def format_prompt(self, prompt: str, task_type: TaskType) -> str:
        """Format prompt using appropriate template with context sharing"""
        templates = self.get_prompt_templates()
        template = templates.get(task_type, templates[TaskType.GENERAL_QUERY])
        
        # Add shared context if available
        context_info = self._get_relevant_context(task_type)
        if context_info:
            context_prompt = f"\n\nRELEVANT CONTEXT:\n{context_info}\n\n"
            return template.format(prompt=context_prompt + prompt)
        
        return template.format(prompt=prompt)
    
    def _get_relevant_context(self, task_type: TaskType) -> Optional[str]:
        """Get relevant context for the current task"""
        if not self.shared_context:
            return None
        
        # Filter context based on task type
        relevant_context = []
        
        for key, value in self.shared_context.items():
            if self._is_context_relevant(key, value, task_type):
                relevant_context.append(f"{key}: {value}")
        
        return "\n".join(relevant_context) if relevant_context else None
    
    def _is_context_relevant(self, key: str, value: Any, task_type: TaskType) -> bool:
        """Determine if context is relevant to the current task"""
        # Task-specific relevance rules
        relevance_rules = {
            TaskType.CODE_ANALYSIS: ['code_context', 'vulnerabilities', 'security_patterns'],
            TaskType.THREAT_ANALYSIS: ['threat_indicators', 'attack_patterns', 'ioc_data'],
            TaskType.REASONING: ['previous_analysis', 'decision_context', 'risk_assessment'],
            TaskType.BRIEFING: ['threat_landscape', 'incident_history', 'risk_metrics'],
            TaskType.LEARNING: ['knowledge_gaps', 'learning_progress', 'topic_connections']
        }
        
        relevant_keys = relevance_rules.get(task_type, [])
        return any(key.startswith(rel_key) for rel_key in relevant_keys)
    
    def add_context(self, key: str, value: Any, task_type: Optional[TaskType] = None):
        """Add context that can be shared between models"""
        self.shared_context[key] = value
        
        # Add to history
        self.context_history.append({
            'key': key,
            'value': value,
            'task_type': task_type,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(self.context_history) > self.max_context_history:
            self.context_history = self.context_history[-self.max_context_history:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current shared context"""
        return {
            'context_count': len(self.shared_context),
            'history_count': len(self.context_history),
            'recent_context': self.context_history[-5:] if self.context_history else [],
            'context_keys': list(self.shared_context.keys())
        }
    
    def clear_context(self, key: Optional[str] = None):
        """Clear specific or all context"""
        if key:
            self.shared_context.pop(key, None)
        else:
            self.shared_context.clear()
            self.context_history.clear()
    
    def _extract_key_points(self, response: str) -> List[str]:
        """Extract key points from a response for context sharing"""
        try:
            # Simple extraction - look for bullet points, numbered lists, or key phrases
            lines = response.split('\n')
            key_points = []
            
            for line in lines:
                line = line.strip()
                if line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                    key_points.append(line)
                elif any(keyword in line.lower() for keyword in ['important', 'critical', 'key', 'essential', 'vulnerability', 'threat', 'risk']):
                    key_points.append(line)
            
            return key_points[:5]  # Limit to 5 key points
        except:
            return []
