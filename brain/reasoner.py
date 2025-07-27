#!/usr/bin/env python3
"""
Vizor Meta-Reasoner
Core reasoning engine with gap detection and model routing
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

from models.llm_manager import LLMManager, TaskType
from brain.memory import VectorMemory
from brain.gap_detector import KnowledgeGapDetector

@dataclass
class QueryContext:
    """Context information for a query"""
    user_query: str
    conversation_history: List[Dict] = None
    domain_context: Optional[str] = None
    urgency_level: str = "normal"  # low, normal, high, critical
    expected_response_type: str = "explanation"  # explanation, action, analysis, briefing

class ConfidenceLevel(Enum):
    """Confidence levels for responses"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class ReasoningResult:
    """Result of reasoning process"""
    answer: str
    confidence: float
    model_used: str
    processing_time: float
    knowledge_gaps: List[str]
    sources_consulted: List[str]
    reasoning_chain: List[str]
    suggested_actions: List[str]

class MetaReasoner:
    """
    Meta-reasoning engine that orchestrates thinking and response generation
    
    This is the core "brain" of Vizor that:
    1. Analyzes incoming queries for complexity and domain
    2. Detects knowledge gaps and confidence levels
    3. Routes to appropriate models based on task type
    4. Manages conversation context and memory
    5. Triggers learning flows when needed
    """
    
    def __init__(self, config):
        self.config = config
        self.llm_manager = LLMManager(config)
        self.vector_memory = VectorMemory(config)
        self.gap_detector = KnowledgeGapDetector(config)
        self.conversation_memory = {}
        
    async def process_query(
        self,
        question: str,
        context: Optional[Any] = None,
        model: Optional[str] = None,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        
        Args:
            question: User's question or request
            context: Additional context (conversation history, etc.)
            model: Specific model to use (auto-select if None)
            confidence_threshold: Minimum confidence required
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze the query
            query_analysis = await self._analyze_query(question, context)
            
            # Step 2: Check existing knowledge and detect gaps
            knowledge_check = await self._check_knowledge_base(question, query_analysis)
            
            # Step 3: Determine task type and select appropriate model
            task_type = self._determine_task_type(question, query_analysis)
            selected_model = model or self.llm_manager.select_best_model(
                task_type, 
                len(question) // 4
            )
            
            # Step 4: Generate response using selected model
            response = await self._generate_response(
                question, 
                query_analysis, 
                knowledge_check, 
                task_type, 
                selected_model
            )
            
            # Step 5: Evaluate confidence and detect gaps
            confidence_assessment = await self._assess_confidence(
                question, 
                response, 
                knowledge_check
            )
            
            # Step 6: Handle low confidence scenarios
            if confidence_assessment['confidence'] < confidence_threshold:
                gap_analysis = await self._handle_low_confidence(
                    question, 
                    response, 
                    confidence_assessment
                )
                knowledge_gaps = gap_analysis['gaps']
            else:
                knowledge_gaps = []
            
            # Step 7: Update memory and learning
            await self._update_memory(question, response, confidence_assessment)
            
            processing_time = time.time() - start_time
            
            return {
                'answer': response['content'],
                'confidence': confidence_assessment['confidence'],
                'model': selected_model,
                'model_display_name': self.llm_manager.models.get(selected_model, {}).display_name,
                'task_type': task_type.value,
                'processing_time': processing_time,
                'knowledge_gaps': knowledge_gaps,
                'reasoning_chain': confidence_assessment.get('reasoning_steps', []),
                'sources_consulted': knowledge_check.get('sources', []),
                'suggested_actions': confidence_assessment.get('suggested_actions', [])
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'confidence': 0.0,
                'model': 'error',
                'model_display_name': 'Error Handler',
                'task_type': 'error',
                'processing_time': processing_time,
                'knowledge_gaps': ['error_handling'],
                'error': str(e)
            }
    
    async def _analyze_query(self, question: str, context: Any) -> Dict[str, Any]:
        """Analyze the incoming query for complexity, domain, and intent"""
        
        # Use Mistral for complex reasoning tasks
        analysis_prompt = f"""
        Analyze this cybersecurity query for the following characteristics:
        
        Query: "{question}"
        
        Please provide analysis in JSON format:
        {{
            "complexity": "low|medium|high",
            "domain": "general|malware|network|incident_response|threat_intel|compliance|other",
            "intent": "question|request_action|analysis|briefing|learning",
            "technical_level": "beginner|intermediate|advanced|expert",
            "urgency_indicators": ["list", "of", "urgency", "keywords"],
            "key_concepts": ["list", "of", "key", "cybersecurity", "concepts"],
            "requires_real_time_data": true/false,
            "estimated_response_length": "short|medium|long"
        }}
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=analysis_prompt,
                model_name="mistral",  # Use Mistral for reasoning
                task_type=TaskType.REASONING,
                temperature=0.3
            )
            
            # Parse JSON response
            analysis = json.loads(response['content'])
            return analysis
            
        except Exception as e:
            # Fallback analysis
            return {
                "complexity": "medium",
                "domain": "general",
                "intent": "question",
                "technical_level": "intermediate",
                "urgency_indicators": [],
                "key_concepts": [],
                "requires_real_time_data": False,
                "estimated_response_length": "medium"
            }
    
    async def _check_knowledge_base(self, question: str, analysis: Dict) -> Dict[str, Any]:
        """Check vector memory for relevant existing knowledge"""
        
        try:
            # Search vector memory for relevant information
            relevant_docs = await self.vector_memory.search(
                query=question,
                top_k=5,
                filter_domain=analysis.get('domain')
            )
            
            # Check for knowledge gaps
            gap_assessment = await self.gap_detector.assess_knowledge_gaps(
                question, 
                relevant_docs, 
                analysis
            )
            
            return {
                'relevant_documents': relevant_docs,
                'knowledge_coverage': gap_assessment['coverage_score'],
                'identified_gaps': gap_assessment['gaps'],
                'sources': [doc['source'] for doc in relevant_docs],
                'confidence_boost': len(relevant_docs) * 0.1  # Boost confidence based on available knowledge
            }
            
        except Exception as e:
            return {
                'relevant_documents': [],
                'knowledge_coverage': 0.0,
                'identified_gaps': ['knowledge_base_error'],
                'sources': [],
                'confidence_boost': 0.0
            }
    
    def _determine_task_type(self, question: str, analysis: Dict) -> TaskType:
        """Determine the appropriate task type for model selection"""
        
        intent = analysis.get('intent', 'question')
        domain = analysis.get('domain', 'general')
        
        # Check for code-related queries
        code_indicators = ['code', 'script', 'function', 'vulnerability', 'exploit', 'payload']
        if any(indicator in question.lower() for indicator in code_indicators):
            if 'analyze' in question.lower() or 'review' in question.lower():
                return TaskType.CODE_ANALYSIS
            elif 'generate' in question.lower() or 'create' in question.lower() or 'write' in question.lower():
                return TaskType.CODE_GENERATION
        
        # Check for threat analysis
        threat_indicators = ['threat', 'attack', 'malware', 'ioc', 'indicator', 'campaign']
        if any(indicator in question.lower() for indicator in threat_indicators):
            return TaskType.THREAT_ANALYSIS
        
        # Check for briefing requests
        briefing_indicators = ['brief', 'summary', 'report', 'overview', 'status']
        if any(indicator in question.lower() for indicator in briefing_indicators):
            return TaskType.BRIEFING
        
        # Check for learning requests
        learning_indicators = ['explain', 'how does', 'what is', 'teach me', 'learn about']
        if any(indicator in question.lower() for indicator in learning_indicators):
            return TaskType.LEARNING
        
        # Complex reasoning for strategic questions
        if analysis.get('complexity') == 'high' or intent == 'analysis':
            return TaskType.REASONING
        
        # Default to general query
        return TaskType.GENERAL_QUERY
    
    async def _generate_response(
        self, 
        question: str, 
        analysis: Dict, 
        knowledge_check: Dict, 
        task_type: TaskType, 
        model: str
    ) -> Dict[str, Any]:
        """Generate response using the selected model and context"""
        
        # Build context from relevant documents
        context_docs = knowledge_check.get('relevant_documents', [])
        context_text = ""
        if context_docs:
            context_text = "\n\nRelevant context from knowledge base:\n"
            for doc in context_docs[:3]:  # Use top 3 most relevant
                context_text += f"- {doc['content'][:200]}...\n"
        
        # Format prompt using task-specific template
        formatted_prompt = self.llm_manager.format_prompt(
            question + context_text, 
            task_type
        )
        
        # Generate response
        response = await self.llm_manager.generate_response(
            prompt=formatted_prompt,
            model_name=model,
            task_type=task_type,
            temperature=0.7 if task_type == TaskType.CODE_GENERATION else 0.5
        )
        
        return response
    
    async def _assess_confidence(
        self, 
        question: str, 
        response: Dict, 
        knowledge_check: Dict
    ) -> Dict[str, Any]:
        """Assess confidence in the generated response"""
        
        base_confidence = 0.5
        
        # Factors that increase confidence
        if knowledge_check['knowledge_coverage'] > 0.7:
            base_confidence += 0.2
        
        if len(knowledge_check['relevant_documents']) > 2:
            base_confidence += 0.1
        
        if response.get('success', False):
            base_confidence += 0.1
        
        # Factors that decrease confidence
        if knowledge_check['identified_gaps']:
            base_confidence -= len(knowledge_check['identified_gaps']) * 0.1
        
        if 'I don\'t know' in response.get('content', '') or 'uncertain' in response.get('content', '').lower():
            base_confidence -= 0.3
        
        # Clamp confidence between 0 and 1
        final_confidence = max(0.0, min(1.0, base_confidence))
        
        # Generate reasoning steps
        reasoning_steps = [
            f"Initial confidence: {base_confidence:.2f}",
            f"Knowledge coverage: {knowledge_check['knowledge_coverage']:.2f}",
            f"Relevant documents: {len(knowledge_check['relevant_documents'])}",
            f"Identified gaps: {len(knowledge_check['identified_gaps'])}",
            f"Final confidence: {final_confidence:.2f}"
        ]
        
        # Suggest actions based on confidence
        suggested_actions = []
        if final_confidence < 0.6:
            suggested_actions.extend([
                f"Consider running: vizor learn '{question[:50]}...'",
                "Gather more specific information about the topic",
                "Consult additional threat intelligence sources"
            ])
        
        return {
            'confidence': final_confidence,
            'reasoning_steps': reasoning_steps,
            'suggested_actions': suggested_actions
        }
    
    async def _handle_low_confidence(
        self, 
        question: str, 
        response: Dict, 
        confidence_assessment: Dict
    ) -> Dict[str, Any]:
        """Handle scenarios where confidence is below threshold"""
        
        # Identify specific knowledge gaps
        gaps = await self.gap_detector.identify_specific_gaps(
            question, 
            response, 
            confidence_assessment
        )
        
        # Log gaps for future learning
        await self._log_knowledge_gaps(gaps)
        
        return {
            'gaps': gaps,
            'learning_suggestions': [f"Learn about: {gap}" for gap in gaps],
            'confidence_improvement_needed': True
        }
    
    async def _update_memory(
        self, 
        question: str, 
        response: Dict, 
        confidence_assessment: Dict
    ):
        """Update vector memory with new knowledge"""
        
        try:
            # Only store high-confidence responses
            if confidence_assessment['confidence'] > 0.7:
                await self.vector_memory.add_document(
                    content=f"Q: {question}\nA: {response['content']}",
                    metadata={
                        'type': 'qa_pair',
                        'confidence': confidence_assessment['confidence'],
                        'model_used': response.get('model', 'unknown'),
                        'timestamp': time.time()
                    }
                )
        except Exception as e:
            # Log error but don't fail the main process
            print(f"Warning: Failed to update memory: {e}")
    
    async def _log_knowledge_gaps(self, gaps: List[str]):
        """Log knowledge gaps for future learning"""
        
        gap_file = Path(self.config.data_dir) / "gap_memory.json"
        
        try:
            # Load existing gaps
            if gap_file.exists():
                with open(gap_file, 'r') as f:
                    existing_gaps = json.load(f)
            else:
                existing_gaps = {}
            
            # Add new gaps with timestamps
            for gap in gaps:
                if gap not in existing_gaps:
                    existing_gaps[gap] = {
                        'first_encountered': time.time(),
                        'encounter_count': 1,
                        'priority': 'medium'
                    }
                else:
                    existing_gaps[gap]['encounter_count'] += 1
                    existing_gaps[gap]['last_encountered'] = time.time()
            
            # Save updated gaps
            gap_file.parent.mkdir(parents=True, exist_ok=True)
            with open(gap_file, 'w') as f:
                json.dump(existing_gaps, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to log knowledge gaps: {e}")
    
    def save_conversation(self, question: str, response: Dict):
        """Save conversation for context in future interactions"""
        
        conversation_id = f"conv_{int(time.time())}"
        self.conversation_memory[conversation_id] = {
            'question': question,
            'response': response,
            'timestamp': time.time()
        }
        
        # Keep only last 50 conversations in memory
        if len(self.conversation_memory) > 50:
            oldest_key = min(self.conversation_memory.keys(), 
                           key=lambda k: self.conversation_memory[k]['timestamp'])
            del self.conversation_memory[oldest_key]
