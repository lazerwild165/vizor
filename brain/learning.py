#!/usr/bin/env python3
"""
Vizor Learning Engine
Automated learning and knowledge acquisition system
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
from datetime import datetime, timedelta

from brain.memory import VectorMemory
from brain.gap_detector import KnowledgeGapDetector
from models.llm_manager import LLMManager, TaskType

class LearningEngine:
    """
    Automated learning system that fetches intelligence and updates knowledge
    
    This engine:
    1. Monitors knowledge gaps and triggers learning flows
    2. Fetches intelligence from configured sources
    3. Processes and stores new knowledge in vector memory
    4. Adapts to user preferences and learning patterns
    """
    
    def __init__(self, config):
        self.config = config
        self.vector_memory = VectorMemory(config)
        self.gap_detector = KnowledgeGapDetector(config)
        self.llm_manager = LLMManager(config)
        
        # Learning sources configuration
        self.learning_sources = {
            'cisa.gov': {
                'url': 'https://www.cisa.gov/news-events/cybersecurity-advisories',
                'type': 'threat_advisory',
                'parser': self._parse_cisa_content
            },
            'mitre.org': {
                'url': 'https://attack.mitre.org/techniques/',
                'type': 'technique_knowledge',
                'parser': self._parse_mitre_content
            },
            'nvd.nist.gov': {
                'url': 'https://nvd.nist.gov/vuln/data-feeds',
                'type': 'vulnerability_data',
                'parser': self._parse_nvd_content
            }
        }
    
    async def learn_topic(
        self, 
        topic: str, 
        sources: Optional[List[str]] = None,
        update_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Learn about a specific topic
        
        Args:
            topic: Topic to learn about
            sources: Specific sources to use (use all if None)
            update_memory: Whether to update vector memory
            
        Returns:
            Learning results with sources processed and items added
        """
        
        if not sources:
            sources = list(self.learning_sources.keys())
        
        learning_results = {
            'topic': topic,
            'sources_processed': 0,
            'sources_count': len(sources),
            'items_added': 0,
            'errors': [],
            'start_time': time.time()
        }
        
        try:
            # Use Mistral for learning coordination and analysis
            learning_plan = await self._create_learning_plan(topic, sources)
            
            # Process each source
            for source in sources:
                try:
                    source_results = await self._process_learning_source(
                        source, topic, learning_plan
                    )
                    
                    learning_results['sources_processed'] += 1
                    learning_results['items_added'] += source_results['items_added']
                    
                    if update_memory:
                        await self._store_learning_results(topic, source, source_results)
                        
                except Exception as e:
                    learning_results['errors'].append(f"{source}: {str(e)}")
            
            # Mark related gaps as learned
            await self._update_gap_status(topic)
            
            learning_results['processing_time'] = time.time() - learning_results['start_time']
            
            return learning_results
            
        except Exception as e:
            learning_results['errors'].append(f"Learning engine error: {str(e)}")
            return learning_results
    
    async def _create_learning_plan(self, topic: str, sources: List[str]) -> Dict[str, Any]:
        """Create a learning plan using Mistral for strategic thinking"""
        
        planning_prompt = f"""
        Create a learning plan for the cybersecurity topic: "{topic}"
        
        Available sources: {', '.join(sources)}
        
        Provide a JSON response with:
        {{
            "learning_objectives": ["list", "of", "specific", "objectives"],
            "key_concepts": ["concepts", "to", "focus", "on"],
            "source_priorities": {{"source": "priority_level"}},
            "search_keywords": ["keywords", "for", "content", "filtering"],
            "expected_outcomes": ["what", "should", "be", "learned"]
        }}
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=planning_prompt,
                model_name="mistral",
                task_type=TaskType.LEARNING,
                temperature=0.3
            )
            
            return json.loads(response['content'])
            
        except Exception as e:
            # Fallback plan
            return {
                "learning_objectives": [f"Understand {topic}"],
                "key_concepts": [topic],
                "source_priorities": {source: "medium" for source in sources},
                "search_keywords": [topic.lower()],
                "expected_outcomes": [f"Improved knowledge of {topic}"]
            }
    
    async def _process_learning_source(
        self, 
        source: str, 
        topic: str, 
        learning_plan: Dict
    ) -> Dict[str, Any]:
        """Process a specific learning source"""
        
        if source not in self.learning_sources:
            raise Exception(f"Unknown learning source: {source}")
        
        source_config = self.learning_sources[source]
        
        # Simulate content fetching (in real implementation, would fetch from APIs/RSS)
        content_items = await self._fetch_source_content(source, topic, learning_plan)
        
        # Process and analyze content
        processed_items = []
        for item in content_items:
            processed_item = await self._analyze_content_item(item, topic, learning_plan)
            if processed_item:
                processed_items.append(processed_item)
        
        return {
            'source': source,
            'items_fetched': len(content_items),
            'items_processed': len(processed_items),
            'items_added': len(processed_items),
            'processed_items': processed_items
        }
    
    async def _fetch_source_content(
        self, 
        source: str, 
        topic: str, 
        learning_plan: Dict
    ) -> List[Dict[str, Any]]:
        """Fetch content from a learning source"""
        
        # In a real implementation, this would fetch from actual APIs
        # For now, we'll simulate content based on the source type
        
        simulated_content = []
        
        if source == 'cisa.gov':
            simulated_content = [
                {
                    'title': f'CISA Advisory: {topic} Threat Analysis',
                    'content': f'Recent analysis of {topic} threats and mitigation strategies.',
                    'type': 'advisory',
                    'date': datetime.now().isoformat(),
                    'url': f'https://cisa.gov/advisory/{topic.lower()}'
                }
            ]
        
        elif source == 'mitre.org':
            simulated_content = [
                {
                    'title': f'MITRE ATT&CK: {topic} Techniques',
                    'content': f'Detailed analysis of {topic} attack techniques and detection methods.',
                    'type': 'technique',
                    'date': datetime.now().isoformat(),
                    'url': f'https://attack.mitre.org/techniques/{topic.lower()}'
                }
            ]
        
        elif source == 'nvd.nist.gov':
            simulated_content = [
                {
                    'title': f'NVD Vulnerability Data: {topic}',
                    'content': f'Vulnerability information related to {topic} with CVSS scores and mitigation.',
                    'type': 'vulnerability',
                    'date': datetime.now().isoformat(),
                    'url': f'https://nvd.nist.gov/vuln/search/results?query={topic}'
                }
            ]
        
        return simulated_content
    
    async def _analyze_content_item(
        self, 
        item: Dict[str, Any], 
        topic: str, 
        learning_plan: Dict
    ) -> Optional[Dict[str, Any]]:
        """Analyze a content item for relevance and extract key information"""
        
        # Use Phi3 for quick content analysis
        analysis_prompt = f"""
        Analyze this cybersecurity content for relevance to the topic "{topic}":
        
        Title: {item['title']}
        Content: {item['content']}
        
        Provide analysis in JSON format:
        {{
            "relevance_score": 0.0-1.0,
            "key_insights": ["list", "of", "key", "insights"],
            "actionable_items": ["actionable", "recommendations"],
            "threat_level": "low|medium|high|critical",
            "summary": "brief summary of content"
        }}
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=analysis_prompt,
                model_name="phi3",
                task_type=TaskType.LEARNING,
                temperature=0.3
            )
            
            analysis = json.loads(response['content'])
            
            # Only keep relevant content
            if analysis.get('relevance_score', 0) >= 0.6:
                return {
                    'original_item': item,
                    'analysis': analysis,
                    'processed_content': self._format_learning_content(item, analysis),
                    'metadata': {
                        'topic': topic,
                        'source': item.get('url', 'unknown'),
                        'relevance_score': analysis.get('relevance_score', 0),
                        'threat_level': analysis.get('threat_level', 'unknown'),
                        'processing_timestamp': time.time()
                    }
                }
            
            return None
            
        except Exception as e:
            # Fallback: include content with basic metadata
            return {
                'original_item': item,
                'analysis': {'relevance_score': 0.5, 'summary': item['content'][:200]},
                'processed_content': item['content'],
                'metadata': {
                    'topic': topic,
                    'source': item.get('url', 'unknown'),
                    'relevance_score': 0.5,
                    'processing_timestamp': time.time(),
                    'processing_error': str(e)
                }
            }
    
    def _format_learning_content(self, item: Dict, analysis: Dict) -> str:
        """Format content for storage in vector memory"""
        
        formatted_content = f"""
Learning Content: {item['title']}
Source: {item.get('url', 'Unknown')}
Date: {item.get('date', 'Unknown')}

Summary: {analysis.get('summary', 'No summary available')}

Key Insights:
{chr(10).join(f"• {insight}" for insight in analysis.get('key_insights', []))}

Actionable Items:
{chr(10).join(f"• {action}" for action in analysis.get('actionable_items', []))}

Threat Level: {analysis.get('threat_level', 'Unknown')}

Full Content:
{item['content']}
        """.strip()
        
        return formatted_content
    
    async def _store_learning_results(
        self, 
        topic: str, 
        source: str, 
        results: Dict[str, Any]
    ):
        """Store learning results in vector memory"""
        
        for item in results['processed_items']:
            await self.vector_memory.add_learning_content(
                topic=topic,
                content=item['processed_content'],
                source=source
            )
    
    async def _update_gap_status(self, topic: str):
        """Update knowledge gap status after learning"""
        
        # Find related gaps and mark as learned
        gap_priorities = await self.gap_detector.get_gap_priorities()
        
        for gap in gap_priorities:
            gap_name = gap['name']
            if (topic.lower() in gap_name.lower() or 
                any(keyword in gap_name.lower() for keyword in topic.lower().split())):
                await self.gap_detector.mark_gap_as_learned(gap_name)
    
    async def auto_learn_from_gaps(self) -> Dict[str, Any]:
        """Automatically learn from highest priority knowledge gaps"""
        
        gap_priorities = await self.gap_detector.get_gap_priorities()
        
        if not gap_priorities:
            return {
                'gaps_processed': 0,
                'learning_sessions': 0,
                'message': 'No knowledge gaps identified for learning'
            }
        
        # Learn from top 3 priority gaps
        top_gaps = gap_priorities[:3]
        learning_results = []
        
        for gap in top_gaps:
            gap_name = gap['name']
            
            # Convert gap name to learning topic
            topic = self._gap_to_topic(gap_name)
            
            if topic:
                result = await self.learn_topic(topic)
                learning_results.append({
                    'gap': gap_name,
                    'topic': topic,
                    'result': result
                })
        
        return {
            'gaps_processed': len(top_gaps),
            'learning_sessions': len(learning_results),
            'results': learning_results,
            'message': f'Completed auto-learning for {len(learning_results)} knowledge gaps'
        }
    
    def _gap_to_topic(self, gap_name: str) -> Optional[str]:
        """Convert a knowledge gap name to a learning topic"""
        
        # Remove prefixes and clean up gap names
        topic = gap_name
        
        prefixes_to_remove = [
            'uncertainty_', 'unknown_topic_', 'malware_family_',
            'threat_actor_', 'tool_expertise_', 'emerging_threats_'
        ]
        
        for prefix in prefixes_to_remove:
            if topic.startswith(prefix):
                topic = topic[len(prefix):]
                break
        
        # Convert underscores to spaces
        topic = topic.replace('_', ' ')
        
        # Capitalize properly
        topic = ' '.join(word.capitalize() for word in topic.split())
        
        return topic if len(topic) > 2 else None
    
    async def schedule_periodic_learning(self):
        """Schedule periodic learning based on configuration"""
        
        if not self.config.learning_config.auto_learn_enabled:
            return
        
        interval_hours = self.config.learning_config.update_interval_hours
        
        while True:
            try:
                # Auto-learn from gaps
                await self.auto_learn_from_gaps()
                
                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                print(f"Periodic learning error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    # Parser methods for different sources (simplified for demo)
    def _parse_cisa_content(self, content: str) -> List[Dict]:
        """Parse CISA advisory content"""
        return [{'type': 'cisa_advisory', 'content': content}]
    
    def _parse_mitre_content(self, content: str) -> List[Dict]:
        """Parse MITRE ATT&CK content"""
        return [{'type': 'mitre_technique', 'content': content}]
    
    def _parse_nvd_content(self, content: str) -> List[Dict]:
        """Parse NVD vulnerability content"""
        return [{'type': 'nvd_vulnerability', 'content': content}]
