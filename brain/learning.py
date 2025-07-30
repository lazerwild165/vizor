#!/usr/bin/env python3
"""
Vizor Learning Engine
Handles self-learning capabilities and knowledge acquisition
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
from datetime import datetime, timedelta
import feedparser
import re

from .memory import VectorMemory
from .gap_detector import GapDetector
from models.llm_manager import LLMManager, TaskType

class LearningEngine:
    """
    Learning engine for Vizor's self-improvement capabilities
    
    Handles knowledge acquisition from various sources,
    gap detection, and continuous learning.
    """
    
    def __init__(self, config):
        self.config = config
        self.vector_memory = VectorMemory(config)
        self.gap_detector = GapDetector(config)
        self.llm_manager = LLMManager(config)
        
        # Learning sources configuration
        self.sources = {
            'cisa': {
                'url': 'https://www.cisa.gov/news-events/cybersecurity-advisories/feed',
                'type': 'rss',
                'enabled': True
            },
            'mitre': {
                'url': 'https://cve.mitre.org/data/downloads/allitems.csv',
                'type': 'csv',
                'enabled': True
            },
            'threatpost': {
                'url': 'https://threatpost.com/feed/',
                'type': 'rss',
                'enabled': True
            },
            'krebsonsecurity': {
                'url': 'https://krebsonsecurity.com/feed/',
                'type': 'rss',
                'enabled': True
            }
        }
        
        # Learning schedule
        self.last_learning_run = None
        self.learning_interval = config.get('learning_interval', 3600)  # 1 hour default
        
    async def learn_topic(self, topic: str, sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Learn about a specific topic
        
        Args:
            topic: Topic to learn about
            sources: Specific sources to use (None for all enabled)
            
        Returns:
            Learning results
        """
        try:
            results = {
            'topic': topic,
                'sources_consulted': [],
                'articles_found': 0,
                'knowledge_stored': 0,
                'confidence_gain': 0.0,
                'errors': []
            }
            
            # Determine sources to use
            if sources is None:
                sources = [name for name, config in self.sources.items() if config['enabled']]
            
            # Search for topic across sources
            for source_name in sources:
                if source_name not in self.sources:
                    results['errors'].append(f"Unknown source: {source_name}")
                    continue
                
                source_config = self.sources[source_name]
                if not source_config['enabled']:
                    continue
                
                try:
                    source_results = await self._search_source(topic, source_name, source_config)
                    results['sources_consulted'].append(source_name)
                    results['articles_found'] += source_results.get('articles_found', 0)
                    results['knowledge_stored'] += source_results.get('knowledge_stored', 0)
                        
                except Exception as e:
                    results['errors'].append(f"Error with {source_name}: {str(e)}")
            
            # Process and store knowledge
            if results['articles_found'] > 0:
                await self._process_and_store_knowledge(topic, results)
                results['confidence_gain'] = min(1.0, results['knowledge_stored'] / 10.0)
            
            # Update gap memory
            self.gap_detector.mark_topic_learned(topic)
            
            return results
            
        except Exception as e:
            return {
                'topic': topic,
                'error': str(e),
                'sources_consulted': [],
                'articles_found': 0,
                'knowledge_stored': 0,
                'confidence_gain': 0.0
            }
    
    async def continuous_learning(self) -> Dict[str, Any]:
        """
        Run continuous learning process
        
        Checks for new intelligence and updates knowledge base
        """
        try:
            # Check if it's time to learn
            if not self._should_run_learning():
                return {
                    'status': 'skipped',
                    'reason': 'Too soon since last run'
                }
            
            results = {
                'status': 'completed',
                'sources_checked': [],
                'new_articles': 0,
                'knowledge_updated': 0,
                'errors': []
            }
            
            # Check each enabled source
            for source_name, source_config in self.sources.items():
                if not source_config['enabled']:
                    continue
                
                try:
                    source_results = await self._check_source_updates(source_name, source_config)
                    results['sources_checked'].append(source_name)
                    results['new_articles'] += source_results.get('new_articles', 0)
                    results['knowledge_updated'] += source_results.get('knowledge_updated', 0)
                    
                except Exception as e:
                    results['errors'].append(f"Error with {source_name}: {str(e)}")
            
            # Update last run time
            self.last_learning_run = time.time()
            
            return results
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def learn_from_gaps(self) -> Dict[str, Any]:
        """
        Learn from detected knowledge gaps
        
        Automatically learns about topics that Vizor has low confidence in
        """
        try:
            # Get knowledge gaps
            gaps = self.gap_detector.get_knowledge_gaps()
            
            if not gaps:
                return {
                    'status': 'no_gaps',
                    'gaps_processed': 0
                }
            
            results = {
                'status': 'completed',
                'gaps_processed': 0,
                'successful_learnings': 0,
                'errors': []
            }
            
            # Learn about each gap
            for gap in gaps[:5]:  # Limit to 5 gaps per run
                try:
                    learning_result = await self.learn_topic(gap['topic'])
                    if learning_result.get('confidence_gain', 0) > 0.1:
                        results['successful_learnings'] += 1
                    results['gaps_processed'] += 1
                    
                except Exception as e:
                    results['errors'].append(f"Error learning {gap['topic']}: {str(e)}")
            
            return results
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _search_source(self, topic: str, source_name: str, source_config: Dict) -> Dict[str, Any]:
        """Search a specific source for topic information"""
        results = {
            'articles_found': 0,
            'knowledge_stored': 0,
            'articles': []
        }
        
        if source_config['type'] == 'rss':
            results = await self._search_rss_source(topic, source_name, source_config)
        elif source_config['type'] == 'csv':
            results = await self._search_csv_source(topic, source_name, source_config)
        elif source_config['type'] == 'api':
            results = await self._search_api_source(topic, source_name, source_config)
        
        return results
    
    async def _search_rss_source(self, topic: str, source_name: str, source_config: Dict) -> Dict[str, Any]:
        """Search RSS feed for topic"""
        try:
            # Parse RSS feed
            feed = feedparser.parse(source_config['url'])
            
            articles = []
            for entry in feed.entries:
                # Check if entry is relevant to topic
                if self._is_relevant_to_topic(entry, topic):
                    articles.append({
                        'title': entry.title,
                        'summary': entry.summary,
                        'link': entry.link,
                        'published': entry.published,
                        'source': source_name
                    })
            
            # Store relevant articles in vector memory
            knowledge_stored = 0
            for article in articles:
                try:
                    # Create knowledge entry
                    knowledge_entry = {
                        'type': 'article',
                        'topic': topic,
                        'title': article['title'],
                        'content': article['summary'],
                        'source': source_name,
                        'url': article['link'],
                        'timestamp': time.time()
                    }
                    
                    # Store in vector memory
                    await self.vector_memory.store_knowledge(knowledge_entry)
                    knowledge_stored += 1
                    
                except Exception as e:
                    print(f"Error storing article: {e}")
            
            return {
                'articles_found': len(articles),
                'knowledge_stored': knowledge_stored,
                'articles': articles
            }
            
        except Exception as e:
            print(f"Error searching RSS source {source_name}: {e}")
            return {
                'articles_found': 0,
                'knowledge_stored': 0,
                'articles': []
            }
    
    async def _search_csv_source(self, topic: str, source_name: str, source_config: Dict) -> Dict[str, Any]:
        """Search CSV source for topic"""
        try:
            # Download CSV file
            response = requests.get(source_config['url'], timeout=30)
            response.raise_for_status()
            
            # Parse CSV content
            lines = response.text.split('\n')
            articles = []
            
            for line in lines[1:]:  # Skip header
                if line.strip() and self._is_csv_line_relevant(line, topic):
                    # Parse CSV line (basic parsing)
                    parts = line.split(',')
                    if len(parts) >= 3:
                        articles.append({
                            'title': parts[0].strip('"'),
                            'summary': parts[1].strip('"'),
                            'link': parts[2].strip('"'),
                            'source': source_name
                        })
            
            # Store relevant entries
            knowledge_stored = 0
            for article in articles:
                try:
                    knowledge_entry = {
                        'type': 'cve',
                        'topic': topic,
                        'title': article['title'],
                        'content': article['summary'],
                        'source': source_name,
                        'url': article['link'],
                        'timestamp': time.time()
                    }
                    
                    await self.vector_memory.store_knowledge(knowledge_entry)
                    knowledge_stored += 1
                    
                except Exception as e:
                    print(f"Error storing CSV entry: {e}")
            
            return {
                'articles_found': len(articles),
                'knowledge_stored': knowledge_stored,
                'articles': articles
            }
            
        except Exception as e:
            print(f"Error searching CSV source {source_name}: {e}")
            return {
                'articles_found': 0,
                'knowledge_stored': 0,
                'articles': []
            }
    
    async def _search_api_source(self, topic: str, source_name: str, source_config: Dict) -> Dict[str, Any]:
        """Search API source for topic"""
        # This would implement API-specific searching
        # For now, return empty results
        return {
            'articles_found': 0,
            'knowledge_stored': 0,
            'articles': []
        }
    
    def _is_relevant_to_topic(self, entry, topic: str) -> bool:
        """Check if RSS entry is relevant to topic"""
        # Simple keyword matching
        topic_keywords = topic.lower().split()
        entry_text = f"{entry.title} {entry.summary}".lower()
        
        # Check if any topic keyword appears in entry
        for keyword in topic_keywords:
            if keyword in entry_text:
                return True
        
        return False
    
    def _is_csv_line_relevant(self, line: str, topic: str) -> bool:
        """Check if CSV line is relevant to topic"""
        topic_keywords = topic.lower().split()
        line_lower = line.lower()
        
        for keyword in topic_keywords:
            if keyword in line_lower:
                return True
        
        return False
    
    async def _check_source_updates(self, source_name: str, source_config: Dict) -> Dict[str, Any]:
        """Check for updates from a source"""
        # This would check for new content since last check
        # For now, return basic results
        return {
            'new_articles': 0,
            'knowledge_updated': 0
        }
    
    async def _process_and_store_knowledge(self, topic: str, results: Dict[str, Any]):
        """Process and store learned knowledge"""
        # This would process the raw articles and extract structured knowledge
        # For now, just log the results
        print(f"Processed {results['knowledge_stored']} knowledge items for topic: {topic}")
    
    def _should_run_learning(self) -> bool:
        """Check if continuous learning should run"""
        if self.last_learning_run is None:
            return True
        
        time_since_last = time.time() - self.last_learning_run
        return time_since_last >= self.learning_interval
    
    def add_source(self, name: str, url: str, source_type: str, enabled: bool = True):
        """Add a new learning source"""
        self.sources[name] = {
            'url': url,
            'type': source_type,
            'enabled': enabled
        }
    
    def remove_source(self, name: str):
        """Remove a learning source"""
        if name in self.sources:
            del self.sources[name]
    
    def enable_source(self, name: str):
        """Enable a learning source"""
        if name in self.sources:
            self.sources[name]['enabled'] = True
    
    def disable_source(self, name: str):
        """Disable a learning source"""
        if name in self.sources:
            self.sources[name]['enabled'] = False
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'sources_configured': len(self.sources),
            'sources_enabled': len([s for s in self.sources.values() if s['enabled']]),
            'last_learning_run': self.last_learning_run,
            'learning_interval': self.learning_interval
        }
