#!/usr/bin/env python3
"""
Vizor Knowledge Gap Detector
Identifies knowledge gaps and triggers learning flows
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

class GapDetector:
    """
    Detects knowledge gaps in responses and triggers learning
    
    This component analyzes responses for uncertainty indicators,
    identifies specific knowledge domains that need improvement,
    and maintains a gap memory for prioritized learning.
    """
    
    def __init__(self, config):
        self.config = config
        self.gap_memory_file = config.data_dir / "gap_memory.json"
        
        # Uncertainty indicators
        self.uncertainty_patterns = [
            r"i don't know",
            r"i'm not sure",
            r"uncertain",
            r"unclear",
            r"might be",
            r"could be",
            r"possibly",
            r"perhaps",
            r"not familiar with",
            r"limited information",
            r"insufficient data"
        ]
        
        # Knowledge domains for cybersecurity
        self.knowledge_domains = {
            'malware_analysis': ['malware', 'virus', 'trojan', 'ransomware', 'payload'],
            'network_security': ['firewall', 'ids', 'ips', 'network', 'traffic'],
            'incident_response': ['incident', 'response', 'forensics', 'investigation'],
            'threat_intelligence': ['threat', 'actor', 'campaign', 'ioc', 'indicator'],
            'vulnerability_management': ['vulnerability', 'cve', 'patch', 'exploit'],
            'compliance': ['compliance', 'audit', 'regulation', 'standard'],
            'cryptography': ['encryption', 'crypto', 'hash', 'certificate', 'key'],
            'cloud_security': ['cloud', 'aws', 'azure', 'gcp', 'container'],
            'application_security': ['application', 'web', 'api', 'code', 'software'],
            'identity_management': ['identity', 'authentication', 'authorization', 'iam']
        }
    
    async def assess_knowledge_gaps(
        self, 
        question: str, 
        relevant_docs: List[Dict], 
        analysis: Dict
    ) -> Dict[str, Any]:
        """
        Assess knowledge gaps for a given question
        
        Args:
            question: Original user question
            relevant_docs: Documents found in vector memory
            analysis: Query analysis results
            
        Returns:
            Gap assessment with coverage score and identified gaps
        """
        
        # Calculate coverage score based on relevant documents
        coverage_score = self._calculate_coverage_score(question, relevant_docs, analysis)
        
        # Identify specific knowledge gaps
        gaps = await self._identify_domain_gaps(question, analysis, coverage_score)
        
        # Check for technical complexity gaps
        technical_gaps = self._identify_technical_gaps(question, analysis)
        gaps.extend(technical_gaps)
        
        return {
            'coverage_score': coverage_score,
            'gaps': list(set(gaps)),  # Remove duplicates
            'gap_count': len(gaps),
            'needs_learning': coverage_score < 0.6 or len(gaps) > 0
        }
    
    def _calculate_coverage_score(
        self, 
        question: str, 
        relevant_docs: List[Dict], 
        analysis: Dict
    ) -> float:
        """Calculate how well existing knowledge covers the question"""
        
        base_score = 0.0
        
        # Score based on number of relevant documents
        if relevant_docs:
            doc_score = min(len(relevant_docs) * 0.2, 0.8)
            base_score += doc_score
        
        # Score based on document relevance/similarity
        if relevant_docs:
            avg_similarity = sum(doc.get('similarity', 0) for doc in relevant_docs) / len(relevant_docs)
            base_score += avg_similarity * 0.3
        
        # Penalty for high complexity with low document coverage
        if analysis.get('complexity') == 'high' and len(relevant_docs) < 2:
            base_score -= 0.3
        
        # Penalty for specialized domains with no coverage
        domain = analysis.get('domain', 'general')
        if domain != 'general' and not relevant_docs:
            base_score -= 0.4
        
        return max(0.0, min(1.0, base_score))
    
    async def _identify_domain_gaps(
        self, 
        question: str, 
        analysis: Dict, 
        coverage_score: float
    ) -> List[str]:
        """Identify knowledge gaps in specific cybersecurity domains"""
        
        gaps = []
        question_lower = question.lower()
        
        # Check each knowledge domain
        for domain, keywords in self.knowledge_domains.items():
            if any(keyword in question_lower for keyword in keywords):
                # This question relates to this domain
                if coverage_score < 0.5:
                    gaps.append(domain)
        
        # Check for emerging threat patterns
        emerging_patterns = [
            'zero-day', 'apt', 'supply chain', 'ai security', 
            'quantum cryptography', 'iot security', 'devsecops'
        ]
        
        for pattern in emerging_patterns:
            if pattern in question_lower and coverage_score < 0.7:
                gaps.append(f"emerging_threats_{pattern.replace(' ', '_')}")
        
        return gaps
    
    def _identify_technical_gaps(self, question: str, analysis: Dict) -> List[str]:
        """Identify technical complexity gaps"""
        
        gaps = []
        
        # Check for code-related gaps
        if 'code' in question.lower() or analysis.get('domain') == 'application_security':
            code_patterns = ['exploit', 'payload', 'shellcode', 'reverse engineering']
            if any(pattern in question.lower() for pattern in code_patterns):
                gaps.append('advanced_code_analysis')
        
        # Check for tool-specific gaps
        security_tools = [
            'metasploit', 'nmap', 'wireshark', 'burp suite', 'ida pro',
            'ghidra', 'volatility', 'yara', 'sigma', 'suricata'
        ]
        
        for tool in security_tools:
            if tool in question.lower():
                gaps.append(f'tool_expertise_{tool.replace(" ", "_")}')
        
        return gaps
    
    async def identify_specific_gaps(
        self, 
        question: str, 
        response: Dict, 
        confidence_assessment: Dict
    ) -> List[str]:
        """Identify specific knowledge gaps from response analysis"""
        
        gaps = []
        response_content = response.get('content', '').lower()
        
        # Check for uncertainty patterns in response
        for pattern in self.uncertainty_patterns:
            if re.search(pattern, response_content):
                gaps.append(f"uncertainty_{pattern.replace(' ', '_')}")
        
        # Extract specific topics mentioned as unknown
        unknown_patterns = [
            r"don't know about (.+?)(?:\.|,|$)",
            r"unfamiliar with (.+?)(?:\.|,|$)",
            r"need more information about (.+?)(?:\.|,|$)"
        ]
        
        for pattern in unknown_patterns:
            matches = re.findall(pattern, response_content)
            for match in matches:
                clean_topic = match.strip().replace(' ', '_')
                gaps.append(f"unknown_topic_{clean_topic}")
        
        # Add domain-specific gaps based on question analysis
        question_gaps = await self._analyze_question_for_gaps(question)
        gaps.extend(question_gaps)
        
        return list(set(gaps))  # Remove duplicates
    
    async def _analyze_question_for_gaps(self, question: str) -> List[str]:
        """Analyze question content for potential knowledge gaps"""
        
        gaps = []
        question_lower = question.lower()
        
        # Check for specific threat actors
        threat_actors = [
            'apt1', 'apt28', 'apt29', 'lazarus', 'carbanak', 'fin7',
            'cozy bear', 'fancy bear', 'equation group'
        ]
        
        for actor in threat_actors:
            if actor in question_lower:
                gaps.append(f"threat_actor_{actor.replace(' ', '_')}")
        
        # Check for specific malware families
        malware_families = [
            'stuxnet', 'wannacry', 'notpetya', 'emotet', 'trickbot',
            'ryuk', 'maze', 'conti', 'lockbit'
        ]
        
        for malware in malware_families:
            if malware in question_lower:
                gaps.append(f"malware_family_{malware}")
        
        # Check for specific vulnerabilities
        if re.search(r'cve-\d{4}-\d+', question_lower):
            gaps.append('specific_vulnerability_analysis')
        
        return gaps
    
    async def get_gap_priorities(self) -> List[Dict[str, Any]]:
        """Get prioritized list of knowledge gaps for learning"""
        
        try:
            if not self.gap_memory_file.exists():
                return []
            
            with open(self.gap_memory_file, 'r') as f:
                gaps = json.load(f)
            
            # Calculate priority scores
            prioritized_gaps = []
            current_time = time.time()
            
            for gap_name, gap_data in gaps.items():
                # Priority based on encounter frequency and recency
                encounter_count = gap_data.get('encounter_count', 1)
                last_encountered = gap_data.get('last_encountered', gap_data.get('first_encountered', current_time))
                
                # Time decay factor (more recent = higher priority)
                time_diff = current_time - last_encountered
                time_factor = max(0.1, 1.0 - (time_diff / (7 * 24 * 3600)))  # Decay over 7 days
                
                # Frequency factor
                frequency_factor = min(1.0, encounter_count / 10.0)
                
                # Domain importance factor
                domain_importance = self._get_domain_importance(gap_name)
                
                priority_score = (frequency_factor * 0.4 + 
                                time_factor * 0.4 + 
                                domain_importance * 0.2)
                
                prioritized_gaps.append({
                    'name': gap_name,
                    'priority_score': priority_score,
                    'encounter_count': encounter_count,
                    'last_encountered': last_encountered,
                    'suggested_action': self._suggest_learning_action(gap_name)
                })
            
            # Sort by priority score
            prioritized_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return prioritized_gaps[:20]  # Return top 20
            
        except Exception as e:
            print(f"Warning: Could not load gap priorities: {e}")
            return []
    
    def _get_domain_importance(self, gap_name: str) -> float:
        """Get importance score for a knowledge domain"""
        
        # High importance domains
        high_importance = [
            'malware_analysis', 'threat_intelligence', 'incident_response',
            'vulnerability_management', 'network_security'
        ]
        
        # Medium importance domains
        medium_importance = [
            'application_security', 'cloud_security', 'cryptography'
        ]
        
        for domain in high_importance:
            if domain in gap_name:
                return 1.0
        
        for domain in medium_importance:
            if domain in gap_name:
                return 0.7
        
        # Emerging threats get high priority
        if 'emerging_threats' in gap_name:
            return 0.9
        
        # Tool expertise gets medium priority
        if 'tool_expertise' in gap_name:
            return 0.6
        
        return 0.5  # Default importance
    
    def _suggest_learning_action(self, gap_name: str) -> str:
        """Suggest specific learning action for a gap"""
        
        if 'malware' in gap_name:
            return "Study recent malware analysis reports and techniques"
        elif 'threat_actor' in gap_name:
            return "Research threat actor profiles and TTPs"
        elif 'vulnerability' in gap_name:
            return "Review CVE details and exploitation techniques"
        elif 'tool_expertise' in gap_name:
            return "Practice with security tools and read documentation"
        elif 'emerging_threats' in gap_name:
            return "Monitor threat intelligence feeds and security blogs"
        else:
            return "Gather more information from authoritative sources"
    
    async def update_gap_memory(self, gaps: List[str]):
        """Update gap memory with new gaps"""
        
        try:
            # Load existing gaps
            if self.gap_memory_file.exists():
                with open(self.gap_memory_file, 'r') as f:
                    existing_gaps = json.load(f)
            else:
                existing_gaps = {}
            
            current_time = time.time()
            
            # Update gaps
            for gap in gaps:
                if gap in existing_gaps:
                    existing_gaps[gap]['encounter_count'] += 1
                    existing_gaps[gap]['last_encountered'] = current_time
                else:
                    existing_gaps[gap] = {
                        'first_encountered': current_time,
                        'last_encountered': current_time,
                        'encounter_count': 1,
                        'priority': 'medium',
                        'learning_status': 'pending'
                    }
            
            # Save updated gaps
            self.gap_memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.gap_memory_file, 'w') as f:
                json.dump(existing_gaps, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not update gap memory: {e}")
    
    async def mark_gap_as_learned(self, gap_name: str):
        """Mark a knowledge gap as learned/resolved"""
        
        try:
            if not self.gap_memory_file.exists():
                return
            
            with open(self.gap_memory_file, 'r') as f:
                gaps = json.load(f)
            
            if gap_name in gaps:
                gaps[gap_name]['learning_status'] = 'learned'
                gaps[gap_name]['learned_timestamp'] = time.time()
                
                with open(self.gap_memory_file, 'w') as f:
                    json.dump(gaps, f, indent=2)
                    
        except Exception as e:
            print(f"Warning: Could not mark gap as learned: {e}")
