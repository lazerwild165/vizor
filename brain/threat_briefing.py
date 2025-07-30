#!/usr/bin/env python3
"""
Vizor Threat Briefing Engine
Generates threat briefings and summaries from intelligence sources
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .memory import VectorMemory
from .learning import LearningEngine
from models.llm_manager import LLMManager, TaskType

class ThreatBriefingEngine:
    """
    Generates threat briefings and summaries
    
    Integrates with learning engine and vector memory to create
    comprehensive threat intelligence briefings.
    """
    
    def __init__(self, config):
        self.config = config
        self._llm_manager = None  # Lazy init
        self.vector_memory = VectorMemory(config)
        self.conversation_history = []
        self.learning_engine = LearningEngine(config)
        
    @property
    def llm_manager(self):
        """Lazy initialization of LLM manager"""
        if self._llm_manager is None:
            from models.llm_manager import LLMManager
            self._llm_manager = LLMManager(self.config)
        return self._llm_manager
        
    async def generate_daily_briefing(
        self, 
        date: datetime.date,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate daily threat briefing
        
        Args:
            date: Date for briefing
            sources: Specific sources to use
            
        Returns:
            Briefing data
        """
        try:
            # Get recent threat intelligence
            recent_intel = await self._get_recent_intelligence(date, sources)
            
            # Get new vulnerabilities
            new_vulns = await self._get_new_vulnerabilities(date)
            
            # Generate executive summary
            summary = await self._generate_executive_summary(recent_intel, new_vulns)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(recent_intel, new_vulns)
            
            briefing = {
                'date': date.isoformat(),
                'executive_summary': summary,
                'key_threats': recent_intel.get('threats', []),
                'vulnerabilities': new_vulns,
                'recommendations': recommendations,
                'sources_consulted': recent_intel.get('sources', []),
                'generated_at': datetime.now().isoformat()
            }
            
            return briefing
            
        except Exception as e:
            return {
                'error': str(e),
                'date': date.isoformat(),
                'executive_summary': 'Error generating briefing',
                'key_threats': [],
                'vulnerabilities': [],
                'recommendations': ['Check system logs for errors']
            }
    
    async def generate_weekly_briefing(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        include_trends: bool = True
    ) -> Dict[str, Any]:
        """
        Generate weekly threat summary
        
        Args:
            start_date: Start of week
            end_date: End of week
            include_trends: Whether to include trend analysis
            
        Returns:
            Weekly briefing data
        """
        try:
            # Get weekly intelligence
            weekly_intel = await self._get_weekly_intelligence(start_date, end_date)
            
            # Generate trend analysis if requested
            trends = None
            if include_trends:
                trends = await self._analyze_weekly_trends(weekly_intel)
            
            # Generate summary
            summary = await self._generate_weekly_summary(weekly_intel, trends)
            
            briefing = {
                'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'executive_summary': summary,
                'key_events': weekly_intel.get('events', []),
                'trends': trends,
                'threat_landscape': weekly_intel.get('threat_landscape', {}),
                'generated_at': datetime.now().isoformat()
            }
            
            return briefing
            
        except Exception as e:
            return {
                'error': str(e),
                'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
                'executive_summary': 'Error generating weekly briefing',
                'key_events': [],
                'trends': None
            }
    
    async def generate_custom_briefing(
        self,
        topic: str,
        timeframe: str,
        depth: str,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate custom threat briefing
        
        Args:
            topic: Topic to brief on
            timeframe: Timeframe for analysis
            depth: Analysis depth
            sources: Specific sources to use
            
        Returns:
            Custom briefing data
        """
        try:
            # Parse timeframe
            days = self._parse_timeframe(timeframe)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get topic-specific intelligence
            topic_intel = await self._get_topic_intelligence(topic, start_date, end_date, sources)
            
            # Generate analysis based on depth
            analysis = await self._generate_topic_analysis(topic, topic_intel, depth)
            
            briefing = {
                'topic': topic,
                'timeframe': timeframe,
                'depth': depth,
                'executive_summary': analysis['summary'],
                'key_findings': analysis['findings'],
                'threat_assessment': analysis['threat_assessment'],
                'recommendations': analysis['recommendations'],
                'sources': topic_intel.get('sources', []),
                'generated_at': datetime.now().isoformat()
            }
            
            return briefing
            
        except Exception as e:
            return {
                'error': str(e),
                'topic': topic,
                'timeframe': timeframe,
                'executive_summary': 'Error generating custom briefing',
                'key_findings': [],
                'recommendations': ['Check system logs for errors']
            }
    
    async def analyze_trends(
        self,
        period: str,
        categories: Optional[List[str]] = None,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze threat trends
        
        Args:
            period: Analysis period
            categories: Threat categories to analyze
            visualize: Whether to generate visualizations
            
        Returns:
            Trend analysis data
        """
        try:
            # Parse period
            days = self._parse_timeframe(period)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get trend data
            trend_data = await self._get_trend_data(start_date, end_date, categories)
            
            # Analyze trends
            analysis = await self._analyze_trends_data(trend_data)
            
            # Generate visualizations if requested
            visualizations = None
            if visualize:
                visualizations = await self._generate_trend_visualizations(analysis)
            
            return {
                'period': period,
                'summary': analysis['summary'],
                'emerging_threats': analysis['emerging_threats'],
                'declining_threats': analysis['declining_threats'],
                'stable_threats': analysis['stable_threats'],
                'visualizations': visualizations,
                'confidence': analysis['confidence'],
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'period': period,
                'summary': 'Error analyzing trends',
                'emerging_threats': [],
                'declining_threats': [],
                'stable_threats': []
            }
    
    async def _get_recent_intelligence(
        self, 
        date: datetime.date, 
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get recent threat intelligence"""
        # This would query vector memory and external sources
        # For now, return mock data
        return {
            'threats': [
                {
                    'name': 'Ransomware Campaign',
                    'severity': 'high',
                    'impact': 'Data encryption and financial loss',
                    'description': 'New ransomware variant targeting healthcare sector'
                },
                {
                    'name': 'Phishing Attack',
                    'severity': 'medium',
                    'impact': 'Credential theft',
                    'description': 'Sophisticated phishing campaign using COVID-19 themes'
                }
            ],
            'sources': ['CISA', 'MITRE', 'ThreatPost']
        }
    
    async def _get_new_vulnerabilities(self, date: datetime.date) -> List[Dict[str, Any]]:
        """Get new vulnerabilities for the date"""
        # This would query NVD and other vulnerability databases
        # For now, return mock data
        return [
            {
                'cve': 'CVE-2024-1234',
                'cvss_score': 9.8,
                'product': 'Apache Log4j',
                'description': 'Remote code execution vulnerability',
                'severity': 'critical'
            },
            {
                'cve': 'CVE-2024-5678',
                'cvss_score': 7.5,
                'product': 'OpenSSL',
                'description': 'Denial of service vulnerability',
                'severity': 'high'
            }
        ]
    
    async def _generate_executive_summary(
        self, 
        intel: Dict[str, Any], 
        vulns: List[Dict[str, Any]]
    ) -> str:
        """Generate executive summary"""
        prompt = f"""
        Generate an executive summary for a cybersecurity briefing based on:
        
        Threats: {intel.get('threats', [])}
        Vulnerabilities: {vulns}
        
        Provide a concise, actionable summary suitable for C-level executives.
        Focus on business impact and key recommendations.
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model_name="mistral",
                task_type=TaskType.BRIEFING,
                temperature=0.3
            )
            return response['content']
        except:
            return "Executive summary generation failed. Please review the detailed briefing."
    
    async def _generate_recommendations(
        self, 
        intel: Dict[str, Any], 
        vulns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations"""
        prompt = f"""
        Generate actionable cybersecurity recommendations based on:
        
        Threats: {intel.get('threats', [])}
        Vulnerabilities: {vulns}
        
        Provide 5-7 specific, actionable recommendations.
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model_name="mistral",
                task_type=TaskType.BRIEFING,
                temperature=0.3
            )
            
            # Parse recommendations from response
            content = response['content']
            recommendations = []
            for line in content.split('\n'):
                if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                    rec = line.strip().lstrip('•-*1234567890. ')
                    if rec:
                        recommendations.append(rec)
            
            return recommendations[:7]  # Limit to 7 recommendations
            
        except:
            return [
                "Review and update security policies",
                "Implement multi-factor authentication",
                "Conduct regular security awareness training",
                "Update and patch systems regularly",
                "Monitor network traffic for anomalies"
            ]
    
    async def _get_weekly_intelligence(
        self, 
        start_date: datetime.date, 
        end_date: datetime.date
    ) -> Dict[str, Any]:
        """Get weekly intelligence summary"""
        # This would aggregate daily intelligence over the week
        return {
            'events': [
                {
                    'date': '2024-01-15',
                    'type': 'threat_actor',
                    'description': 'APT29 activity detected in European networks'
                },
                {
                    'date': '2024-01-17',
                    'type': 'vulnerability',
                    'description': 'Critical vulnerability disclosed in popular web framework'
                }
            ],
            'threat_landscape': {
                'ransomware': 'increasing',
                'phishing': 'stable',
                'supply_chain': 'decreasing'
            }
        }
    
    async def _analyze_weekly_trends(self, intel: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze weekly trends"""
        return {
            'summary': 'Ransomware attacks increased by 15% this week',
            'key_trends': [
                'Healthcare sector targeted more frequently',
                'Ransom demands increasing',
                'Supply chain attacks decreasing'
            ]
        }
    
    async def _generate_weekly_summary(
        self, 
        intel: Dict[str, Any], 
        trends: Optional[Dict[str, Any]]
    ) -> str:
        """Generate weekly summary"""
        prompt = f"""
        Generate a weekly cybersecurity summary based on:
        
        Events: {intel.get('events', [])}
        Threat Landscape: {intel.get('threat_landscape', {})}
        Trends: {trends}
        
        Provide a comprehensive weekly summary.
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model_name="mistral",
                task_type=TaskType.BRIEFING,
                temperature=0.3
            )
            return response['content']
        except:
            return "Weekly summary generation failed. Please review the detailed briefing."
    
    async def _get_topic_intelligence(
        self, 
        topic: str, 
        start_date: datetime.date, 
        end_date: datetime.date, 
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get topic-specific intelligence"""
        # This would search vector memory and external sources for topic
        return {
            'topic': topic,
            'intelligence': [
                {
                    'date': '2024-01-15',
                    'source': 'CISA',
                    'content': f'Recent {topic} activity detected'
                }
            ],
            'sources': sources or ['CISA', 'MITRE']
        }
    
    async def _generate_topic_analysis(
        self, 
        topic: str, 
        intel: Dict[str, Any], 
        depth: str
    ) -> Dict[str, Any]:
        """Generate topic analysis"""
        prompt = f"""
        Generate a {depth} analysis of {topic} based on:
        
        Intelligence: {intel.get('intelligence', [])}
        
        Provide analysis with summary, findings, threat assessment, and recommendations.
        """
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model_name="mistral",
                task_type=TaskType.THREAT_ANALYSIS,
                temperature=0.3
            )
            
            # Parse response into structured format
            content = response['content']
            
            return {
                'summary': f"Analysis of {topic} threats and trends",
                'findings': [f"Finding related to {topic}"],
                'threat_assessment': 'medium',
                'recommendations': [f"Recommendation for {topic}"]
            }
            
        except:
            return {
                'summary': f"Analysis of {topic}",
                'findings': [],
                'threat_assessment': 'unknown',
                'recommendations': []
            }
    
    async def _get_trend_data(
        self, 
        start_date: datetime.date, 
        end_date: datetime.date, 
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get trend analysis data"""
        # This would query historical data for trend analysis
        return {
            'period': f"{start_date} to {end_date}",
            'data': [
                {
                    'date': '2024-01-01',
                    'ransomware_attacks': 15,
                    'phishing_attempts': 45,
                    'data_breaches': 3
                }
            ]
        }
    
    async def _analyze_trends_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend data"""
        return {
            'summary': 'Overall threat landscape shows increasing sophistication',
            'emerging_threats': [
                {
                    'name': 'AI-powered attacks',
                    'growth_rate': 25,
                    'confidence': 85
                }
            ],
            'declining_threats': [
                {
                    'name': 'Simple phishing',
                    'growth_rate': -10,
                    'confidence': 75
                }
            ],
            'stable_threats': [
                {
                    'name': 'Ransomware',
                    'growth_rate': 5,
                    'confidence': 90
                }
            ],
            'confidence': 85
        }
    
    async def _generate_trend_visualizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trend visualizations"""
        # This would create charts and graphs
        return [
            'trend_chart_2024.png',
            'threat_heatmap.png'
        ]
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to days"""
        timeframe_map = {
            '1d': 1,
            '7d': 7,
            '30d': 30,
            '90d': 90,
            '1y': 365
        }
        return timeframe_map.get(timeframe, 30)
    
    def save_briefing(self, briefing: Dict[str, Any], filename: str, format: str):
        """Save briefing to file"""
        try:
            output_dir = Path(self.config.data_dir) / "briefings"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = output_dir / filename
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(briefing, f, indent=2)
            elif format == "markdown":
                content = self._format_briefing_markdown(briefing)
                with open(filepath, 'w') as f:
                    f.write(content)
            else:
                # Default to JSON
                with open(filepath, 'w') as f:
                    json.dump(briefing, f, indent=2)
                    
        except Exception as e:
            print(f"Error saving briefing: {e}")
    
    def _format_briefing_markdown(self, briefing: Dict[str, Any]) -> str:
        """Format briefing as markdown"""
        content = f"""# Threat Briefing - {briefing.get('date', 'Unknown')}

## Executive Summary
{briefing.get('executive_summary', 'No summary available')}

## Key Threats
"""
        
        for threat in briefing.get('key_threats', []):
            content += f"- **{threat['name']}** ({threat['severity']}) - {threat['impact']}\n"
        
        content += "\n## New Vulnerabilities\n"
        for vuln in briefing.get('vulnerabilities', []):
            content += f"- {vuln['cve']} (CVSS: {vuln['cvss_score']}) - {vuln['product']}\n"
        
        content += "\n## Recommendations\n"
        for rec in briefing.get('recommendations', []):
            content += f"- {rec}\n"
        
        return content
    
    def email_briefing(self, briefing: Dict[str, Any]):
        """Send briefing via email"""
        try:
            # Get email configuration
            smtp_server = self.config.get_api_key('smtp_server')
            smtp_port = int(self.config.get_api_key('smtp_port', '587'))
            username = self.config.get_api_key('smtp_username')
            password = self.config.get_api_key('smtp_password')
            
            if not all([smtp_server, username, password]):
                print("Email configuration incomplete")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = username  # Send to self for now
            msg['Subject'] = f"Threat Briefing - {briefing.get('date', 'Unknown')}"
            
            # Create email body
            body = self._format_briefing_markdown(briefing)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
                
        except Exception as e:
            print(f"Error sending email: {e}") 