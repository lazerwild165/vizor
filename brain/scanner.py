#!/usr/bin/env python3
"""
Vizor Security Scanner
Scans and analyzes security artifacts
"""

import hashlib
import requests
import socket
import dns.resolver
import whois
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time
import re
from urllib.parse import urlparse

class SecurityScanner:
    """
    Security scanner for various artifacts
    
    Provides scanning capabilities for files, URLs, hashes,
    IP addresses, and domains with threat intelligence integration.
    """
    
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Vizor Security Scanner/1.0'
        })
    
    def scan_file(
        self, 
        file_path: Path, 
        scan_type: str = "auto",
        deep_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Scan a file for security threats
        
        Args:
            file_path: Path to file to scan
            scan_type: Type of scan (auto, malware, hash, metadata)
            deep_analysis: Whether to perform deep analysis
            
        Returns:
            Scan results
        """
        try:
            if not file_path.exists():
                return {
                    'error': 'File not found',
                    'threat_level': 'unknown',
                    'confidence': 0.0
                }
            
            # Get basic file information
            file_info = self._get_file_info(file_path)
            
            # Determine scan type
            if scan_type == "auto":
                scan_type = self._determine_scan_type(file_path)
            
            # Perform scan based on type
            if scan_type == "malware":
                results = self._scan_for_malware(file_path, deep_analysis)
            elif scan_type == "hash":
                results = self._scan_hash(file_path)
            elif scan_type == "metadata":
                results = self._scan_metadata(file_path)
            else:
                results = self._scan_general(file_path, deep_analysis)
            
            # Combine results
            results.update(file_info)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'threat_level': 'unknown',
                'confidence': 0.0
            }
    
    def scan_url(
        self, 
        url: str,
        check_reputation: bool = True,
        take_screenshot: bool = False,
        deep_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Scan a URL for security threats
        
        Args:
            url: URL to scan
            check_reputation: Whether to check reputation
            take_screenshot: Whether to take screenshot
            deep_analysis: Whether to perform deep analysis
            
        Returns:
            Scan results
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    'error': 'Invalid URL',
                    'threat_level': 'unknown',
                    'confidence': 0.0
                }
            
            results = {
                'url': url,
                'domain': parsed_url.netloc,
                'scheme': parsed_url.scheme,
                'threat_level': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
            
            # Check reputation if requested
            if check_reputation:
                reputation = self._check_url_reputation(url)
                results.update(reputation)
            
            # Take screenshot if requested
            if take_screenshot:
                screenshot_path = self._take_screenshot(url)
                results['screenshot'] = screenshot_path
            
            # Perform deep analysis if requested
            if deep_analysis:
                deep_results = self._deep_url_analysis(url)
                results.update(deep_results)
            
            # Determine threat level
            results['threat_level'] = self._determine_url_threat_level(results)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'url': url,
                'threat_level': 'unknown',
                'confidence': 0.0
            }
    
    def lookup_hash(
        self, 
        hash_value: str,
        hash_type: str = "auto",
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Lookup hash in threat intelligence
        
        Args:
            hash_value: Hash to lookup
            hash_type: Type of hash (auto, md5, sha1, sha256)
            sources: Threat intel sources to query
            
        Returns:
            Hash lookup results
        """
        try:
            # Determine hash type if auto
            if hash_type == "auto":
                hash_type = self._determine_hash_type(hash_value)
            
            # Validate hash
            if not self._validate_hash(hash_value, hash_type):
                return {
                    'error': 'Invalid hash',
                    'found': False,
                    'confidence': 0.0
                }
            
            results = {
                'hash': hash_value,
                'hash_type': hash_type,
                'found': False,
                'detection_count': 0,
                'detections': [],
                'confidence': 0.0
            }
            
            # Query threat intelligence sources
            if sources:
                for source in sources:
                    source_results = self._query_threat_intel(hash_value, hash_type, source)
                    if source_results.get('found'):
                        results['found'] = True
                        results['detection_count'] += source_results.get('detection_count', 0)
                        results['detections'].extend(source_results.get('detections', []))
            
            # Calculate confidence
            results['confidence'] = min(1.0, results['detection_count'] / 10.0)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'hash': hash_value,
                'found': False,
                'confidence': 0.0
            }
    
    def analyze_ip(
        self, 
        ip_address: str,
        include_geolocation: bool = True,
        check_reputation: bool = True,
        scan_ports: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze IP address for threats
        
        Args:
            ip_address: IP address to analyze
            include_geolocation: Whether to include geolocation
            check_reputation: Whether to check reputation
            scan_ports: Whether to scan ports
            
        Returns:
            IP analysis results
        """
        try:
            # Validate IP address
            if not self._validate_ip(ip_address):
                return {
                    'error': 'Invalid IP address',
                    'threat_level': 'unknown',
                    'confidence': 0.0
                }
            
            results = {
                'ip': ip_address,
                'threat_level': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
            
            # Get geolocation if requested
            if include_geolocation:
                geolocation = self._get_ip_geolocation(ip_address)
                results['geolocation'] = geolocation
            
            # Check reputation if requested
            if check_reputation:
                reputation = self._check_ip_reputation(ip_address)
                results.update(reputation)
            
            # Scan ports if requested
            if scan_ports:
                port_scan = self._scan_ip_ports(ip_address)
                results['port_scan'] = port_scan
            
            # Determine threat level
            results['threat_level'] = self._determine_ip_threat_level(results)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'ip': ip_address,
                'threat_level': 'unknown',
                'confidence': 0.0
            }
    
    def analyze_domain(
        self, 
        domain: str,
        dns_analysis: bool = True,
        whois_lookup: bool = True,
        enumerate_subdomains: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze domain for security threats
        
        Args:
            domain: Domain to analyze
            dns_analysis: Whether to perform DNS analysis
            whois_lookup: Whether to perform WHOIS lookup
            enumerate_subdomains: Whether to enumerate subdomains
            
        Returns:
            Domain analysis results
        """
        try:
            # Validate domain
            if not self._validate_domain(domain):
                return {
                    'error': 'Invalid domain',
                    'threat_level': 'unknown',
                    'confidence': 0.0
                }
            
            results = {
                'domain': domain,
                'threat_level': 'unknown',
                'confidence': 0.0,
                'indicators': []
            }
            
            # Perform DNS analysis if requested
            if dns_analysis:
                dns_results = self._analyze_dns(domain)
                results['dns'] = dns_results
            
            # Perform WHOIS lookup if requested
            if whois_lookup:
                whois_results = self._get_whois_info(domain)
                results['whois'] = whois_results
            
            # Enumerate subdomains if requested
            if enumerate_subdomains:
                subdomains = self._enumerate_subdomains(domain)
                results['subdomains'] = subdomains
            
            # Check reputation
            reputation = self._check_domain_reputation(domain)
            results.update(reputation)
            
            # Determine threat level
            results['threat_level'] = self._determine_domain_threat_level(results)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'domain': domain,
                'threat_level': 'unknown',
                'confidence': 0.0
            }
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file information"""
        stat = file_path.stat()
        
        return {
            'file_name': file_path.name,
            'file_size': stat.st_size,
            'file_type': self._get_file_type(file_path),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'file_path': str(file_path)
        }
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type"""
        extension = file_path.suffix.lower()
        
        type_map = {
            '.exe': 'executable',
            '.dll': 'library',
            '.py': 'python',
            '.js': 'javascript',
            '.php': 'php',
            '.html': 'html',
            '.pdf': 'pdf',
            '.doc': 'document',
            '.txt': 'text'
        }
        
        return type_map.get(extension, 'unknown')
    
    def _determine_scan_type(self, file_path: Path) -> str:
        """Determine appropriate scan type for file"""
        file_type = self._get_file_type(file_path)
        
        if file_type in ['executable', 'library']:
            return 'malware'
        elif file_type in ['python', 'javascript', 'php']:
            return 'malware'
        else:
            return 'metadata'
    
    def _scan_for_malware(self, file_path: Path, deep: bool) -> Dict[str, Any]:
        """Scan file for malware indicators"""
        # This would integrate with antivirus engines
        # For now, return basic analysis
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Calculate hashes
        md5_hash = hashlib.md5(content).hexdigest()
        sha1_hash = hashlib.sha1(content).hexdigest()
        sha256_hash = hashlib.sha256(content).hexdigest()
        
        # Basic malware indicators
        indicators = []
        if b'virus' in content.lower() or b'malware' in content.lower():
            indicators.append('Suspicious strings found')
        
        if len(content) < 1000:
            indicators.append('Suspiciously small file')
        
        return {
            'scan_type': 'malware',
            'md5': md5_hash,
            'sha1': sha1_hash,
            'sha256': sha256_hash,
            'indicators': indicators,
            'threat_level': 'low' if not indicators else 'medium',
            'confidence': 0.6
        }
    
    def _scan_hash(self, file_path: Path) -> Dict[str, Any]:
        """Scan file hash against threat intelligence"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        sha256_hash = hashlib.sha256(content).hexdigest()
        
        # Lookup hash
        return self.lookup_hash(sha256_hash, 'sha256')
    
    def _scan_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract and analyze file metadata"""
        # This would extract metadata based on file type
        # For now, return basic info
        
        return {
            'scan_type': 'metadata',
            'metadata_extracted': True,
            'threat_level': 'low',
            'confidence': 0.5
        }
    
    def _scan_general(self, file_path: Path, deep: bool) -> Dict[str, Any]:
        """Perform general file analysis"""
        return {
            'scan_type': 'general',
            'analysis_complete': True,
            'threat_level': 'low',
            'confidence': 0.5
        }
    
    def _check_url_reputation(self, url: str) -> Dict[str, Any]:
        """Check URL reputation"""
        # This would query reputation services
        # For now, return basic check
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Basic reputation check
        reputation_score = 50  # Neutral score
        
        if any(suspicious in domain for suspicious in ['malware', 'virus', 'spam']):
            reputation_score = 10
        elif any(trusted in domain for trusted in ['google', 'microsoft', 'github']):
            reputation_score = 90
        
        return {
            'reputation_score': reputation_score,
            'category': 'unknown',
            'status_code': 200
        }
    
    def _take_screenshot(self, url: str) -> Optional[str]:
        """Take screenshot of URL"""
        # This would use a headless browser
        # For now, return None
        return None
    
    def _deep_url_analysis(self, url: str) -> Dict[str, Any]:
        """Perform deep URL analysis"""
        return {
            'deep_analysis': True,
            'indicators': []
        }
    
    def _determine_url_threat_level(self, results: Dict[str, Any]) -> str:
        """Determine URL threat level"""
        reputation_score = results.get('reputation_score', 50)
        
        if reputation_score < 20:
            return 'high'
        elif reputation_score < 50:
            return 'medium'
        else:
            return 'low'
    
    def _determine_hash_type(self, hash_value: str) -> str:
        """Determine hash type from length"""
        length = len(hash_value)
        
        if length == 32:
            return 'md5'
        elif length == 40:
            return 'sha1'
        elif length == 64:
            return 'sha256'
        else:
            return 'unknown'
    
    def _validate_hash(self, hash_value: str, hash_type: str) -> bool:
        """Validate hash format"""
        if hash_type == 'md5':
            return len(hash_value) == 32 and hash_value.isalnum()
        elif hash_type == 'sha1':
            return len(hash_value) == 40 and hash_value.isalnum()
        elif hash_type == 'sha256':
            return len(hash_value) == 64 and hash_value.isalnum()
        else:
            return False
    
    def _query_threat_intel(self, hash_value: str, hash_type: str, source: str) -> Dict[str, Any]:
        """Query threat intelligence source"""
        # This would query actual threat intelligence APIs
        # For now, return mock data
        
        return {
            'found': False,
            'detection_count': 0,
            'detections': []
        }
    
    def _validate_ip(self, ip_address: str) -> bool:
        """Validate IP address format"""
        try:
            socket.inet_aton(ip_address)
            return True
        except socket.error:
            return False
    
    def _get_ip_geolocation(self, ip_address: str) -> Dict[str, Any]:
        """Get IP geolocation information"""
        # This would query geolocation services
        # For now, return mock data
        
        return {
            'country': 'Unknown',
            'city': 'Unknown',
            'isp': 'Unknown',
            'latitude': 0.0,
            'longitude': 0.0
        }
    
    def _check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP reputation"""
        # This would query reputation services
        # For now, return mock data
        
        return {
            'reputation_score': 50,
            'category': 'unknown'
        }
    
    def _scan_ip_ports(self, ip_address: str) -> Dict[str, Any]:
        """Scan common ports on IP"""
        # This would perform actual port scanning
        # For now, return mock data
        
        return {
            'ports_scanned': [80, 443, 22, 21],
            'open_ports': [80, 443],
            'services': {
                80: 'http',
                443: 'https'
            }
        }
    
    def _determine_ip_threat_level(self, results: Dict[str, Any]) -> str:
        """Determine IP threat level"""
        reputation_score = results.get('reputation_score', 50)
        
        if reputation_score < 20:
            return 'high'
        elif reputation_score < 50:
            return 'medium'
        else:
            return 'low'
    
    def _validate_domain(self, domain: str) -> bool:
        """Validate domain format"""
        # Basic domain validation
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return bool(re.match(pattern, domain))
    
    def _analyze_dns(self, domain: str) -> Dict[str, Any]:
        """Analyze DNS records"""
        try:
            results = {
                'a_records': [],
                'aaaa_records': [],
                'mx_records': [],
                'ns_records': [],
                'txt_records': []
            }
            
            # Query A records
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                results['a_records'] = [str(record) for record in a_records]
            except:
                pass
            
            # Query MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                results['mx_records'] = [str(record.exchange) for record in mx_records]
            except:
                pass
            
            return results
            
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def _get_whois_info(self, domain: str) -> Dict[str, Any]:
        """Get WHOIS information"""
        try:
            w = whois.whois(domain)
            return {
                'registrar': w.registrar,
                'creation_date': str(w.creation_date),
                'expiration_date': str(w.expiration_date),
                'name_servers': w.name_servers
            }
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def _enumerate_subdomains(self, domain: str) -> List[str]:
        """Enumerate subdomains"""
        # This would perform subdomain enumeration
        # For now, return common subdomains
        
        common_subdomains = ['www', 'mail', 'ftp', 'admin', 'blog']
        subdomains = []
        
        for subdomain in common_subdomains:
            try:
                full_domain = f"{subdomain}.{domain}"
                socket.gethostbyname(full_domain)
                subdomains.append(full_domain)
            except:
                pass
        
        return subdomains
    
    def _check_domain_reputation(self, domain: str) -> Dict[str, Any]:
        """Check domain reputation"""
        # This would query reputation services
        # For now, return mock data
        
        return {
            'reputation_score': 50,
            'category': 'unknown'
        }
    
    def _determine_domain_threat_level(self, results: Dict[str, Any]) -> str:
        """Determine domain threat level"""
        reputation_score = results.get('reputation_score', 50)
        
        if reputation_score < 20:
            return 'high'
        elif reputation_score < 50:
            return 'medium'
        else:
            return 'low'
    
    def save_results(self, results: Dict[str, Any], output_path: str, format: str):
        """Save scan results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif format == "csv":
                # Convert results to CSV format
                import csv
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write CSV headers and data
                    pass
            else:
                # Default to JSON
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        except Exception as e:
            print(f"Error saving results: {e}") 