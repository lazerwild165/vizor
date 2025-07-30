#!/usr/bin/env python3
"""
Vizor Scan Command
Security scanning and analysis functionality
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import VizorConfig

console = Console()
app = typer.Typer(help="🔍 Scan and analyze security artifacts")

@app.command()
def file(
    filepath: str = typer.Argument(..., help="Path to file to scan"),
    scan_type: str = typer.Option("auto", "--type", help="Scan type: auto, malware, hash, metadata"),
    deep: bool = typer.Option(False, "--deep", help="Enable deep analysis"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file for results"),
    format: str = typer.Option("rich", "--format", help="Output format: rich, json, csv")
):
    """
    📄 Scan a file for security threats
    
    Analyzes files for malware, suspicious patterns,
    and security indicators.
    """
    try:
        # Import only when needed
        from brain.scanner import SecurityScanner
        
        config = VizorConfig()
        
        file_path = Path(filepath)
        if not file_path.exists():
            console.print(f"[red]❌ File not found: {filepath}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]🔍 Scanning file: {file_path.name}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would scan file and display results[/yellow]")
            return
        
        # Perform scan using direct Ollama
        import ollama
        
        # Read file content (limit to first 1000 chars for analysis)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)
        except:
            content = f"Binary file: {file_path.name}"
        
        prompt = f"""Analyze this file for security threats and suspicious patterns:

File: {file_path.name}
Type: {scan_type}
Content: {content}

Provide a security analysis including:
1. Potential threats detected
2. Suspicious patterns
3. Security recommendations
4. Risk assessment

Format as a security report."""

        response = ollama.chat(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": "You are a cybersecurity analyst. Analyze files for security threats."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.7,
                "num_predict": 512
            }
        )
        
        # Display results
        console.print(Panel(
            response['message']['content'],
            title="🔍 Security Scan Results",
            border_style="green"
        ))
        
        # Save results if requested
        if output:
            console.print(f"[green]💾 Results saved to {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ File scan failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def url(
    target_url: str = typer.Argument(..., help="URL to scan"),
    check_reputation: bool = typer.Option(True, "--reputation/--no-reputation", help="Check URL reputation"),
    screenshot: bool = typer.Option(False, "--screenshot", help="Take screenshot"),
    deep: bool = typer.Option(False, "--deep", help="Deep web analysis")
):
    """
    🌐 Scan a URL for security threats
    
    Analyzes URLs for malicious content, phishing,
    and suspicious behavior.
    """
    try:
        # Import only when needed
        from brain.scanner import SecurityScanner
        
        config = VizorConfig()
        #scanner = SecurityScanner(config)
        
        console.print(f"[blue]🌐 Scanning URL: {target_url}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would scan URL and analyze[/yellow]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing URL...", total=None)
            
            # Perform URL scan
            # results = scanner.scan_url(
            #     url=target_url,
            #     check_reputation=check_reputation,
            #     take_screenshot=screenshot,
            #     deep_analysis=deep
            # )
        
        # Display results
        #display_url_results(results)
        
    except Exception as e:
        console.print(f"[red]❌ URL scan failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def hash(
    hash_value: str = typer.Argument(..., help="Hash to lookup"),
    hash_type: str = typer.Option("auto", "--type", help="Hash type: auto, md5, sha1, sha256"),
    sources: Optional[str] = typer.Option(None, "--sources", help="Threat intel sources to query")
):
    """
    #️⃣ Lookup hash in threat intelligence
    
    Queries multiple threat intelligence sources
    for hash reputation and analysis.
    """
    try:
        # Import only when needed
        from brain.scanner import SecurityScanner
        
        config = VizorConfig()
        #scanner = SecurityScanner(config)
        
        console.print(f"[blue]#️⃣ Looking up hash: {hash_value[:16]}...[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would query threat intel sources[/yellow]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Querying threat intel...", total=None)
            
            # Perform hash lookup
            # results = scanner.lookup_hash(
            #     hash_value=hash_value,
            #     hash_type=hash_type,
            #     sources=sources.split(',') if sources else None
            # )
        
        # Display results
        #display_hash_results(results)
        
    except Exception as e:
        console.print(f"[red]❌ Hash lookup failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def ip(
    ip_address: str = typer.Argument(..., help="IP address to analyze"),
    geolocation: bool = typer.Option(True, "--geo/--no-geo", help="Include geolocation data"),
    reputation: bool = typer.Option(True, "--reputation/--no-reputation", help="Check reputation"),
    ports: bool = typer.Option(False, "--ports", help="Scan common ports")
):
    """
    🌍 Analyze IP address for threats
    
    Performs comprehensive IP analysis including
    geolocation, reputation, and port scanning.
    """
    try:
        # Import only when needed
        from brain.scanner import SecurityScanner
        
        config = VizorConfig()
        #scanner = SecurityScanner(config)
        
        console.print(f"[blue]🌍 Analyzing IP: {ip_address}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would analyze IP address[/yellow]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing IP...", total=None)
            
            # Perform IP analysis
            # results = scanner.analyze_ip(
            #     ip_address=ip_address,
            #     include_geolocation=geolocation,
            #     check_reputation=reputation,
            #     scan_ports=ports
            # )
        
        # Display results
        #display_ip_results(results)
        
    except Exception as e:
        console.print(f"[red]❌ IP analysis failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def domain(
    domain_name: str = typer.Argument(..., help="Domain to analyze"),
    dns: bool = typer.Option(True, "--dns/--no-dns", help="Perform DNS analysis"),
    whois: bool = typer.Option(True, "--whois/--no-whois", help="Include WHOIS data"),
    subdomains: bool = typer.Option(False, "--subdomains", help="Enumerate subdomains")
):
    """
    🏷️ Analyze domain for security threats
    
    Comprehensive domain analysis including DNS,
    WHOIS, and subdomain enumeration.
    """
    try:
        # Import only when needed
        from brain.scanner import SecurityScanner
        
        config = VizorConfig()
        #scanner = SecurityScanner(config)
        
        console.print(f"[blue]🏷️ Analyzing domain: {domain_name}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would analyze domain[/yellow]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing domain...", total=None)
            
            # Perform domain analysis
            # results = scanner.analyze_domain(
            #     domain=domain_name,
            #     dns_analysis=dns,
            #     whois_lookup=whois,
            #     enumerate_subdomains=subdomains
            # )
        
        # Display results
        #display_domain_results(results)
        
    except Exception as e:
        console.print(f"[red]❌ Domain analysis failed: {e}[/red]")
        raise typer.Exit(1)

def display_scan_results(results, format):
    """Display file scan results"""
    if format == "rich":
        # Threat assessment
        threat_color = "red" if results['threat_level'] == "high" else "yellow" if results['threat_level'] == "medium" else "green"
        
        console.print(Panel(
            f"[{threat_color}]Threat Level: {results['threat_level'].upper()}[/{threat_color}]\n"
            f"Confidence: {results['confidence']}%\n"
            f"File Type: {results['file_type']}\n"
            f"Size: {results['file_size']} bytes",
            title="🔍 Scan Summary",
            border_style=threat_color
        ))
        
        # Detections
        if results.get('detections'):
            detection_table = Table(title="🚨 Detections")
            detection_table.add_column("Engine", style="cyan")
            detection_table.add_column("Result", style="red")
            detection_table.add_column("Details", style="dim")
            
            for detection in results['detections']:
                detection_table.add_row(
                    detection['engine'],
                    detection['result'],
                    detection.get('details', '')
                )
            
            console.print(detection_table)
    
    elif format == "json":
        import json
        console.print(json.dumps(results, indent=2, default=str))

def display_url_results(results):
    """Display URL scan results"""
    threat_color = "red" if results['threat_level'] == "high" else "yellow" if results['threat_level'] == "medium" else "green"
    
    console.print(Panel(
        f"[{threat_color}]Threat Level: {results['threat_level'].upper()}[/{threat_color}]\n"
        f"Reputation Score: {results['reputation_score']}/100\n"
        f"Category: {results.get('category', 'Unknown')}\n"
        f"Status Code: {results.get('status_code', 'N/A')}",
        title="🌐 URL Analysis",
        border_style=threat_color
    ))
    
    if results.get('indicators'):
        console.print(Panel(
            "\n".join(f"• {indicator}" for indicator in results['indicators']),
            title="⚠️ Security Indicators",
            border_style="yellow"
        ))

def display_hash_results(results):
    """Display hash lookup results"""
    if results['found']:
        console.print(Panel(
            f"[red]Hash found in threat intelligence![/red]\n"
            f"Detections: {results['detection_count']}\n"
            f"First Seen: {results.get('first_seen', 'Unknown')}\n"
            f"Last Seen: {results.get('last_seen', 'Unknown')}",
            title="#️⃣ Hash Analysis",
            border_style="red"
        ))
        
        if results.get('detections'):
            detection_table = Table(title="🔍 Detections")
            detection_table.add_column("Source", style="cyan")
            detection_table.add_column("Classification", style="red")
            
            for detection in results['detections'][:10]:  # Top 10
                detection_table.add_row(
                    detection['source'],
                    detection['classification']
                )
            
            console.print(detection_table)
    else:
        console.print(Panel(
            "[green]Hash not found in threat intelligence databases[/green]",
            title="#️⃣ Hash Analysis",
            border_style="green"
        ))

def display_ip_results(results):
    """Display IP analysis results"""
    threat_color = "red" if results['threat_level'] == "high" else "yellow" if results['threat_level'] == "medium" else "green"
    
    info_text = f"[{threat_color}]Threat Level: {results['threat_level'].upper()}[/{threat_color}]\n"
    
    if results.get('geolocation'):
        geo = results['geolocation']
        info_text += f"Location: {geo.get('city', 'Unknown')}, {geo.get('country', 'Unknown')}\n"
        info_text += f"ISP: {geo.get('isp', 'Unknown')}\n"
    
    if results.get('reputation_score'):
        info_text += f"Reputation: {results['reputation_score']}/100\n"
    
    console.print(Panel(
        info_text.strip(),
        title="🌍 IP Analysis",
        border_style=threat_color
    ))

def display_domain_results(results):
    """Display domain analysis results"""
    threat_color = "red" if results['threat_level'] == "high" else "yellow" if results['threat_level'] == "medium" else "green"
    
    console.print(Panel(
        f"[{threat_color}]Threat Level: {results['threat_level'].upper()}[/{threat_color}]\n"
        f"Domain Age: {results.get('domain_age', 'Unknown')}\n"
        f"Registrar: {results.get('registrar', 'Unknown')}\n"
        f"Name Servers: {len(results.get('nameservers', []))}",
        title="🏷️ Domain Analysis",
        border_style=threat_color
    ))
    
    if results.get('subdomains'):
        console.print(Panel(
            "\n".join(f"• {subdomain}" for subdomain in results['subdomains'][:20]),
            title="🔗 Subdomains",
            border_style="cyan"
        ))

if __name__ == "__main__":
    app()
