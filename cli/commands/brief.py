#!/usr/bin/env python3
"""
Vizor Brief Command
Threat briefing and summary generation functionality
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.threat_briefing import ThreatBriefingEngine
from config.settings import VizorConfig

console = Console()
app = typer.Typer(help="📋 Generate threat briefings and summaries")

@app.command()
def daily(
    date: Optional[str] = typer.Option(None, "--date", help="Date for briefing (YYYY-MM-DD)"),
    sources: Optional[str] = typer.Option(None, "--sources", help="Comma-separated threat intel sources"),
    format: str = typer.Option("rich", "--format", help="Output format: rich, markdown, json"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save briefing to file"),
    email: bool = typer.Option(False, "--email", help="Send briefing via email")
):
    """
    📅 Generate daily threat briefing
    
    Creates a comprehensive daily briefing with latest threats,
    vulnerabilities, and security trends.
    """
    try:
        config = VizorConfig()
        briefing_engine = ThreatBriefingEngine(config)
        
        # Parse date or use today
        if date:
            briefing_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            briefing_date = datetime.now().date()
        
        console.print(f"[blue]📋 Generating daily briefing for {briefing_date}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would generate and display briefing[/yellow]")
            return
        
        # Generate briefing
        briefing = briefing_engine.generate_daily_briefing(
            date=briefing_date,
            sources=sources.split(',') if sources else None
        )
        
        # Display briefing based on format
        if format == "rich":
            display_rich_briefing(briefing)
        elif format == "markdown":
            display_markdown_briefing(briefing)
        elif format == "json":
            display_json_briefing(briefing)
        
        # Save if requested
        if save:
            filename = f"briefing_{briefing_date.strftime('%Y%m%d')}.{format}"
            briefing_engine.save_briefing(briefing, filename, format)
            console.print(f"[green]💾 Briefing saved to {filename}[/green]")
        
        # Email if requested
        if email:
            briefing_engine.email_briefing(briefing)
            console.print("[green]📧 Briefing sent via email[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Briefing generation failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def weekly(
    week_start: Optional[str] = typer.Option(None, "--start", help="Week start date (YYYY-MM-DD)"),
    include_trends: bool = typer.Option(True, "--trends/--no-trends", help="Include trend analysis"),
    format: str = typer.Option("rich", "--format", help="Output format: rich, markdown, json")
):
    """
    📊 Generate weekly threat summary
    
    Creates a comprehensive weekly summary with trends,
    patterns, and strategic insights.
    """
    try:
        config = VizorConfig()
        briefing_engine = ThreatBriefingEngine(config)
        
        # Calculate week dates
        if week_start:
            start_date = datetime.strptime(week_start, "%Y-%m-%d").date()
        else:
            today = datetime.now().date()
            start_date = today - timedelta(days=today.weekday())
        
        end_date = start_date + timedelta(days=6)
        
        console.print(f"[blue]📊 Generating weekly briefing for {start_date} to {end_date}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would generate weekly summary[/yellow]")
            return
        
        # Generate weekly briefing
        briefing = briefing_engine.generate_weekly_briefing(
            start_date=start_date,
            end_date=end_date,
            include_trends=include_trends
        )
        
        # Display briefing
        if format == "rich":
            display_rich_briefing(briefing)
        elif format == "markdown":
            display_markdown_briefing(briefing)
        elif format == "json":
            display_json_briefing(briefing)
            
    except Exception as e:
        console.print(f"[red]❌ Weekly briefing failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def custom(
    topic: str = typer.Argument(..., help="Topic or threat to brief on"),
    timeframe: str = typer.Option("7d", "--timeframe", help="Timeframe: 1d, 7d, 30d, 90d"),
    depth: str = typer.Option("medium", "--depth", help="Analysis depth: shallow, medium, deep"),
    sources: Optional[str] = typer.Option(None, "--sources", help="Specific sources to use")
):
    """
    🎯 Generate custom threat briefing
    
    Creates a focused briefing on a specific topic, threat actor,
    or security domain.
    """
    try:
        config = VizorConfig()
        briefing_engine = ThreatBriefingEngine(config)
        
        console.print(f"[blue]🎯 Generating custom briefing on: {topic}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would generate custom briefing[/yellow]")
            return
        
        # Generate custom briefing
        briefing = briefing_engine.generate_custom_briefing(
            topic=topic,
            timeframe=timeframe,
            depth=depth,
            sources=sources.split(',') if sources else None
        )
        
        # Display briefing
        display_rich_briefing(briefing)
        
    except Exception as e:
        console.print(f"[red]❌ Custom briefing failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def trends(
    period: str = typer.Option("30d", "--period", help="Analysis period: 7d, 30d, 90d, 1y"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Threat categories to analyze"),
    visualize: bool = typer.Option(False, "--visualize", help="Generate trend visualizations")
):
    """
    📈 Analyze threat trends and patterns
    
    Identifies emerging threats, attack patterns, and
    security trends over time.
    """
    try:
        config = VizorConfig()
        briefing_engine = ThreatBriefingEngine(config)
        
        console.print(f"[blue]📈 Analyzing threat trends for {period}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would analyze trends[/yellow]")
            return
        
        # Generate trend analysis
        trends = briefing_engine.analyze_trends(
            period=period,
            categories=categories.split(',') if categories else None,
            visualize=visualize
        )
        
        # Display trends
        display_trends(trends)
        
    except Exception as e:
        console.print(f"[red]❌ Trend analysis failed: {e}[/red]")
        raise typer.Exit(1)

def display_rich_briefing(briefing):
    """Display briefing in rich format"""
    # Executive Summary
    console.print(Panel(
        briefing['executive_summary'],
        title="🎯 Executive Summary",
        border_style="blue"
    ))
    
    # Key Threats
    if briefing.get('key_threats'):
        threats_table = Table(title="🚨 Key Threats")
        threats_table.add_column("Threat", style="red")
        threats_table.add_column("Severity", style="yellow")
        threats_table.add_column("Impact", style="cyan")
        
        for threat in briefing['key_threats']:
            threats_table.add_row(
                threat['name'],
                threat['severity'],
                threat['impact']
            )
        
        console.print(threats_table)
    
    # Vulnerabilities
    if briefing.get('vulnerabilities'):
        vuln_table = Table(title="🔓 New Vulnerabilities")
        vuln_table.add_column("CVE", style="red")
        vuln_table.add_column("Score", style="yellow")
        vuln_table.add_column("Product", style="cyan")
        
        for vuln in briefing['vulnerabilities'][:10]:  # Top 10
            vuln_table.add_row(
                vuln['cve'],
                str(vuln['cvss_score']),
                vuln['product']
            )
        
        console.print(vuln_table)
    
    # Recommendations
    if briefing.get('recommendations'):
        console.print(Panel(
            "\n".join(f"• {rec}" for rec in briefing['recommendations']),
            title="💡 Recommendations",
            border_style="green"
        ))

def display_markdown_briefing(briefing):
    """Display briefing in markdown format"""
    markdown_content = f"""
# Threat Briefing - {briefing['date']}

## Executive Summary
{briefing['executive_summary']}

## Key Threats
{chr(10).join(f"- **{t['name']}** ({t['severity']}) - {t['impact']}" for t in briefing.get('key_threats', []))}

## New Vulnerabilities
{chr(10).join(f"- {v['cve']} (CVSS: {v['cvss_score']}) - {v['product']}" for v in briefing.get('vulnerabilities', [])[:10])}

## Recommendations
{chr(10).join(f"- {rec}" for rec in briefing.get('recommendations', []))}
    """
    
    console.print(Markdown(markdown_content))

def display_json_briefing(briefing):
    """Display briefing in JSON format"""
    import json
    console.print(json.dumps(briefing, indent=2, default=str))

def display_trends(trends):
    """Display trend analysis"""
    console.print(Panel(
        trends['summary'],
        title="📈 Trend Analysis Summary",
        border_style="cyan"
    ))
    
    # Emerging threats
    if trends.get('emerging_threats'):
        emerging_table = Table(title="🆕 Emerging Threats")
        emerging_table.add_column("Threat", style="red")
        emerging_table.add_column("Growth", style="yellow")
        emerging_table.add_column("Confidence", style="green")
        
        for threat in trends['emerging_threats']:
            emerging_table.add_row(
                threat['name'],
                f"+{threat['growth_rate']}%",
                f"{threat['confidence']}%"
            )
        
        console.print(emerging_table)

if __name__ == "__main__":
    app()
