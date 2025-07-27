#!/usr/bin/env python3
"""
Vizor CLI Main Application
Entry point for all Vizor commands
"""

import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import from other modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.commands import ask, brief, scan, build
from config.settings import VizorConfig
from brain.logger import VizorLogger

# Initialize console and app
console = Console()
app = typer.Typer(
    name="vizor",
    help="🔍 Vizor v1.0 - Local-first Cybersecurity Copilot",
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Initialize configuration and logger
config = VizorConfig()
logger = VizorLogger()

def version_callback(value: bool):
    """Show version information"""
    if value:
        console.print(Panel.fit(
            "[bold blue]Vizor v1.0[/bold blue]\n"
            "[dim]Local-first Cybersecurity Copilot[/dim]\n"
            "[dim]Built for autonomy, privacy, and adaptation[/dim]",
            title="🔍 Vizor",
            border_style="blue"
        ))
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, 
        help="Show version information"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", 
        help="Path to custom config file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", 
        help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", 
        help="Simulate actions without execution"
    )
):
    """
    🔍 Vizor - Your local-first cybersecurity copilot
    
    Vizor acts as both a soldier (executor) and advisor (thinker),
    adapting to your style while maintaining privacy and autonomy.
    """
    # Set global options
    if config_path:
        config.load_custom_config(config_path)
    
    if verbose:
        logger.set_verbose(True)
    
    if dry_run:
        config.set_dry_run(True)
        console.print("[yellow]🔸 Dry run mode enabled - no actions will be executed[/yellow]")

# Add command groups
app.add_typer(ask.app, name="ask", help="💬 Ask Vizor questions and get intelligent responses")
app.add_typer(brief.app, name="brief", help="📋 Generate threat briefings and summaries")
app.add_typer(scan.app, name="scan", help="🔍 Scan and analyze security artifacts")
app.add_typer(build.app, name="build", help="🔧 Build API wrappers and plugins")

@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="Overwrite existing configuration")
):
    """
    🚀 Initialize Vizor configuration
    
    Creates default configuration files and sets up the local environment.
    """
    try:
        config.initialize(force=force)
        console.print("[green]✅ Vizor initialized successfully![/green]")
        console.print(f"[dim]Config file: {config.config_path}[/dim]")
        console.print("[dim]Run 'vizor ask --help' to get started[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def status():
    """
    📊 Show Vizor system status
    
    Displays configuration, model status, and system health.
    """
    try:
        status_info = config.get_status()
        
        # Create status panel
        status_text = f"""
[bold]Configuration:[/bold] {status_info['config_status']}
[bold]Local LLM:[/bold] {status_info['llm_status']}
[bold]Vector Store:[/bold] {status_info['vector_status']}
[bold]Plugins:[/bold] {status_info['plugin_count']} loaded
[bold]Memory:[/bold] {status_info['memory_status']}
        """.strip()
        
        console.print(Panel(
            status_text,
            title="🔍 Vizor Status",
            border_style="green" if status_info['healthy'] else "red"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def learn(
    topic: str = typer.Argument(..., help="Topic to learn about"),
    sources: Optional[str] = typer.Option(None, "--sources", help="Comma-separated list of sources"),
    update_memory: bool = typer.Option(True, "--update-memory/--no-update-memory", help="Update vector memory")
):
    """
    🧠 Trigger learning flow for a specific topic
    
    Fetches intelligence and updates knowledge base.
    """
    from brain.learning import LearningEngine
    
    try:
        learning_engine = LearningEngine(config)
        
        console.print(f"[blue]🧠 Learning about: {topic}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would fetch intelligence and update memory[/yellow]")
            return
        
        result = learning_engine.learn_topic(
            topic=topic,
            sources=sources.split(',') if sources else None,
            update_memory=update_memory
        )
        
        console.print(f"[green]✅ Learning completed![/green]")
        console.print(f"[dim]Sources processed: {result['sources_count']}[/dim]")
        console.print(f"[dim]Knowledge items added: {result['items_added']}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Learning failed: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
