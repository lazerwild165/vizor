#!/usr/bin/env python3
"""
Vizor Build Command
API wrapper and plugin generation functionality
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import sys
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import VizorConfig

console = Console()
app = typer.Typer(help="🔧 Build API wrappers and plugins")

@app.command()
def api(
    url: str = typer.Argument(..., help="API URL or OpenAPI spec URL"),
    name: Optional[str] = typer.Option(None, "--name", help="Custom name for the wrapper"),
    output_dir: Optional[str] = typer.Option(None, "--output", help="Output directory"),
    auth_type: str = typer.Option("none", "--auth", help="Auth type: none, api_key, bearer, basic"),
    test: bool = typer.Option(True, "--test/--no-test", help="Generate test cases")
):
    """
    🌐 Generate API wrapper from OpenAPI spec or URL
    
    Automatically creates a Python wrapper for external APIs
    with proper authentication and error handling.
    """
    try:
        # Import only when needed
        from plugins.wrapper_generator import WrapperGenerator
        from plugins.plugin_registry import PluginRegistry
        
        config = VizorConfig()
        generator = WrapperGenerator(config)
        
        console.print(f"[blue]🔧 Building API wrapper for: {url}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would generate API wrapper[/yellow]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating wrapper...", total=None)
            
            # Generate the wrapper
            result = generator.generate_from_url(
                url=url,
                wrapper_name=name,
                output_dir=output_dir,
                auth_type=auth_type,
                include_tests=test
            )
        
        console.print(Panel(
            f"[green]✅ API wrapper generated successfully![/green]\n"
            f"[dim]Location: {result['wrapper_path']}[/dim]\n"
            f"[dim]Endpoints: {result['endpoint_count']}[/dim]\n"
            f"[dim]Methods: {result['method_count']}[/dim]",
            title="🔧 Build Complete",
            border_style="green"
        ))
        
        # Register the plugin
        registry = PluginRegistry(config)
        registry.register_plugin(Path(result['wrapper_path']))
        
        console.print(f"[green]📦 Plugin registered in Vizor[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ API wrapper generation failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def plugin(
    template: str = typer.Argument(..., help="Plugin template: basic, threat_intel, scanner, enrichment"),
    name: str = typer.Option(..., "--name", help="Plugin name"),
    description: Optional[str] = typer.Option(None, "--description", help="Plugin description"),
    author: Optional[str] = typer.Option(None, "--author", help="Plugin author")
):
    """
    🧩 Generate plugin template
    
    Creates a new plugin template with boilerplate code
    and proper structure for Vizor integration.
    """
    try:
        # Import only when needed
        from plugins.wrapper_generator import WrapperGenerator
        from plugins.plugin_registry import PluginRegistry
        
        config = VizorConfig()
        generator = WrapperGenerator(config)
        
        console.print(f"[blue]🧩 Creating {template} plugin: {name}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would create plugin template[/yellow]")
            return
        
        # Generate plugin template
        result = generator.generate_plugin_template(
            template_type=template,
            plugin_name=name,
            description=description,
            author=author
        )
        
        console.print(Panel(
            f"[green]✅ Plugin template created![/green]\n"
            f"[dim]Location: {result['plugin_path']}[/dim]\n"
            f"[dim]Template: {template}[/dim]\n"
            f"[dim]Files created: {len(result['files'])}[/dim]",
            title="🧩 Plugin Created",
            border_style="green"
        ))
        
        console.print("[dim]Next steps:[/dim]")
        console.print("[dim]1. Edit the plugin code in the generated files[/dim]")
        console.print("[dim]2. Test your plugin with 'vizor build test-plugin'[/dim]")
        console.print("[dim]3. Register with 'vizor build register-plugin'[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Plugin generation failed: {e}[/red]")
        raise typer.Exit(1)

@app.command("test-plugin")
def test_plugin(
    plugin_path: str = typer.Argument(..., help="Path to plugin directory or file"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose test output")
):
    """
    🧪 Test a plugin
    
    Runs tests and validation on a plugin to ensure
    it meets Vizor standards and works correctly.
    """
    try:
        # Import only when needed
        from plugins.plugin_registry import PluginRegistry
        
        config = VizorConfig()
        registry = PluginRegistry(config)
        
        plugin_path_obj = Path(plugin_path)
        if not plugin_path_obj.exists():
            console.print(f"[red]❌ Plugin path not found: {plugin_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]🧪 Testing plugin: {plugin_path_obj.name}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would test plugin[/yellow]")
            return
        
        # Test the plugin
        test_results = registry.test_plugin(plugin_path_obj, verbose=verbose)
        
        # Display results
        if test_results['success']:
            console.print(Panel(
                f"[green]✅ All tests passed![/green]\n"
                f"[dim]Tests run: {test_results['tests_run']}[/dim]\n"
                f"[dim]Warnings: {len(test_results.get('warnings', []))}[/dim]",
                title="🧪 Test Results",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[red]❌ Tests failed![/red]\n"
                f"[dim]Tests run: {test_results['tests_run']}[/dim]\n"
                f"[dim]Errors: {len(test_results.get('errors', []))}[/dim]",
                title="🧪 Test Results",
                border_style="red"
            ))
            
            if test_results.get('errors'):
                console.print("\n[red]Error Details:[/red]")
                for error in test_results['errors']:
                    console.print(f"[dim]• {error}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Plugin testing failed: {e}[/red]")
        raise typer.Exit(1)

@app.command("register-plugin")
def register_plugin(
    plugin_path: str = typer.Argument(..., help="Path to plugin directory or file"),
    force: bool = typer.Option(False, "--force", help="Force registration even if tests fail")
):
    """
    📦 Register a plugin with Vizor
    
    Adds a plugin to the Vizor registry, making it
    available for use in commands and workflows.
    """
    try:
        # Import only when needed
        from plugins.plugin_registry import PluginRegistry
        
        config = VizorConfig()
        registry = PluginRegistry(config)
        
        plugin_path_obj = Path(plugin_path)
        if not plugin_path_obj.exists():
            console.print(f"[red]❌ Plugin path not found: {plugin_path}[/red]")
            raise typer.Exit(1)
        
        console.print(f"[blue]📦 Registering plugin: {plugin_path_obj.name}[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would register plugin[/yellow]")
            return
        
        # Test first unless forced
        if not force:
            test_results = registry.test_plugin(plugin_path_obj, verbose=False)
            if not test_results['success']:
                console.print("[red]❌ Plugin tests failed. Use --force to register anyway.[/red]")
                raise typer.Exit(1)
        
        # Register the plugin
        registration_result = registry.register_plugin(plugin_path_obj)
        
        console.print(Panel(
            f"[green]✅ Plugin registered successfully![/green]\n"
            f"[dim]Name: {registration_result['name']}[/dim]\n"
            f"[dim]Version: {registration_result['version']}[/dim]\n"
            f"[dim]Methods: {len(registration_result['methods'])}[/dim]",
            title="📦 Registration Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Plugin registration failed: {e}[/red]")
        raise typer.Exit(1)

@app.command("list-plugins")
def list_plugins(
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed plugin information")
):
    """
    📋 List registered plugins
    
    Shows all plugins currently registered with Vizor
    and their status.
    """
    try:
        # Import only when needed
        from plugins.plugin_registry import PluginRegistry
        
        config = VizorConfig()
        registry = PluginRegistry(config)
        
        plugins = registry.list_plugins()
        
        if not plugins:
            console.print("[yellow]No plugins registered[/yellow]")
            return
        
        console.print(f"[blue]📋 Registered Plugins ({len(plugins)})[/blue]\n")
        
        for plugin in plugins:
            status_color = "green" if plugin['status'] == 'active' else "yellow"
            
            if detailed:
                console.print(Panel(
                    f"[bold]{plugin['name']}[/bold] v{plugin['version']}\n"
                    f"[dim]{plugin['description']}[/dim]\n"
                    f"Status: [{status_color}]{plugin['status']}[/{status_color}]\n"
                    f"Methods: {len(plugin['methods'])}\n"
                    f"Path: {plugin['path']}",
                    border_style=status_color
                ))
            else:
                console.print(f"[{status_color}]●[/{status_color}] {plugin['name']} v{plugin['version']} - {plugin['description']}")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list plugins: {e}[/red]")
        raise typer.Exit(1)

@app.command("update-plugins")
def update_plugins(
    plugin_name: Optional[str] = typer.Option(None, "--plugin", help="Update specific plugin"),
    check_only: bool = typer.Option(False, "--check-only", help="Only check for updates")
):
    """
    🔄 Update plugins
    
    Checks for and applies updates to registered plugins.
    Part of Vizor's self-growth capabilities.
    """
    try:
        # Import only when needed
        from plugins.plugin_registry import PluginRegistry
        
        config = VizorConfig()
        registry = PluginRegistry(config)
        
        if plugin_name:
            console.print(f"[blue]🔄 Checking updates for: {plugin_name}[/blue]")
        else:
            console.print("[blue]🔄 Checking all plugins for updates[/blue]")
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would check and apply updates[/yellow]")
            return
        
        # Check for updates
        update_results = registry.check_updates(plugin_name=plugin_name)
        
        if not update_results['updates_available']:
            console.print("[green]✅ All plugins are up to date[/green]")
            return
        
        console.print(f"[yellow]📦 {len(update_results['updates'])} updates available[/yellow]")
        
        if check_only:
            for update in update_results['updates']:
                console.print(f"[dim]• {update['name']}: {update['current_version']} → {update['new_version']}[/dim]")
            return
        
        # Apply updates
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Updating plugins...", total=None)
            
            apply_results = registry.apply_updates(update_results['updates'])
        
        console.print(f"[green]✅ {apply_results['successful']} plugins updated successfully[/green]")
        if apply_results['failed'] > 0:
            console.print(f"[red]❌ {apply_results['failed']} plugins failed to update[/red]")
        
    except Exception as e:
        console.print(f"[red]❌ Plugin update failed: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
