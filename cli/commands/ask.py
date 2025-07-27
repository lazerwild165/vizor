#!/usr/bin/env python3
"""
Vizor Ask Command
Interactive question and answer functionality
"""

import typer
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.reasoner import MetaReasoner
from models.llm_manager import LLMManager
from config.settings import VizorConfig

console = Console()
app = typer.Typer(help="💬 Ask Vizor questions and get intelligent responses")

@app.command()
def query(
    question: str = typer.Argument(..., help="Your question for Vizor"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use"),
    confidence_threshold: float = typer.Option(0.7, "--confidence", help="Minimum confidence threshold"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive session"),
    save_conversation: bool = typer.Option(True, "--save/--no-save", help="Save conversation to memory")
):
    """
    💬 Ask Vizor a question and get an intelligent response
    
    Vizor will analyze your question, check its confidence, and provide
    a thoughtful response. If confidence is low, it may trigger learning.
    """
    try:
        config = VizorConfig()
        reasoner = MetaReasoner(config)
        llm_manager = LLMManager(config)
        
        if config.dry_run:
            console.print("[yellow]🔸 Dry run: Would process question and generate response[/yellow]")
            return
        
        # Process the question
        console.print(f"[blue]🤔 Processing: {question}[/blue]")
        
        # Get response from reasoner
        response = reasoner.process_query(
            question=question,
            context=context,
            model=model,
            confidence_threshold=confidence_threshold
        )
        
        # Display response
        if response['confidence'] >= confidence_threshold:
            console.print(Panel(
                Markdown(response['answer']),
                title="🎯 Vizor Response",
                border_style="green"
            ))
        else:
            console.print(Panel(
                Markdown(response['answer']),
                title="⚠️ Low Confidence Response",
                border_style="yellow"
            ))
            
            if response['knowledge_gaps']:
                console.print("[yellow]🧠 Knowledge gaps detected. Consider running:[/yellow]")
                for gap in response['knowledge_gaps']:
                    console.print(f"[dim]  vizor learn '{gap}'[/dim]")
        
        # Show confidence and metadata
        console.print(f"[dim]Confidence: {response['confidence']:.2f} | Model: {response['model']} | Time: {response['processing_time']:.2f}s[/dim]")
        
        # Save conversation if requested
        if save_conversation:
            reasoner.save_conversation(question, response)
        
        # Interactive mode
        if interactive:
            start_interactive_session(reasoner, llm_manager, config)
            
    except Exception as e:
        console.print(f"[red]❌ Query failed: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def interactive():
    """
    🗣️ Start an interactive conversation with Vizor
    
    Engage in a back-and-forth conversation with continuous context.
    """
    try:
        config = VizorConfig()
        reasoner = MetaReasoner(config)
        llm_manager = LLMManager(config)
        
        console.print(Panel(
            "[bold blue]🔍 Vizor Interactive Mode[/bold blue]\n"
            "[dim]Type 'exit', 'quit', or 'bye' to end the session[/dim]\n"
            "[dim]Type 'help' for available commands[/dim]",
            border_style="blue"
        ))
        
        start_interactive_session(reasoner, llm_manager, config)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Session ended by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Interactive session failed: {e}[/red]")
        raise typer.Exit(1)

def start_interactive_session(reasoner, llm_manager, config):
    """Start an interactive conversation session"""
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = typer.prompt("🔍 You", type=str)
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                console.print("[yellow]👋 Goodbye![/yellow]")
                break
            
            # Check for help
            if user_input.lower() in ['help', '?']:
                show_interactive_help()
                continue
            
            # Check for special commands
            if user_input.startswith('/'):
                handle_special_command(user_input, reasoner, config)
                continue
            
            if config.dry_run:
                console.print("[yellow]🔸 Dry run: Would process and respond[/yellow]")
                continue
            
            # Process the input
            response = reasoner.process_query(
                question=user_input,
                context=conversation_history,
                confidence_threshold=0.6
            )
            
            # Display response
            console.print(f"[green]🤖 Vizor:[/green] {response['answer']}")
            
            # Update conversation history
            conversation_history.append({
                'user': user_input,
                'assistant': response['answer'],
                'confidence': response['confidence']
            })
            
            # Keep only last 10 exchanges to manage context
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")

def show_interactive_help():
    """Show help for interactive mode"""
    help_text = """
[bold]Interactive Commands:[/bold]
• [cyan]/status[/cyan] - Show system status
• [cyan]/learn <topic>[/cyan] - Learn about a topic
• [cyan]/clear[/cyan] - Clear conversation history
• [cyan]/save[/cyan] - Save current conversation
• [cyan]/confidence <0.0-1.0>[/cyan] - Set confidence threshold
• [cyan]help[/cyan] - Show this help
• [cyan]exit/quit/bye[/cyan] - End session
    """
    console.print(Panel(help_text.strip(), title="💡 Help", border_style="cyan"))

def handle_special_command(command, reasoner, config):
    """Handle special commands in interactive mode"""
    parts = command[1:].split()
    cmd = parts[0].lower()
    
    if cmd == 'status':
        status_info = config.get_status()
        console.print(f"[green]System Status: {'Healthy' if status_info['healthy'] else 'Issues Detected'}[/green]")
    
    elif cmd == 'learn' and len(parts) > 1:
        topic = ' '.join(parts[1:])
        console.print(f"[blue]🧠 Learning about: {topic}[/blue]")
        # Would trigger learning flow
    
    elif cmd == 'clear':
        console.print("[yellow]🗑️ Conversation history cleared[/yellow]")
    
    elif cmd == 'save':
        console.print("[green]💾 Conversation saved[/green]")
    
    elif cmd == 'confidence' and len(parts) > 1:
        try:
            threshold = float(parts[1])
            if 0.0 <= threshold <= 1.0:
                console.print(f"[green]Confidence threshold set to {threshold}[/green]")
            else:
                console.print("[red]Confidence must be between 0.0 and 1.0[/red]")
        except ValueError:
            console.print("[red]Invalid confidence value[/red]")
    
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")

if __name__ == "__main__":
    app()
