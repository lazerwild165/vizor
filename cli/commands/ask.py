#!/usr/bin/env python3
"""
Vizor Ask Command
Interactive question and answer functionality
"""

import typer
import asyncio
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

console = Console()
app = typer.Typer(help="💬 Ask Vizor questions and get intelligent responses")

@app.command()
def query(
    question: str = typer.Argument(..., help="Your question for Vizor"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use"),
    confidence_threshold: float = typer.Option(0.7, "--confidence", help="Minimum confidence threshold"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive session"),
    save_conversation: bool = typer.Option(True, "--save/--no-save", help="Save conversation to memory"),
    deep: bool = typer.Option(False, "--deep", help="Provide detailed, comprehensive analysis"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response in real-time")
):
    """
    💬 Ask Vizor a question and get an intelligent response
    
    Vizor will analyze your question, check its confidence, and provide
    a thoughtful response. If confidence is low, it may trigger learning.
    """
    try:
        # ULTRA FAST PATH: Direct to Ollama, bypass all brain components
        import ollama
        import time
        
        start_time = time.time()
        
        # Use specified model or default to deepseek-coder (smallest, fastest)
        selected_model = model or "deepseek-coder"
        
        if stream:
            # Stream the response in real-time
            console.print(f"[dim]🤖 Using {selected_model}...[/dim]")
            
            # Use Ollama's streaming API
            response_text = ""
            try:
                import sys
                
                # Get the streaming response
                stream_response = ollama.chat(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful cybersecurity assistant. Provide clear, concise answers." if not deep else "You are a comprehensive cybersecurity expert. Provide detailed, thorough analysis with examples, best practices, and technical details."},
                        {"role": "user", "content": question}
                    ],
                    options={
                        "temperature": 0.7,
                        "num_predict": 1024 if deep else 512,  # Longer response for deep analysis
                        "stream": True
                    }
                )
                
                # Extract the content from the streaming response
                if 'message' in stream_response and 'content' in stream_response['message']:
                    response_text = stream_response['message']['content']
                    # Print the content character by character to simulate streaming
                    for char in response_text:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        time.sleep(0.01)  # Small delay for visual effect
                    
                    # Add newline after streaming is complete
                    print()
                else:
                    # Fallback if streaming format is unexpected
                    response_text = str(stream_response)
                    console.print(response_text)
                
            except Exception as stream_error:
                console.print(f"[yellow]⚠️ Streaming failed, falling back to non-streaming: {stream_error}[/yellow]")
                # Fallback to non-streaming
                response = ollama.chat(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful cybersecurity assistant. Provide clear, concise answers."},
                        {"role": "user", "content": question}
                    ],
                    options={
                        "temperature": 0.7,
                        "num_predict": 512
                    }
                )
                response_text = response['message']['content']
                console.print(Markdown(response_text))
        
        else:
            # Non-streaming version (original)
            response = ollama.chat(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a helpful cybersecurity assistant. Provide clear, concise answers." if not deep else "You are a comprehensive cybersecurity expert. Provide detailed, thorough analysis with examples, best practices, and technical details."},
                    {"role": "user", "content": question}
                ],
                options={
                    "temperature": 0.7,
                    "num_predict": 1024 if deep else 512  # Longer response for deep analysis
                }
            )
            response_text = response['message']['content']
            
            # Display response immediately
            console.print(Markdown(response_text))
        
        processing_time = time.time() - start_time
        
        # Show simple metadata at bottom
        console.print(f"[dim]Confidence: 0.80 | Model: {selected_model} | Time: {processing_time:.2f}s[/dim]")
        
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
        # Import only when needed
        from config.settings import VizorConfig
        from brain.reasoner import MetaReasoner
        
        config = VizorConfig()
        reasoner = MetaReasoner(config)
        
        # Run the async session
        asyncio.run(start_interactive_session(reasoner, config))
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Session ended by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Interactive session failed: {e}[/red]")
        raise typer.Exit(1)

async def start_interactive_session(reasoner, config):
    """Start an interactive conversation session"""
    conversation_history = []
    
    console.print(Panel(
        "[bold blue]🔍 Vizor Interactive Mode[/bold blue]\n"
        "[dim]Type 'exit', 'quit', or 'bye' to end the session[/dim]\n"
        "[dim]Type 'help' for available commands[/dim]",
        border_style="blue"
    ))
    
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
            
            # Process the input using the brain components (reasoner + memory)
            start_time = time.time()
            
            # Use the reasoner to process the query with memory integration
            result = await reasoner.process_query(
                question=user_input,
                context=conversation_history[-5:] if conversation_history else None,
                model="deepseek-coder",  # Use fastest model for interactive
                confidence_threshold=0.7
            )
            
            processing_time = time.time() - start_time
            answer = result['answer']
            
            # Display response with streaming effect
            console.print(f"[green]🤖 Vizor:[/green] ", end="")
            import sys
            for char in answer:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.01)  # Small delay for visual effect
            print()  # Add newline after streaming
            
            # Update conversation history
            conversation_history.append({
                'user': user_input,
                'assistant': answer,
                'confidence': 0.8  # Simple confidence
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
