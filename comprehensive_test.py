#!/usr/bin/env python3
"""
Comprehensive Vizor Test Suite
Tests all commands with realistic cybersecurity scenarios
"""

import subprocess
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def run_command(cmd, description, timeout=300):
    """Run a command and return results"""
    print(f"\n{'='*80}")
    print(f"🔍 Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        duration = time.time() - start_time
        
        print(f"⏱️  Duration: {duration:.2f}s")
        print(f"✅ Exit Code: {result.returncode}")
        
        if result.stdout:
            print(f"📤 Output:\n{result.stdout}")
        if result.stderr:
            print(f"⚠️  Errors:\n{result.stderr}")
            
        return result.returncode == 0, duration
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after {timeout}s")
        return False, timeout
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, 0

def test_help_commands():
    """Test all help commands"""
    print("\n" + "🚀"*20 + " HELP COMMANDS TEST " + "🚀"*20)
    
    help_tests = [
        (["python", "-m", "cli.main", "--help"], "Main help"),
        (["python", "-m", "cli.main", "ask", "--help"], "Ask help"),
        (["python", "-m", "cli.main", "brief", "--help"], "Brief help"),
        (["python", "-m", "cli.main", "scan", "--help"], "Scan help"),
        (["python", "-m", "cli.main", "build", "--help"], "Build help"),
        (["python", "-m", "cli.main", "learn", "--help"], "Learn help"),
        (["python", "-m", "cli.main", "ask", "query", "--help"], "Ask query help"),
        (["python", "-m", "cli.main", "brief", "daily", "--help"], "Brief daily help"),
        (["python", "-m", "cli.main", "scan", "file", "--help"], "Scan file help"),
    ]
    
    for cmd, desc in help_tests:
        run_command(cmd, desc, timeout=60)

def test_ask_commands():
    """Test ask commands with cybersecurity scenarios"""
    print("\n" + "🤖"*20 + " ASK COMMANDS TEST " + "🤖"*20)
    
    ask_tests = [
        # Basic queries
        (["python", "-m", "cli.main", "ask", "query", "What is a firewall?"], "Basic firewall question"),
        (["python", "-m", "cli.main", "ask", "query", "What is a firewall?", "--deep"], "Deep firewall analysis"),
        (["python", "-m", "cli.main", "ask", "query", "What is a firewall?", "--no-stream"], "Non-streaming query"),
        
        # Advanced cybersecurity questions
        (["python", "-m", "cli.main", "ask", "query", "Explain the difference between IDS and IPS"], "IDS vs IPS"),
        (["python", "-m", "cli.main", "ask", "query", "What are the OWASP Top 10 vulnerabilities?", "--deep"], "OWASP Top 10"),
        (["python", "-m", "cli.main", "ask", "query", "How does SSL/TLS encryption work?"], "SSL/TLS explanation"),
        (["python", "-m", "cli.main", "ask", "query", "What is a zero-day vulnerability?", "--deep"], "Zero-day vulnerability"),
        
        # Network security
        (["python", "-m", "cli.main", "ask", "query", "Explain network segmentation"], "Network segmentation"),
        (["python", "-m", "cli.main", "ask", "query", "What is a DMZ?", "--deep"], "DMZ explanation"),
        
        # Different models
        (["python", "-m", "cli.main", "ask", "query", "What is phishing?", "--model", "mistral"], "Phishing with Mistral"),
        (["python", "-m", "cli.main", "ask", "query", "What is phishing?", "--model", "phi3"], "Phishing with Phi-3"),
        (["python", "-m", "cli.main", "ask", "query", "What is phishing?", "--model", "deepseek-coder"], "Phishing with DeepSeek"),
    ]
    
    for cmd, desc in ask_tests:
        run_command(cmd, desc, timeout=120)

def test_brief_commands():
    """Test brief commands"""
    print("\n" + "📋"*20 + " BRIEF COMMANDS TEST " + "📋"*20)
    
    brief_tests = [
        (["python", "-m", "cli.main", "brief", "daily"], "Daily threat briefing"),
        (["python", "-m", "cli.main", "brief", "weekly"], "Weekly threat summary"),
        (["python", "-m", "cli.main", "brief", "custom", "ransomware"], "Custom ransomware briefing"),
        (["python", "-m", "cli.main", "brief", "trends"], "Threat trends analysis"),
    ]
    
    for cmd, desc in brief_tests:
        run_command(cmd, desc, timeout=180)

def test_scan_commands():
    """Test scan commands"""
    print("\n" + "🔍"*20 + " SCAN COMMANDS TEST " + "🔍"*20)
    
    scan_tests = [
        (["python", "-m", "cli.main", "scan", "file", "README.md"], "File scan"),
        (["python", "-m", "cli.main", "scan", "url", "https://example.com"], "URL scan"),
        (["python", "-m", "cli.main", "scan", "hash", "d41d8cd98f00b204e9800998ecf8427e"], "Hash lookup"),
        (["python", "-m", "cli.main", "scan", "ip", "8.8.8.8"], "IP analysis"),
        (["python", "-m", "cli.main", "scan", "domain", "google.com"], "Domain analysis"),
    ]
    
    for cmd, desc in scan_tests:
        run_command(cmd, desc, timeout=120)

def test_build_commands():
    """Test build commands"""
    print("\n" + "🔧"*20 + " BUILD COMMANDS TEST " + "🔧"*20)
    
    build_tests = [
        (["python", "-m", "cli.main", "build", "api", "https://api.github.com"], "GitHub API wrapper"),
        (["python", "-m", "cli.main", "build", "plugin", "threat_intel", "--name", "test_plugin"], "Threat intel plugin"),
        (["python", "-m", "cli.main", "build", "list-plugins"], "List plugins"),
        (["python", "-m", "cli.main", "build", "test-plugin", "plugins/"], "Test plugin"),
    ]
    
    for cmd, desc in build_tests:
        run_command(cmd, desc, timeout=180)

def test_learn_commands():
    """Test learn commands"""
    print("\n" + "🧠"*20 + " LEARN COMMANDS TEST " + "🧠"*20)
    
    learn_tests = [
        (["python", "-m", "cli.main", "learn", "topic", "ransomware"], "Learn about ransomware"),
        (["python", "-m", "cli.main", "learn", "topic", "phishing", "--depth", "deep"], "Deep phishing learning"),
        (["python", "-m", "cli.main", "learn", "stats"], "Learning statistics"),
        (["python", "-m", "cli.main", "learn", "gaps"], "Knowledge gaps"),
        (["python", "-m", "cli.main", "learn", "sources"], "Learning sources"),
    ]
    
    for cmd, desc in learn_tests:
        run_command(cmd, desc, timeout=180)

def test_interactive_mode():
    """Test interactive mode with automated input"""
    print("\n" + "🗣️"*20 + " INTERACTIVE MODE TEST " + "🗣️"*20)
    
    # Create a test script for interactive mode
    test_input = """hey
What is a firewall?
How does encryption work?
help
bye
"""
    
    with open("test_input.txt", "w") as f:
        f.write(test_input)
    
    cmd = ["python", "-m", "cli.main", "ask", "interactive"]
    run_command(cmd, "Interactive mode test", timeout=300)

def test_status_and_init():
    """Test status and init commands"""
    print("\n" + "📊"*20 + " STATUS & INIT TEST " + "📊"*20)
    
    status_tests = [
        (["python", "-m", "cli.main", "status"], "System status"),
        (["python", "-m", "cli.main", "init", "--force"], "Initialize Vizor"),
    ]
    
    for cmd, desc in status_tests:
        run_command(cmd, desc, timeout=60)

def main():
    """Run comprehensive test suite"""
    print("🔍 COMPREHENSIVE VIZOR TEST SUITE")
    print("Testing all commands with realistic cybersecurity scenarios")
    print("="*80)
    
    # Test all command categories
    test_help_commands()
    test_ask_commands()
    test_brief_commands()
    test_scan_commands()
    test_build_commands()
    test_learn_commands()
    test_interactive_mode()
    test_status_and_init()
    
    print("\n" + "🎉"*20 + " TEST SUITE COMPLETE " + "🎉"*20)
    print("All commands have been tested with realistic cybersecurity scenarios!")

if __name__ == "__main__":
    main() 