#!/usr/bin/env python3
"""
Quick Vizor Test - Verify all main commands work
"""

import subprocess
import time

def test_command(cmd, description):
    """Test a single command"""
    print(f"\n🔍 Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({duration:.2f}s)")
            if result.stdout:
                print(f"📤 Output: {result.stdout[:200]}...")
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            if result.stderr:
                print(f"⚠️  Error: {result.stderr}")
                
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>120s)")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def main():
    """Test all main commands"""
    print("🚀 QUICK VIZOR COMMAND TEST")
    print("=" * 60)
    
    # Test all main commands
    tests = [
        # Help commands
        (["python", "-m", "cli.main", "--help"], "Main help"),
        (["python", "-m", "cli.main", "ask", "--help"], "Ask help"),
        
        # Ask commands
        (["python", "-m", "cli.main", "ask", "query", "What is a firewall?"], "Basic ask"),
        (["python", "-m", "cli.main", "ask", "query", "What is phishing?", "--deep"], "Deep ask"),
        (["python", "-m", "cli.main", "ask", "query", "What is encryption?", "--no-stream"], "Non-streaming ask"),
        
        # Brief commands
        (["python", "-m", "cli.main", "brief", "daily"], "Daily briefing"),
        (["python", "-m", "cli.main", "brief", "weekly"], "Weekly briefing"),
        
        # Scan commands
        (["python", "-m", "cli.main", "scan", "file", "README.md"], "File scan"),
        (["python", "-m", "cli.main", "scan", "url", "https://example.com"], "URL scan"),
        
        # Learn commands
        (["python", "-m", "cli.main", "learn", "topic", "phishing"], "Learn topic"),
        (["python", "-m", "cli.main", "learn", "topic", "malware", "--depth", "deep"], "Deep learning"),
        
        # Status and init
        (["python", "-m", "cli.main", "status"], "System status"),
    ]
    
    for cmd, desc in tests:
        test_command(cmd, desc)
    
    print("\n" + "🎉" * 20 + " TEST COMPLETE " + "🎉" * 20)

if __name__ == "__main__":
    main() 