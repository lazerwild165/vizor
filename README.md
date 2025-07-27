# 🔍 Vizor v1.0 - Local-first Cybersecurity Copilot

> Your adaptive, privacy-first cybersecurity assistant that acts as both soldier (executor) and advisor (thinker).

## 🎯 Philosophy

Vizor is built on the principle of **local-first autonomy**. It runs entirely on your machine, adapts to your style, and enables independence rather than dependence. No cloud required, no telemetry, just pure cybersecurity intelligence at your fingertips.

## ✨ Key Features

- **🏠 Local-First**: Runs entirely on your machine with your local LLMs
- **🧠 Intelligent Routing**: Automatically selects the best model for each task
- **📚 Adaptive Learning**: Detects knowledge gaps and learns autonomously
- **🔒 Privacy-First**: No telemetry, air-gapped core functionality
- **🔧 Modular Design**: Extensible plugin system for custom integrations
- **⚡ Fast & Lightweight**: Terminal-first interface optimized for speed

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- Your local models: Phi3, Mistral, Deepseek-coder, Wizardcoder

### Installation

```bash
# Clone and install
git clone (https://github.com/lazerwild165/vizor.git)
cd vizor
pip install -e .

# Initialize Vizor
vizor init

# Check status
vizor status
```

### First Steps

```bash
# Ask Vizor a question
vizor ask "What are the latest APT techniques?"

# Generate a daily threat briefing
vizor brief daily

# Scan a file for threats
vizor scan file suspicious.exe

# Build an API wrapper
vizor build api https://api.virustotal.com/v3/openapi.json
```

## 🏗️ Architecture

```
vizor/
├── cli/                    # Command-line interface
│   ├── main.py            # Main CLI application
│   └── commands/          # Individual commands
│       ├── ask.py         # Q&A functionality
│       ├── brief.py       # Threat briefings
│       ├── scan.py        # Security scanning
│       └── build.py       # Plugin/wrapper building
├── brain/                 # Reasoning engine
│   ├── reasoner.py        # Meta-reasoning and routing
│   ├── memory.py          # Vector memory (ChromaDB)
│   ├── gap_detector.py    # Knowledge gap detection
│   └── learning.py        # Automated learning
├── models/                # LLM management
│   └── llm_manager.py     # Model routing and management
├── plugins/               # Plugin system
├── config/                # Configuration management
│   └── settings.py        # Settings and privacy config
└── docs/                  # Documentation
```

## 🤖 Model Integration

Vizor intelligently routes tasks to your local models based on their strengths:

| Model | Strengths | Use Cases |
|-------|-----------|-----------|
| **Phi3** | General queries, reasoning, briefings | Quick Q&A, status reports |
| **Mistral** | Complex reasoning, threat analysis | Strategic analysis, learning coordination |
| **Deepseek-coder** | Code analysis, vulnerability detection | Security code review, exploit analysis |
| **Wizardcoder** | Code generation, security tooling | Writing security scripts, automation |

### Model Selection Logic

```python
# Automatic model selection based on task type
vizor ask "Analyze this malware code"  # → Deepseek-coder
vizor ask "What's the APT landscape?"  # → Mistral  
vizor ask "Generate a Python scanner" # → Wizardcoder
vizor ask "Brief me on today's threats" # → Phi3
```

## 📋 Commands Reference

### Ask Commands
```bash
# Interactive Q&A
vizor ask "How do I detect lateral movement?"
vizor ask --interactive  # Start conversation mode
vizor ask --model mistral "Complex threat analysis question"
```

### Brief Commands
```bash
# Daily briefing
vizor brief daily

# Weekly summary with trends
vizor brief weekly --trends

# Custom topic briefing
vizor brief custom "ransomware campaigns" --depth deep

# Trend analysis
vizor brief trends --period 30d --visualize
```

### Scan Commands
```bash
# File scanning
vizor scan file malware.exe --deep
vizor scan file document.pdf --type metadata

# Network scanning
vizor scan url https://suspicious-site.com --screenshot
vizor scan ip 192.168.1.100 --ports
vizor scan domain evil.com --subdomains

# Hash lookup
vizor scan hash a1b2c3d4e5f6... --sources virustotal,greynoise
```

### Build Commands
```bash
# API wrapper generation
vizor build api https://api.shodan.io/openapi.json --name shodan
vizor build api https://api.greynoise.io/docs --auth api_key

# Plugin development
vizor build plugin threat_intel --name "Custom TI Plugin"
vizor build plugin scanner --name "Network Scanner"

# Plugin management
vizor build list-plugins
vizor build test-plugin ./my-plugin/
vizor build register-plugin ./my-plugin/
vizor build update-plugins
```

### Learning Commands
```bash
# Manual learning
vizor learn "zero-day exploits"
vizor learn "APT29 techniques" --sources cisa.gov,mitre.org

# System commands
vizor init --force          # Reinitialize configuration
vizor status               # System health check
```

## ⚙️ Configuration

### Main Configuration (`~/.vizor/vizor.yaml`)

```yaml
model:
  default_model: mistral
  fallback_models: [phi3, deepseek-coder, wizardcoder]
  temperature: 0.7
  max_tokens: 2048

vector:
  provider: chromadb
  collection_name: vizor_knowledge
  embedding_model: all-MiniLM-L6-v2
  similarity_threshold: 0.7

privacy:
  telemetry_enabled: false
  cloud_fallback_enabled: false
  encrypt_local_data: true
  log_queries: true

learning:
  auto_learn_enabled: true
  learning_sources: [cisa.gov, mitre.org, nvd.nist.gov]
  update_interval_hours: 24
  confidence_threshold: 0.7
```

### Environment Variables (`~/.vizor/.env`)

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Optional API Keys (for enhanced functionality)
VIRUSTOTAL_API_KEY=your_vt_key_here
GREYNOISE_API_KEY=your_greynoise_key_here

# Optional Cloud Fallback (disabled by default)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

## 🧠 Intelligence Features

### Adaptive Learning
- **Gap Detection**: Automatically identifies knowledge gaps in responses
- **Auto-Learning**: Fetches intelligence from configured sources
- **Memory Updates**: Stores new knowledge in vector memory
- **Confidence Tracking**: Monitors response quality and triggers learning

### Knowledge Sources
- **CISA Advisories**: Latest government threat intelligence
- **MITRE ATT&CK**: Technique and tactic knowledge
- **NVD Database**: Vulnerability information
- **Custom Sources**: Add your own intelligence feeds

### Memory System
- **Vector Storage**: ChromaDB for semantic knowledge retrieval
- **Gap Memory**: Tracks learning priorities (`gap_memory.json`)
- **Conversation Context**: Maintains session context
- **Performance History**: Optimizes model selection over time

## 🔌 Plugin System

### Creating Plugins

```python
# plugins/my_plugin.py
from plugins.base import VizorPlugin

class MyThreatIntelPlugin(VizorPlugin):
    def __init__(self):
        super().__init__(
            name="my_threat_intel",
            version="1.0.0",
            description="Custom threat intelligence plugin"
        )
    
    def enrich_ioc(self, ioc: str) -> dict:
        """Enrich an indicator of compromise"""
        # Your enrichment logic here
        return {"ioc": ioc, "reputation": "clean"}
    
    def get_methods(self):
        return ["enrich_ioc"]
```

### Plugin Templates
- **Basic Plugin**: Simple functionality template
- **Threat Intel**: IOC enrichment and analysis
- **Scanner**: Custom scanning capabilities
- **Enrichment**: Data enhancement plugins

## 🔒 Privacy & Security

### Local-First Design
- **No Cloud Dependencies**: Core functionality works offline
- **No Telemetry**: Zero data collection or phone-home
- **Encrypted Storage**: Local data encryption (optional)
- **Air-Gapped Mode**: Complete isolation from internet

### Optional Cloud Features
- **Explicit Opt-in**: Cloud features require explicit configuration
- **API Key Control**: You control all external API access
- **Fallback Only**: Cloud models only used when local models fail

## 🚀 Advanced Usage

### Dry Run Mode
```bash
# Test commands without execution
vizor --dry-run ask "What would this do?"
vizor --dry-run scan file test.exe
```

### Interactive Mode
```bash
# Start interactive session
vizor ask --interactive

# Available commands in interactive mode
/status          # Show system status
/learn <topic>   # Learn about topic
/clear           # Clear conversation
/confidence 0.8  # Set confidence threshold
```

### Custom Model Selection
```bash
# Force specific model usage
vizor ask --model deepseek-coder "Review this exploit code"
vizor ask --model mistral "Strategic threat analysis"
```

## 📊 Monitoring & Maintenance

### System Health
```bash
vizor status  # Check overall system health
```

### Performance Optimization
- **Model Performance**: Tracks response times and accuracy
- **Memory Management**: Automatic cleanup of old knowledge
- **Gap Prioritization**: Focuses learning on high-impact areas

### Self-Growth Features
- **Auto-Updates**: Plugin and wrapper updates
- **Learning Schedules**: Periodic knowledge refresh
- **Performance Tuning**: Automatic model selection optimization

## 🛠️ Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Testing
```bash
# Run tests
pytest tests/

# Test specific components
pytest tests/test_models.py
pytest tests/test_brain.py
```

### Plugin Development
```bash
# Generate plugin template
vizor build plugin basic --name "MyPlugin"

# Test plugin
vizor build test-plugin ./my-plugin/

# Register plugin
vizor build register-plugin ./my-plugin/
```

## 📚 Examples

### Daily Workflow
```bash
# Morning briefing
vizor brief daily --email

# Investigate suspicious file
vizor scan file suspicious.docx --deep

# Ask follow-up questions
vizor ask "What are macro-based attacks?"

# Learn about new threats
vizor learn "macro malware techniques"
```

### Incident Response
```bash
# Quick IOC analysis
vizor scan hash a1b2c3d4e5f6...
vizor scan ip 192.168.1.100

# Generate incident brief
vizor brief custom "network intrusion" --depth deep

# Code analysis
vizor ask --model deepseek-coder "Analyze this PowerShell script"
```

### Threat Hunting
```bash
# Weekly threat trends
vizor brief trends --period 7d

# Custom threat research
vizor learn "APT40 techniques"
vizor ask "What are APT40's latest TTPs?"

# Build custom scanner
vizor build plugin scanner --name "APT40 Hunter"
```

## 🤝 Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Security**: Report security issues privately

## 📄 License

MIT License - see `LICENSE` file for details.

## 🙏 Acknowledgments

- **Ollama Team**: For local LLM infrastructure
- **ChromaDB**: For vector storage capabilities
- **Typer**: For excellent CLI framework
- **Security Community**: For threat intelligence and knowledge

---

**Vizor v1.0** - Built for cybersecurity professionals who value privacy, autonomy, and intelligence.

*"Your local cybersecurity copilot - always ready, always private, always learning."*
