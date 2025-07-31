# Virtual Agora 🏛️

> A terminal-based application that orchestrates structured multi-agent AI discussions using democratic deliberation principles.

Virtual Agora enables sophisticated discussions where AI agents from different providers (Google Gemini, OpenAI GPT, Anthropic Claude, and Grok) engage in structured, turn-based conversations on complex topics. Built on a LangGraph state machine architecture, it simulates the democratic processes of the ancient Athenian Agora.

## ✨ Key Features

- **🤖 Multi-Provider AI Agents**: Combine agents from Google, OpenAI, Anthropic, and Grok in a single discussion
- **🏗️ Specialized Agent Architecture**: Five distinct agent types optimized for specific roles:
  - **Discussion Agents**: Primary debate participants
  - **Moderator Agent**: Process facilitation and vote synthesis
  - **Summarizer Agent**: Round compression and context management
  - **Report Writer Agent**: Comprehensive topic and session analysis
- **🗳️ Democratic Process**: Agents propose topics, vote on agendas, and reach consensus through structured debate
- **👤 Human-in-the-Loop Control**: Multiple checkpoints for user oversight and intervention
- **📊 Rich Reporting**: Multi-level documentation from round summaries to final comprehensive reports
- **🎨 Beautiful Terminal UI**: Color-coded interface with real-time progress indicators

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **API keys** from at least one AI provider
- **Terminal/Command Prompt** access

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mpelos/virtual-agora.git
cd virtual-agora

# Install dependencies
pip install -r requirements.txt

# Install for global access (optional)
pip install -e .
```

### 2. API Keys Setup

Create a `.env` file in the project root:

```bash
# Add your API keys (only need keys for providers you'll use)
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GROK_API_KEY=your_grok_api_key_here  # If available
```

**Getting API Keys:**
- **Google Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI**: [OpenAI API Keys](https://platform.openai.com/api-keys)
- **Anthropic**: [Anthropic Console](https://console.anthropic.com/account/keys)

### 3. Configuration

Copy and customize the configuration:

```bash
cp config.example.yml config.yml
```

Edit `config.yml` to specify your agents:

```yaml
# Specialized agents for process management
moderator:
  provider: Google
  model: gemini-2.5-pro

summarizer:
  provider: OpenAI
  model: gpt-4o

report_writer:
  provider: Anthropic
  model: claude-3-opus-20240229

# Discussion participants (mix providers for diverse perspectives)
agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2  # Creates gpt-4o-1 and gpt-4o-2

  - provider: Anthropic
    model: claude-3-opus-20240229
    count: 1

  - provider: Google
    model: gemini-2.5-pro
    count: 1
```

### 4. Run Your First Discussion

```bash
# Start Virtual Agora
virtual-agora

# Or validate setup first
virtual-agora --dry-run
```

## 🎮 How It Works

### The Democratic Process

1. **Theme Setting**: You provide the high-level discussion topic
2. **Agenda Creation**: Agents democratically propose and vote on specific sub-topics
3. **Structured Debate**: Agents discuss each agenda item in rotating turns
4. **Checkpoints**: Regular opportunities for you to guide or conclude discussions
5. **Consensus Building**: Agents vote on when to conclude topics (majority + 1 rule)
6. **Comprehensive Reporting**: Detailed reports generated at multiple levels

### Human Control Points

- **Theme Approval**: You set the overall discussion direction
- **Agenda Review**: Approve, edit, or reject the proposed discussion plan
- **Periodic Stops**: Every 3 rounds by default (configurable), you can intervene or redirect
- **Topic Transitions**: Control when to move between agenda items
- **Session Management**: End discussions or continue to new topics at any time

### Output Files

```
reports/session_YYYYMMDD_HHMMSS/
├── [Topic_Name]/
│   └── topic_report_YYYYMMDD_HHMMSS.md
├── final_report/
│   ├── 01_Executive_Summary.md
│   ├── 02_Overarching_Themes.md
│   └── [additional sections...]
└── logs/
    └── session_YYYYMMDD_HHMMSS.log
```

## ⚙️ Configuration Options

### Model Selection Tips

- **Moderator**: Use capable models (gemini-2.5-pro, gpt-4o, claude-3-opus)
- **Participants**: Mix different providers for diverse perspectives
- **Cost Management**: Balance expensive and cheaper models based on your needs

### Command Line Options

```bash
# Custom configuration file
virtual-agora --config my-config.yml

# Custom environment file
virtual-agora --env .env.production

# Enable debug logging
virtual-agora --log-level DEBUG

# Validate configuration only
virtual-agora --dry-run
```

### Context Documents

Place `.txt` or `.md` files in a `context/` directory to provide additional context that will be available to all agents during discussions. Files are automatically loaded and included in agent prompts.

## 🔧 Troubleshooting

**API Key Issues:**
- Ensure `.env` file is in project root
- Verify API keys are valid and have sufficient quota
- Check exact variable naming (case-sensitive)

**Model Availability:**
- Some models require special access (GPT-4, Claude Opus)
- Try alternative models if your original choice is unavailable

**Slow Performance:**
- Consider using faster models like `gpt-3.5-turbo` or `claude-3-haiku`
- Check internet connection and API status

## 📖 Documentation

- **[How Discussions Work](docs/how-discussions-work.md)**: Detailed explanation of the discussion process
- **[Configuration Guide](docs/configuration.md)**: Complete configuration reference
- **[Environment Setup](docs/environment-setup.md)**: Detailed API key and environment setup

## 🤝 Use Cases

- **Research & Analysis**: Explore complex topics from multiple AI perspectives
- **Strategic Planning**: Generate diverse approaches to business challenges
- **Decision Making**: Deliberate on important choices with AI input
- **Educational Content**: Create structured explorations of different viewpoints
- **Academic Research**: Study AI behavior, consensus building, and group dynamics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to facilitate your first AI discussion?** 🚀

```bash
virtual-agora
```