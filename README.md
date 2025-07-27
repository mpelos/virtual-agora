# Virtual Agora 🏛️

> A structured multi-agent AI discussion platform that facilitates democratic deliberation between different AI models.

Virtual Agora enables you to host sophisticated discussions where AI agents from different providers (Google Gemini, OpenAI GPT, Anthropic Claude, and Grok) engage in structured, turn-based conversations on complex topics. Think of it as a digital version of the ancient Athenian Agora, where diverse perspectives come together for democratic discourse.

## ✨ Key Features

- **🤖 Multi-Provider AI Agents**: Combine agents from Google, OpenAI, Anthropic, and Grok in a single discussion
- **🗳️ Democratic Agenda Setting**: Agents propose and vote on discussion topics
- **👨‍💼 AI Moderator**: Neutral moderator manages the flow and ensures relevance
- **👤 Human-in-the-Loop Control**: You approve agendas and control session flow
- **📊 Structured Discussions**: Turn-based rounds with rotating speakers
- **✊ Minority Voice Protection**: Dissenting agents get final considerations before topic closure
- **📝 Automatic Reporting**: Comprehensive reports generated for each session
- **🎨 Rich Terminal UI**: Beautiful, color-coded interface with real-time updates
- **💾 Session Recovery**: Built-in error handling and session state management

## 🎯 Use Cases

- **📚 Research & Analysis**: Explore complex topics from multiple AI perspectives
- **🧠 Strategic Planning**: Generate diverse approaches to business challenges
- **🔍 Decision Making**: Deliberate on important choices with AI input
- **📖 Educational Content**: Create structured content exploring different viewpoints
- **🎭 Creative Writing**: Develop stories, scenarios, or content with multiple AI collaborators
- **🔬 Academic Research**: Study AI behavior, consensus building, and group dynamics

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed on your system
- **API keys** from at least one AI provider (Google, OpenAI, Anthropic, or Grok)
- **Terminal/Command Prompt** access

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/mpelos/virtual-agora.git
cd virtual-agora

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional, for global access)
pip install -e .
```

### 2. Get API Keys

You'll need API keys from at least one provider. Virtual Agora supports:

#### Google Gemini (Recommended for moderator)

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key

#### OpenAI GPT

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy your key

#### Anthropic Claude

1. Go to [Anthropic Console](https://console.anthropic.com/account/keys)
2. Click "Create Key"
3. Copy your key

#### Grok (X.AI)

1. Contact X.AI for API access
2. Follow their setup instructions

### 3. Configuration

#### Create Environment File

Create a `.env` file in the project root:

```bash
# Required: Add your API keys
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GROK_API_KEY=your_grok_api_key_here  # If available

# Optional: Customize behavior
LOG_LEVEL=INFO
```

#### Configure Agents

Copy the example configuration:

```bash
cp examples/config.example.yml config.yml
```

Edit `config.yml` to customize your discussion setup:

```yaml
# The moderator facilitates discussion (recommended: Google Gemini)
moderator:
  provider: Google
  model: gemini-2.5-pro

# Discussion participants (mix different providers for diverse perspectives)
agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2 # Creates gpt-4o-1 and gpt-4o-2

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
./bin/virtual-agora

# Or if installed globally
virtual-agora

# Validate setup without running
virtual-agora --dry-run
```

## 🎮 How to Use

### Starting a Session

1. **Launch the application**: Run `./bin/virtual-agora`
2. **Enter your topic**: Provide a clear, discussable topic when prompted

   _Good examples:_

   - "The ethics of artificial intelligence in healthcare"
   - "Strategies for sustainable urban development"
   - "The future of remote work post-pandemic"

3. **Review the agenda**: Agents will propose sub-topics and vote. You can approve, edit, or reject their proposed agenda.

### During Discussion

- **Watch the conversation**: Agents take turns speaking in rotating order
- **Monitor progress**: The moderator summarizes each round
- **Control topic transitions**: You decide when to move between topics
- **Manage the flow**: You can end the session at any time

### Topic Conclusion

- **Automatic polling**: After a few rounds, agents vote on whether to conclude the current topic
- **Majority rule**: Topics close when majority+1 agents vote "yes"
- **Minority protection**: Dissenting agents get a final word before closure
- **Human approval**: You control whether to continue to the next topic

### Session End

- **Automatic reports**: Complete discussion reports are generated
- **File outputs**: Individual topic summaries and final reports saved as Markdown files
- **Session logs**: Full transcripts saved for later review

## ⚙️ Advanced Configuration

### Model Selection

Choose models based on your needs:

**For Moderators** (need reasoning and neutrality):

- `gemini-2.5-pro` - Excellent reasoning and neutrality
- `gpt-4o` - Strong analytical capabilities
- `claude-3-opus-20240229` - Excellent at following instructions

**For Participants** (diverse perspectives):

- Mix different providers for varied viewpoints
- Use different model sizes for different "personality" types
- Consider model strengths (creative, analytical, balanced)

### Advanced Options

```yaml
# Example advanced configuration
moderator:
  provider: Google
  model: gemini-2.5-pro
  temperature: 0.3 # Lower = more consistent
  max_tokens: 2000 # Response length limit

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 1
    temperature: 0.7 # Higher = more creative

  - provider: Anthropic
    model: claude-3-sonnet-20240229 # Faster, cheaper alternative
    count: 2
    temperature: 0.5
```

### Command Line Options

```bash
# Use custom configuration
virtual-agora --config my-custom-config.yml

# Use custom environment file
virtual-agora --env .env.production

# Enable debug logging
virtual-agora --log-level DEBUG

# Disable colors (for logging/scripting)
virtual-agora --no-color

# Validate configuration only
virtual-agora --dry-run
```

## 📊 Understanding the Output

### During Session

- **💬 Agent Messages**: Color-coded by provider (Blue=Google, Green=OpenAI, Purple=Anthropic)
- **🎯 Moderator Actions**: White text, manages process flow
- **🟡 User Prompts**: Yellow text, waiting for your input
- **🔴 System Messages**: Red text, errors or important notices

### Generated Files

```
reports/
├── session_YYYYMMDD_HHMMSS/
│   ├── topic_summary_topic1.md       # Individual topic summaries
│   ├── topic_summary_topic2.md
│   ├── final_report_01_executive_summary.md  # Final report sections
│   ├── final_report_02_detailed_analysis.md
│   └── final_report_03_conclusions.md
│
logs/
└── session_YYYYMMDD_HHMMSS.log       # Complete session transcript
```

## 🔧 Troubleshooting

### Common Issues

**"API key not found" error:**

- Ensure your `.env` file is in the project root
- Check that API key variable names match exactly
- Verify keys are valid and have sufficient quota

**"Model not available" error:**

- Check if you have access to the specified model
- Some models require special access (like GPT-4 or Claude Opus)
- Try a different model from the same provider

**Slow responses:**

- Some models are slower than others
- Check your internet connection
- Consider using faster models like `gpt-3.5-turbo` or `claude-3-haiku`

**Discussion gets stuck:**

- The human-in-the-loop controls prevent runaway conversations
- You can always interrupt with Ctrl+C and restart
- Check logs for detailed error information

### Getting Help

```bash
# Check version and basic info
virtual-agora --version

# Validate your setup
virtual-agora --dry-run

# Enable detailed logging
virtual-agora --log-level DEBUG
```

### Log Files

Detailed logs are saved in `logs/session_YYYYMMDD_HHMMSS.log`:

- Agent responses and reasoning
- Error details and stack traces
- Performance metrics and timing
- State transitions and checkpoints

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Beautiful terminal UI powered by [Rich](https://rich.readthedocs.io/)
- Configuration management with [PyYAML](https://pyyaml.org/)
- Inspired by ancient Athenian democratic principles

## 📞 Support

- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Report bugs on [GitHub Issues](https://github.com/mpelos/virtual-agora/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/mpelos/virtual-agora/discussions)

---

**Ready to facilitate your first AI discussion?** 🚀

```bash
./bin/virtual-agora
```
