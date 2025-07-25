# Virtual Agora

A terminal-based application that facilitates structured, multi-agent discussions on complex topics using large language models (LLMs) from various providers.

## Overview

Virtual Agora simulates a deliberative assembly where AI agents from different providers (Google, OpenAI, Anthropic, Grok) engage in structured discussions. The system features democratic agenda setting, human-in-the-loop controls, and automatic report generation.

## Features

- **Multi-Provider Agent Pool**: Configure agents from different LLM providers
- **Democratic Process**: Agents propose and vote on discussion topics
- **Human Control**: Approve agendas and control discussion flow
- **Structured Discussions**: Turn-based, moderated conversations
- **Automatic Reporting**: Generate comprehensive discussion summaries
- **Rich Terminal UI**: Color-coded output for clarity

## Installation

1. Clone the repository:
```bash
git clone https://github.com/virtualagora/virtual-agora.git
cd virtual-agora
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. Configure your agents:
```bash
cp examples/config.example.yml config.yml
# Edit config.yml to set up your agent configuration
```

## Usage

Run the application:
```bash
python -m virtual_agora
```

Or if installed via pip:
```bash
virtual-agora
```

## Configuration

The application is configured via `config.yml`. See `examples/config.example.yml` for a template.

### Example Configuration

```yaml
# The Moderator is a dedicated agent responsible for facilitation
moderator:
  provider: Google
  model: gemini-1.5-pro

# The list of agents that will participate in the discussion
agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2  # Creates gpt-4o-1 and gpt-4o-2
  
  - provider: Anthropic
    model: claude-3-opus-20240229
    count: 1
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
```

### Type Checking

```bash
mypy src
```

## Project Structure

```
virtual-agora/
├── src/virtual_agora/    # Main application code
│   ├── agents/          # Agent implementations
│   ├── config/          # Configuration management
│   ├── core/            # Core application logic
│   ├── providers/       # LLM provider integrations
│   ├── ui/              # Terminal UI components
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── docs/                # Documentation
├── logs/                # Session logs
└── reports/             # Generated reports
```

## License

MIT License - see LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.