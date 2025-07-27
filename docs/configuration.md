# Virtual Agora Configuration Guide

This guide explains how to configure Virtual Agora using the YAML configuration file.

## Configuration File Location

By default, Virtual Agora looks for a `config.yml` file in the current directory. You can specify a different location using the `--config` command-line option:

```bash
virtual-agora --config /path/to/your/config.yml
```

## Configuration Structure

The configuration file has two main sections:

1. **moderator** - Configuration for the AI moderator
2. **agents** - List of discussion agent configurations

### Basic Example

```yaml
# Virtual Agora Configuration File
moderator:
  provider: Google
  model: gemini-2.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2

  - provider: Anthropic
    model: claude-3-opus-20240229
    count: 1

  - provider: Google
    model: gemini-2.5-pro
    count: 1
```

## Configuration Schema

### Moderator Configuration

The moderator section configures the AI agent responsible for facilitating the discussion.

| Field    | Type   | Required | Description                                     |
| -------- | ------ | -------- | ----------------------------------------------- |
| provider | string | Yes      | LLM provider (Google, OpenAI, Anthropic, Grok)  |
| model    | string | Yes      | Model identifier (e.g., gemini-2.5-pro, gpt-4o) |

**Example:**

```yaml
moderator:
  provider: Google
  model: gemini-2.5-pro
```

**Notes:**

- The moderator should use a capable model since it has complex responsibilities
- Recommended models: gemini-2.5-pro, gpt-4o, claude-3-opus

### Agents Configuration

The agents section is a list of agent configurations. Each configuration can create one or more agents.

| Field    | Type    | Required | Default | Description                       |
| -------- | ------- | -------- | ------- | --------------------------------- |
| provider | string  | Yes      | -       | LLM provider                      |
| model    | string  | Yes      | -       | Model identifier                  |
| count    | integer | No       | 1       | Number of agents to create (1-10) |

**Example:**

```yaml
agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2 # Creates gpt-4o-1 and gpt-4o-2

  - provider: Anthropic
    model: claude-3-opus-20240229
    # count defaults to 1, creates claude-3-opus-20240229-1
```

## Supported Providers and Models

### Google (Gemini)

- `gemini-2.5-pro` - Most capable, recommended for moderator
- `gemini-2.5-flash` - Faster, lower cost
- `gemini-pro` - Previous generation

### OpenAI

- `gpt-4o` - Most capable GPT-4 variant
- `gpt-4-turbo` - Faster GPT-4 variant
- `gpt-3.5-turbo` - Faster, lower cost (not recommended for moderator)

### Anthropic

- `claude-3-opus-20240229` - Most capable Claude model
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fastest, lower cost
- `claude-2.1` - Previous generation

### Grok

- Model names to be determined based on API availability

## Validation Rules

1. **Minimum Agents**: At least 2 discussion agents are required
2. **Maximum Agents**: Maximum 20 total agents (for performance)
3. **Agent Count**: Each agent configuration can create 1-10 agents
4. **Model Validation**: Models must match known patterns for each provider
5. **Provider Names**: Case-insensitive (Google, google, GOOGLE all work)

## Best Practices

### Provider Diversity

For richer discussions, use agents from different providers:

```yaml
agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2

  - provider: Anthropic
    model: claude-3-opus-20240229
    count: 2

  - provider: Google
    model: gemini-2.5-pro
    count: 1
```

### Model Selection

- **Moderator**: Use the most capable model available (e.g., gemini-2.5-pro, gpt-4o)
- **Agents**: Mix of models based on your needs:
  - Capable models for complex reasoning
  - Faster models for quick responses
  - Different providers for diverse perspectives

### Cost Optimization

To manage API costs:

- Limit the number of agents (3-5 is often sufficient)
- Mix expensive and cheaper models
- Use faster models for some agents

```yaml
agents:
  - provider: OpenAI
    model: gpt-4o # Expensive but capable
    count: 1

  - provider: OpenAI
    model: gpt-3.5-turbo # Cheaper alternative
    count: 2

  - provider: Anthropic
    model: claude-3-haiku-20240307 # Fast and cheap
    count: 2
```

## Environment Variables

In addition to the configuration file, you need to set API keys as environment variables:

```bash
# Required based on your configuration
export GOOGLE_API_KEY=your_google_api_key
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export GROK_API_KEY=your_grok_api_key
```

Or use a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GROK_API_KEY=your_grok_api_key
```

## Error Messages

Common configuration errors and their solutions:

### Missing Configuration File

```
Configuration Error: Configuration file not found: config.yml
```

**Solution**: Create a config.yml file or specify the correct path with --config

### Invalid YAML Syntax

```
Configuration Error: Invalid YAML syntax in configuration file
```

**Solution**: Check your YAML formatting (proper indentation, no tabs)

### Missing Required Fields

```
Configuration Error: Configuration validation failed:
moderator -> model: Field required
```

**Solution**: Add the missing field to your configuration

### Invalid Model

```
Configuration Error: Configuration validation failed:
moderator -> model: Invalid Google model: invalid-model
```

**Solution**: Use a supported model name for the provider

### Too Few Agents

```
Configuration Error: At least 2 agents required for a discussion
```

**Solution**: Add more agents or increase the count

### Too Many Agents

```
Configuration Error: Too many agents configured (25). Maximum 20 agents allowed
```

**Solution**: Reduce the number of agents or decrease counts

## Advanced Configuration

### Development Configuration

For testing with minimal API usage:

```yaml
moderator:
  provider: OpenAI
  model: gpt-3.5-turbo # Cheaper for testing

agents:
  - provider: OpenAI
    model: gpt-3.5-turbo
    count: 2 # Minimum for discussion
```

### Production Configuration

For high-quality discussions:

```yaml
moderator:
  provider: Google
  model: gemini-2.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2

  - provider: Anthropic
    model: claude-3-opus-20240229
    count: 2

  - provider: Google
    model: gemini-2.5-pro
    count: 1

  # Grok support when available
  # - provider: Grok
  #   model: grok-model-name
  #   count: 1
```
