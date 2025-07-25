# Environment Setup Guide for Virtual Agora

This guide will help you set up the environment variables required to run Virtual Agora, including obtaining and configuring API keys for various LLM providers.

## Quick Start

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys
3. Set secure file permissions:
   ```bash
   chmod 600 .env
   ```

## Required Environment Variables

Virtual Agora uses environment variables to securely manage API keys and application settings. The required API keys depend on which LLM providers you configure in your `config.yml`.

### API Keys

You only need API keys for the providers you plan to use. Each provider requires its own API key:

| Provider      | Environment Variable | Required Format                     |
| ------------- | -------------------- | ----------------------------------- |
| Google Gemini | `GOOGLE_API_KEY`     | Starts with `AIza` (39 chars total) |
| OpenAI        | `OPENAI_API_KEY`     | Starts with `sk-` (51 chars total)  |
| Anthropic     | `ANTHROPIC_API_KEY`  | Starts with `sk-ant-`               |
| Grok          | `GROK_API_KEY`       | Provider-specific format            |

## Obtaining API Keys

### Google Gemini

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Select your project or create a new one
5. Copy the generated API key
6. Add to your `.env` file:
   ```
   GOOGLE_API_KEY=AIzaSyD-your-actual-key-here
   ```

**Note**: Google API keys typically start with `AIza` followed by 35 characters.

### OpenAI

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Give your key a descriptive name
5. Copy the key immediately (you won't be able to see it again)
6. Add to your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

**Note**: OpenAI keys start with `sk-` followed by 48 characters.

### Anthropic

1. Go to [Anthropic Console](https://console.anthropic.com/account/keys)
2. Sign in or create an account
3. Click "Create Key"
4. Name your key appropriately
5. Copy the generated key
6. Add to your `.env` file:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
   ```

**Note**: Anthropic keys start with `sk-ant-` followed by alphanumeric characters.

### Grok

Grok API access is currently limited. Check the official Grok documentation for availability and instructions. Once you have a key:

```
GROK_API_KEY=your-grok-api-key-here
```

## Optional Environment Variables

These variables have sensible defaults but can be customized:

### Application Settings

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
VIRTUAL_AGORA_LOG_LEVEL=INFO

# Directory for session logs
VIRTUAL_AGORA_LOG_DIR=logs

# Directory for generated reports
VIRTUAL_AGORA_REPORT_DIR=reports

# Session timeout in seconds (default: 3600 = 1 hour, max: 86400 = 24 hours)
VIRTUAL_AGORA_SESSION_TIMEOUT=3600

# Enable debug mode (true/false)
VIRTUAL_AGORA_DEBUG=false

# Disable colored terminal output (true/false)
VIRTUAL_AGORA_NO_COLOR=false
```

## Security Best Practices

### 1. File Permissions

Always set restrictive permissions on your `.env` file:

```bash
# Make the file readable/writable only by you
chmod 600 .env

# Verify permissions
ls -la .env
# Should show: -rw------- 1 youruser yourgroup
```

### 2. Git Security

The `.env` file should NEVER be committed to version control. Verify it's in `.gitignore`:

```bash
# Check if .env is ignored
git check-ignore .env
# Should output: .env

# If not, add it to .gitignore
echo ".env" >> .gitignore
```

### 3. Key Management

- **Use unique keys**: Don't share API keys between projects
- **Rotate regularly**: Change your API keys periodically
- **Monitor usage**: Check your provider dashboards for unexpected usage
- **Limit scope**: Use keys with minimal required permissions
- **Development vs Production**: Use different keys for different environments

### 4. Key Storage

- Never hardcode API keys in your code
- Don't commit keys to version control
- Don't share keys via email or chat
- Use a password manager for key storage
- Consider using secret management services for production

## Environment Validation

Virtual Agora automatically validates your environment on startup:

1. **Checks for required API keys** based on your `config.yml`
2. **Validates key formats** (warns if format seems incorrect)
3. **Verifies file permissions** (warns if `.env` is world-readable)
4. **Provides helpful error messages** with links to obtain missing keys

### Manual Validation

You can test your environment setup:

```bash
# Dry run to validate configuration
python -m virtual_agora --dry-run

# Check specific environment variables
python -c "import os; print('Google Key:', 'Set' if os.getenv('GOOGLE_API_KEY') else 'Not Set')"
```

## Troubleshooting

### Missing API Keys

If you see an error like:

```
Error: Missing API keys for the following providers:

Provider: Google
To get a Google API key:
1. Go to https://makersuite.google.com/app/apikey
...
```

Follow the provided instructions to obtain the missing keys.

### Invalid Key Format

If you see a warning like:

```
Warning: API key for Google may have incorrect format
```

Double-check that you've copied the complete key without any extra spaces or characters.

### Permission Denied

If you get permission errors:

```bash
# Fix file ownership
chown $USER:$USER .env

# Fix file permissions
chmod 600 .env
```

### Environment Not Loading

If environment variables aren't being loaded:

1. Check file location:

   ```bash
   # Should be in project root
   ls -la .env
   ```

2. Verify file format:

   ```bash
   # Should show KEY=value pairs
   cat .env | grep -v '^#' | grep -v '^$'
   ```

3. Check for syntax errors:
   - No spaces around `=`
   - No quotes needed (unless value contains spaces)
   - Each variable on its own line

### API Rate Limits

If you encounter rate limit errors:

- Check your usage on provider dashboards
- Consider upgrading your plan
- Implement request throttling
- Use different keys for testing

## Multiple Environments

For different environments (development, staging, production):

1. Create environment-specific files:

   ```bash
   .env.development
   .env.staging
   .env.production
   ```

2. Load the appropriate file:

   ```bash
   # Development
   python -m virtual_agora --env .env.development

   # Production
   python -m virtual_agora --env .env.production
   ```

3. Or use system environment variables:
   ```bash
   export GOOGLE_API_KEY=your-production-key
   python -m virtual_agora
   ```

## Environment Variables Priority

Virtual Agora loads environment variables in this order (later sources override earlier ones):

1. System environment variables
2. `.env` file (if exists)
3. File specified with `--env` flag
4. Command-line arguments (for supported options)

## Summary

1. Copy `.env.example` to `.env`
2. Add your API keys for the providers you'll use
3. Set secure file permissions (`chmod 600 .env`)
4. Verify `.env` is in `.gitignore`
5. Run the application

Remember: API keys are like passwords - keep them secret, keep them safe!
