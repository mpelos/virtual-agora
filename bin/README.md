# Virtual Agora Executables

This directory contains executable scripts for running Virtual Agora directly from the source code.

## Files

- **`virtual-agora`** - Unix/Linux/macOS executable script
- **`virtual-agora.bat`** - Windows batch script
- **`README.md`** - This documentation file

## Usage

### Unix/Linux/macOS

From the project root directory:

```bash
# Make executable (if needed)
chmod +x bin/virtual-agora

# Run the application
./bin/virtual-agora

# With options
./bin/virtual-agora --help
./bin/virtual-agora --dry-run
./bin/virtual-agora --config my-config.yml
```

### Windows

From the project root directory:

```cmd
# Run the application
bin\virtual-agora.bat

# With options
bin\virtual-agora.bat --help
bin\virtual-agora.bat --dry-run
bin\virtual-agora.bat --config my-config.yml
```

## Installation Alternative

For system-wide installation that adds `virtual-agora` to your PATH:

```bash
# Install in development mode
pip install -e .

# Then use anywhere
virtual-agora --help
```

This installs the application and creates a `virtual-agora` command available from anywhere in your system.

## Requirements

- Python 3.10 or higher
- All dependencies installed (see requirements.txt)
- Valid API keys configured in .env file (for actual usage)

## Quick Start

1. Copy `examples/config.example.yml` to `config.yml`
2. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```
3. Run: `./bin/virtual-agora` (Unix) or `bin\virtual-agora.bat` (Windows)

## Development

These scripts work by adding the `src/` directory to Python's module path, allowing you to run Virtual Agora directly from source code without installation.