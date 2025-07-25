#!/usr/bin/env python3
"""Main entry point for Virtual Agora application.

This module provides the command-line interface and orchestrates
the application startup and execution flow.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.traceback import install as install_rich_traceback

from virtual_agora import __version__
from virtual_agora.utils.logging import setup_logging, get_logger
from virtual_agora.utils.exceptions import VirtualAgoraError, ConfigurationError


# Install rich traceback handler for better error messages
install_rich_traceback(show_locals=True)

# Initialize console and logger
console = Console()
logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="virtual-agora",
        description="Facilitate structured multi-agent AI discussions",
        epilog="For more information, visit https://github.com/virtualagora/virtual-agora",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version and exit",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to configuration file (default: config.yml)",
    )
    
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to environment file (default: .env)",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the discussion",
    )
    
    return parser.parse_args()


async def run_application(args: argparse.Namespace) -> int:
    """Run the Virtual Agora application.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Set up logging
        setup_logging(level=args.log_level)
        logger.info(f"Starting Virtual Agora v{__version__}")
        
        # Check if configuration file exists
        if not args.config.exists():
            raise ConfigurationError(
                f"Configuration file not found: {args.config}\n"
                "Please create a config.yml file or specify a different path with --config"
            )
        
        # Check if environment file exists
        if not args.env.exists():
            console.print(
                f"[yellow]Warning:[/yellow] Environment file not found: {args.env}\n"
                "Please create a .env file with your API keys or specify a different path with --env"
            )
        
        # TODO: Load environment variables
        # TODO: Load and validate configuration
        # TODO: Initialize providers
        # TODO: Initialize agents
        # TODO: Run discussion workflow
        
        if args.dry_run:
            console.print("[green]Configuration validation successful![/green]")
            return 0
            
        console.print("\n[bold cyan]Welcome to Virtual Agora![/bold cyan]")
        console.print("A structured multi-agent discussion platform\n")
        
        # Placeholder for main application logic
        console.print("[yellow]Application implementation in progress...[/yellow]")
        
        return 0
        
    except ConfigurationError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        logger.error(f"Configuration error: {e}")
        return 1
        
    except VirtualAgoraError as e:
        console.print(f"[red]Application Error:[/red] {e}")
        logger.error(f"Application error: {e}")
        return 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Discussion interrupted by user[/yellow]")
        logger.info("Application interrupted by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        console.print(f"[red]Unexpected Error:[/red] {e}")
        logger.exception("Unexpected error occurred")
        return 1


def main() -> None:
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Disable color if requested
    if args.no_color:
        console.no_color = True
    
    # Run the async application
    exit_code = asyncio.run(run_application(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()