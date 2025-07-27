#!/usr/bin/env python3
"""Main entry point for Virtual Agora application.

This module provides the command-line interface and orchestrates
the application startup and execution flow.
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.traceback import install as install_rich_traceback

from virtual_agora import __version__
from virtual_agora.config.loader import ConfigLoader
from virtual_agora.config.validators import ConfigValidator
from virtual_agora.config.env_schema import EnvironmentConfig
from virtual_agora.utils.env_manager import EnvironmentManager
from virtual_agora.utils.security import create_provider_help_message
from virtual_agora.utils.logging import setup_logging, get_logger
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ConfigurationError,
    CriticalError,
    UserInterventionRequired,
)
from virtual_agora.utils.error_handler import error_handler, ErrorContext
from virtual_agora.utils.error_reporter import ErrorReporter
from virtual_agora.utils.shutdown import shutdown_handler, graceful_shutdown_context
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager
from virtual_agora.ui.human_in_the_loop import (
    get_initial_topic,
    get_agenda_approval,
    get_continuation_approval,
    get_agenda_modifications,
    display_session_status,
)
from virtual_agora.ui.preferences import (
    get_preferences_manager,
    get_user_preferences,
)
from virtual_agora.ui.interrupt_handler import setup_interrupt_handlers
from virtual_agora.ui.components import LoadingSpinner, create_header_panel
from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme, ProviderType
from virtual_agora.ui.accessibility import initialize_accessibility
from virtual_agora.ui.langgraph_integration import (
    get_ui_integration,
    initialize_ui_integration,
    update_ui_from_state_change,
)
from virtual_agora.flow.graph import VirtualAgoraFlow


# Install rich traceback handler for better error messages
install_rich_traceback(show_locals=True)

# Initialize enhanced console and logger
enhanced_console = get_console()
console = enhanced_console.rich_console  # For backward compatibility
logger = get_logger(__name__)

# Initialize error reporter
error_reporter = ErrorReporter(console)

# Initialize UI integration
ui_integration = initialize_ui_integration()


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
    # Initialize managers (will be set during startup)
    state_manager: Optional[StateManager] = None
    recovery_manager: Optional[StateRecoveryManager] = None
    session_id: Optional[str] = None
    preferences_manager = get_preferences_manager()
    interrupt_handler = None

    # Create a wrapper to properly register cleanup after managers are initialized
    def register_cleanup():
        if state_manager and recovery_manager and session_id:
            shutdown_handler.register_cleanup_task(
                lambda: shutdown_handler.save_session_state(
                    state_manager,
                    recovery_manager,
                    session_id,
                ),
                name="save_session_state",
            )
            shutdown_handler.register_cleanup_task(
                lambda: shutdown_handler.generate_shutdown_report(
                    error_reporter,
                    session_id,
                ),
                name="generate_shutdown_report",
            )

    try:
        # Set up logging
        setup_logging(level=args.log_level)
        logger.info(f"Starting Virtual Agora v{__version__}")

        # Track startup in error handler
        with error_handler.error_boundary("application_startup"):

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

            # Load and validate environment variables
            env_manager = EnvironmentManager(
                env_file=args.env if args.env.exists() else None
            )
            env_manager.load()

            # Get environment status
            env_status = env_manager.get_status_report()
            logger.debug(f"Environment status: {env_status}")

            # Load and validate configuration
            config_loader = ConfigLoader(args.config)
            config = config_loader.load()

            # Determine required providers from config
            required_providers = {config.moderator.provider.value}
            for agent in config.agents:
                required_providers.add(agent.provider.value)

            # Validate API keys
            missing_providers = env_manager.get_missing_providers(required_providers)
            if missing_providers:
                console.print(
                    "[red]Error:[/red] Missing API keys for the following providers:\n"
                )

                for provider in missing_providers:
                    console.print(f"[yellow]Provider:[/yellow] {provider}")
                    console.print(create_provider_help_message(provider))
                    console.print()

                console.print(
                    "[dim]After adding the keys to your .env file, run the application again.[/dim]"
                )
                return 1

            # Try to load environment config with Pydantic for additional validation
            try:
                env_config = EnvironmentConfig()
                env_config.validate_required_keys(required_providers)

                # Use environment config for application settings
                if env_config.log_level != args.log_level:
                    logger.info(
                        f"Using log level from environment: {env_config.log_level}"
                    )
                    setup_logging(level=env_config.log_level)

            except ConfigurationError as e:
                # Already handled above, this shouldn't happen
                logger.error(f"Environment validation error: {e}")
                return 1
            except Exception as e:
                logger.warning(f"Could not load environment config with Pydantic: {e}")
                # Continue with basic environment manager

            # Validate configuration
            validator = ConfigValidator(config)
            validation_report = validator.get_validation_report()

            # Show configuration summary
            console.print("\n[bold cyan]Configuration Summary:[/bold cyan]")
            console.print(
                f"Moderator: {config.moderator.model} ({config.moderator.provider.value})"
            )
            console.print(f"Total Agents: {config.get_total_agent_count()}")

            # Show agent breakdown
            console.print("\n[bold]Discussion Agents:[/bold]")
            for agent_config in config.agents:
                console.print(
                    f"  - {agent_config.model} ({agent_config.provider.value}) "
                    f"x{agent_config.count}"
                )

            # Show any validation warnings
            if validation_report.get("warnings"):
                console.print("\n[yellow]Configuration Warnings:[/yellow]")
                for warning in validation_report["warnings"]:
                    console.print(f"  ⚠️  {warning}")

            # Run additional validation
            try:
                validator.validate_all()
            except ConfigurationError as e:
                console.print(f"\n[red]Configuration Error:[/red] {e}")
                return 1

            # Load user preferences
            prefs = preferences_manager.load()
            logger.info(
                f"Loaded user preferences (verbosity: {prefs.display_verbosity})"
            )

            # Apply preference for colored output
            if not prefs.use_color:
                console.no_color = True

            # Initialize accessibility features
            initialize_accessibility()

            # Initialize state management
            state_manager = StateManager(config)
            recovery_manager = StateRecoveryManager()
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Set up interrupt handlers
            interrupt_handler = setup_interrupt_handlers(
                state_manager, recovery_manager
            )
            logger.info("Interrupt handlers configured")

            # Register cleanup tasks now that managers are initialized
            register_cleanup()

            # Initialize state
            state = state_manager.initialize_state(session_id)
            logger.info(f"Initialized session: {session_id}")

            # Initialize UI for session
            state = ui_integration.initialize_ui_for_session(state)

            # Create initial checkpoint
            recovery_manager.create_checkpoint(
                state,
                operation="session_start",
                save_to_disk=True,
            )

            # Initialize VirtualAgoraFlow
            console.print("[cyan]Initializing discussion flow...[/cyan]")
            flow = VirtualAgoraFlow(config, enable_monitoring=True)
            flow.compile()
            logger.info("VirtualAgoraFlow initialized and compiled")

            if args.dry_run:
                console.print("[green]Configuration validation successful![/green]")
                console.print(
                    "[green]Discussion flow initialized successfully![/green]"
                )
                return 0

            # Welcome header is now shown by UI integration initialization

            # Get initial topic from user (uses enhanced UI internally)
            topic = get_initial_topic()

            # Create session with the topic
            session_id = flow.create_session(session_id=session_id, main_topic=topic)
            logger.info(f"Created discussion session: {session_id}")

            # Update UI with topic
            flow_state_manager = flow.get_state_manager()
            current_state = flow_state_manager.get_state()
            ui_integration.update_ui_from_state(current_state)

            recovery_manager.create_checkpoint(
                current_state,
                operation="session_created",
                save_to_disk=True,
            )

            # Run the complete discussion workflow
            console.print("[cyan]Starting discussion workflow...[/cyan]")

            with LoadingSpinner("Running discussion flow..."):
                try:
                    # Stream the graph execution to get real-time updates
                    config_dict = {"configurable": {"thread_id": session_id}}

                    for update in flow.stream(config_dict):
                        logger.debug(f"Flow update: {update}")

                        # Update UI with each step
                        current_state = flow_state_manager.get_state()
                        ui_integration.update_ui_from_state(current_state)

                        # Create checkpoints at key phases
                        if "current_phase" in update:
                            recovery_manager.create_checkpoint(
                                current_state,
                                operation=f"phase_{update['current_phase']}",
                                save_to_disk=True,
                            )

                    console.print("[green]Discussion completed successfully![/green]")

                    # Get final state and show summary
                    final_state = flow_state_manager.get_state()
                    console.print(
                        f"[dim]Final session ID: {final_state['session_id']}[/dim]"
                    )
                    console.print(
                        f"[dim]Topics discussed: {len(final_state.get('completed_topics', []))}[/dim]"
                    )
                    console.print(
                        f"[dim]Total messages: {len(final_state.get('messages', []))}[/dim]"
                    )

                except Exception as e:
                    logger.error(f"Discussion flow error: {e}")
                    console.print(f"[red]Discussion flow error: {e}[/red]")

                    # Attempt recovery
                    if recovery_manager.emergency_recovery(flow_state_manager, e):
                        console.print(
                            "[yellow]Recovered from discussion error[/yellow]"
                        )
                    else:
                        console.print(
                            "[red]Could not recover from discussion error[/red]"
                        )
                        raise

            return 0

    except ConfigurationError as e:
        context = error_handler.capture_context(
            e,
            operation="configuration",
            phase="startup",
        )
        error_reporter.report_error(context)
        return 1

    except CriticalError as e:
        context = error_handler.capture_context(
            e,
            operation="critical_failure",
            phase="runtime",
        )
        error_reporter.report_error(context, show_details=True)

        # Attempt emergency recovery
        if state_manager and recovery_manager:
            if recovery_manager.emergency_recovery(state_manager, e):
                logger.info("Emergency recovery successful")
                console.print("[yellow]System recovered from critical error[/yellow]")
                return 1
            else:
                logger.critical("Emergency recovery failed")
                console.print("[red]System could not recover from critical error[/red]")

        return 2

    except UserInterventionRequired as e:
        context = error_handler.capture_context(
            e,
            operation="user_input_required",
        )
        error_reporter.report_error(context)

        # Show prompt to user
        console.print(f"\n[yellow]{e.prompt}[/yellow]")
        if e.options:
            for i, option in enumerate(e.options, 1):
                console.print(f"  {i}. {option}")

        return 3

    except VirtualAgoraError as e:
        context = error_handler.capture_context(
            e,
            operation="application_error",
        )
        error_reporter.report_error(context)
        return 1

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        shutdown_handler.request_shutdown("User interrupt")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        context = error_handler.capture_context(
            e,
            operation="unexpected_error",
        )
        error_reporter.report_error(context, show_details=True)
        logger.exception("Unexpected error occurred")
        return 1

    finally:
        # Teardown interrupt handlers
        if interrupt_handler:
            interrupt_handler.teardown()

        # Show error summary if any errors occurred
        error_summary = error_handler.get_error_summary()
        if error_summary["total_errors"] > 0:
            console.print("\n")
            error_reporter.show_error_summary()

            # Detect and show error trends
            trends = error_reporter.detect_error_trends()
            if trends:
                console.print("\n[bold]Error Trends Detected:[/bold]")
                for trend in trends:
                    console.print(f"  • {trend}")


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
