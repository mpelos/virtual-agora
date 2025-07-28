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
from typing import Optional, Dict, Any

from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.panel import Panel
from rich.prompt import Prompt

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
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.app_v13 import VirtualAgoraApplicationV13


def process_interrupt_recursive(
    interrupt_data: Any,
    config_dict: Dict[str, Any],
    depth: int = 0,
    stream_depth: int = 0,
) -> Optional[Dict[str, Any]]:
    """Recursively process LangGraph interrupts including nested interrupts.

    Args:
        interrupt_data: The interrupt data from LangGraph
        config_dict: Configuration dictionary for LangGraph operations
        depth: Current interrupt recursion depth for safety limits
        stream_depth: Current stream recursion depth for safety limits

    Returns:
        User response dictionary or None if processing fails
    """
    # Safety limits to prevent infinite recursion
    MAX_INTERRUPT_DEPTH = 5
    MAX_STREAM_DEPTH = 3

    if depth >= MAX_INTERRUPT_DEPTH:
        logger.error(
            f"Maximum interrupt depth ({MAX_INTERRUPT_DEPTH}) exceeded, terminating"
        )
        return None

    if stream_depth >= MAX_STREAM_DEPTH:
        logger.error(f"Maximum stream depth ({MAX_STREAM_DEPTH}) exceeded, terminating")
        return None

    logger.info(f"Processing interrupt at depth {depth} (stream_depth: {stream_depth})")

    # Extract interrupt information - format may vary
    interrupt_type = None
    interrupt_payload = {}

    # Debug logging to understand interrupt structure
    logger.debug(f"Interrupt data type: {type(interrupt_data)}")
    logger.debug(f"Interrupt data: {interrupt_data}")

    # Handle LangGraph interrupt format: (Interrupt(value={...}),)
    if isinstance(interrupt_data, tuple) and len(interrupt_data) > 0:
        interrupt_obj = interrupt_data[0]
        logger.debug(f"Interrupt object type: {type(interrupt_obj)}")
        logger.debug(f"Interrupt object: {interrupt_obj}")

        if hasattr(interrupt_obj, "value"):
            interrupt_payload = interrupt_obj.value
            interrupt_type = interrupt_payload.get("type")
            logger.debug(f"Extracted payload: {interrupt_payload}")
            logger.debug(f"Extracted type: {interrupt_type}")
        else:
            logger.warning("Interrupt object has no 'value' attribute")
            interrupt_payload = interrupt_obj
            interrupt_type = (
                interrupt_payload.get("type")
                if isinstance(interrupt_payload, dict)
                else None
            )
    else:
        logger.warning(f"Unexpected interrupt data format: {type(interrupt_data)}")
        # Fallback handling for other formats
        if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
            if hasattr(interrupt_data[0], "value"):
                interrupt_payload = interrupt_data[0].value
                interrupt_type = interrupt_payload.get("type")
            else:
                interrupt_payload = interrupt_data[0]
                interrupt_type = interrupt_payload.get("type")
        elif isinstance(interrupt_data, dict):
            interrupt_payload = interrupt_data
            interrupt_type = interrupt_payload.get("type")

    user_response = None

    # Handle different interrupt types
    if interrupt_type == "agenda_approval":
        user_response = handle_agenda_approval_interrupt(interrupt_payload)
    elif interrupt_type == "periodic_stop":
        user_response = handle_periodic_stop_interrupt(interrupt_payload)
    elif interrupt_type == "topic_continuation":
        user_response = handle_continuation_interrupt(interrupt_payload)
    else:
        logger.warning(f"Unknown interrupt type at depth {depth}: {interrupt_type}")
        return None

    logger.info(
        f"Processed interrupt at depth {depth} (stream_depth: {stream_depth}), type: {interrupt_type}"
    )
    return user_response


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


def handle_agenda_approval_interrupt(
    interrupt_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle agenda approval interrupt from LangGraph.

    Args:
        interrupt_payload: Interrupt data containing agenda and options

    Returns:
        State updates matching LangGraph schema
    """
    proposed_agenda = interrupt_payload.get("proposed_agenda", [])

    try:
        # Use the existing HITL function for agenda approval
        approved_agenda = get_agenda_approval(proposed_agenda)

        if not approved_agenda:
            # User rejected the agenda - return state for rejection
            return {
                "agenda_approved": False,
                "agenda_rejected": True,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                },
            }
        else:
            # User approved (with or without changes) - return state for approval
            return {
                "agenda_approved": True,
                "topic_queue": approved_agenda,
                "final_agenda": approved_agenda,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                },
            }

    except Exception as e:
        logger.error(f"Error in agenda approval interrupt: {e}")
        # Fallback to approval with original agenda
        return {
            "agenda_approved": True,
            "topic_queue": proposed_agenda,
            "final_agenda": proposed_agenda,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "agenda_approval",
            },
        }


def handle_periodic_stop_interrupt(interrupt_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle periodic stop interrupt from LangGraph.

    Args:
        interrupt_payload: Interrupt data containing round and topic info

    Returns:
        State updates matching LangGraph schema
    """
    current_round = interrupt_payload.get("current_round", 0)
    current_topic = interrupt_payload.get("current_topic", "Unknown")

    try:
        console.clear()
        console.print(
            Panel(
                f"[bold yellow]5-Round Checkpoint (Round {current_round})[/bold yellow]",
                subtitle=f"Currently discussing: {current_topic}",
                border_style="yellow",
            )
        )

        options = {
            "c": "[green]Continue[/green] the discussion",
            "e": "[red]End[/red] the current topic",
            "m": "[blue]Modify[/blue] discussion parameters",
            "s": "[orange]Skip[/orange] to final report",
        }

        console.print("Available options:")
        for key, desc in options.items():
            console.print(f"  {key} - {desc}")

        choice = Prompt.ask(
            "\nWhat would you like to do?", choices=["c", "e", "m", "s"], default="c"
        ).lower()

        action_map = {"c": "continue", "e": "end_topic", "m": "modify", "s": "skip"}

        action = action_map.get(choice, "continue")
        console.print(f"[dim]You selected: {action}[/dim]")

        # Return state updates that match the schema
        updates = {
            "user_periodic_decision": action,
            "periodic_stop_counter": interrupt_payload.get("periodic_stop_counter", 0)
            + 1,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "periodic_stop",
            },
        }

        # Handle different user decisions with additional state updates
        if action == "end_topic":
            updates["user_forced_conclusion"] = True
        elif action == "modify":
            updates["user_requested_modification"] = True
        elif action == "skip":
            updates["user_skip_to_final"] = True

        return updates

    except Exception as e:
        logger.error(f"Error in periodic stop interrupt: {e}")
        # Fallback to continue with proper state format
        return {
            "user_periodic_decision": "continue",
            "periodic_stop_counter": interrupt_payload.get("periodic_stop_counter", 0)
            + 1,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "periodic_stop",
            },
        }


def handle_continuation_interrupt(interrupt_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle topic continuation interrupt from LangGraph.

    Args:
        interrupt_payload: Interrupt data containing topic completion info

    Returns:
        State updates matching LangGraph schema
    """
    completed_topic = interrupt_payload.get("completed_topic", "Unknown")
    remaining_topics = interrupt_payload.get("remaining_topics", [])
    agent_recommendation = interrupt_payload.get("agent_recommendation", "continue")

    try:
        # Use the existing HITL function for continuation approval
        decision = get_continuation_approval(completed_topic, remaining_topics)

        # Map the UI response to state updates
        if decision == "y":
            return {
                "user_approves_continuation": True,
                "user_requests_end": False,
                "user_requested_modification": False,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "topic_continuation",
                },
            }
        elif decision == "n":
            return {
                "user_approves_continuation": False,
                "user_requests_end": True,
                "user_requested_modification": False,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "topic_continuation",
                },
            }
        elif decision == "m":
            return {
                "user_approves_continuation": True,
                "user_requests_end": False,
                "user_requested_modification": True,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "topic_continuation",
                },
            }
        else:
            # Fallback based on remaining topics
            continue_session = bool(remaining_topics)
            return {
                "user_approves_continuation": continue_session,
                "user_requests_end": not continue_session,
                "user_requested_modification": False,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "topic_continuation",
                },
            }

    except Exception as e:
        logger.error(f"Error in continuation interrupt: {e}")
        # Fallback based on remaining topics with proper state format
        continue_session = bool(remaining_topics)
        return {
            "user_approves_continuation": continue_session,
            "user_requests_end": not continue_session,
            "user_requested_modification": False,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "topic_continuation",
            },
        }


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

            # Initialize VirtualAgoraFlow v1.3
            console.print("[cyan]Initializing discussion flow v1.3...[/cyan]")
            flow = VirtualAgoraV13Flow(config, enable_monitoring=True)
            flow.compile()
            logger.info("VirtualAgoraV13Flow initialized and compiled")

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
            current_state = flow_state_manager.state
            ui_integration.update_ui_from_state(current_state)

            recovery_manager.create_checkpoint(
                current_state,
                operation="session_created",
                save_to_disk=True,
            )

            # Run the complete discussion workflow
            console.print("[cyan]Starting discussion workflow...[/cyan]")

            # Create spinner but don't use as context manager for interrupt handling
            spinner = LoadingSpinner("Running discussion flow...")
            spinner.__enter__()

            try:
                # Stream the graph execution to get real-time updates
                config_dict = {"configurable": {"thread_id": session_id}}
                logger.info(f"Starting graph stream with config: {config_dict}")
                logger.debug(
                    f"Current state before stream: {flow_state_manager.state.get('session_id')}"
                )

                for update in flow.stream(config_dict):
                    logger.debug(f"Flow update: {update}")

                    # Log which nodes are executing for debugging
                    if isinstance(update, dict):
                        for node_name, node_data in update.items():
                            if node_name not in ["__interrupt__", "__end__"]:
                                logger.info(
                                    f"=== FLOW DEBUG: Node '{node_name}' executed ==="
                                )
                                if isinstance(node_data, dict) and node_data:
                                    logger.info(
                                        f"Node '{node_name}' updates: {list(node_data.keys())}"
                                    )

                    # Handle LangGraph interrupts for HITL interactions
                    if "__interrupt__" in update:
                        # Stop spinner to allow user input
                        spinner.__exit__(None, None, None)

                        interrupt_data = update["__interrupt__"]
                        logger.info(f"Handling interrupt: {interrupt_data}")

                        # Process interrupt and get user response using recursive handler
                        user_response = process_interrupt_recursive(
                            interrupt_data, config_dict, depth=0, stream_depth=0
                        )

                        # Resume execution with user input
                        if user_response:
                            logger.info(
                                f"Resuming execution with user response: {user_response}"
                            )

                            # Update the graph with user input using the proper LangGraph API
                            try:
                                logger.info(
                                    "=== FLOW DEBUG: Starting state update after interrupt ==="
                                )

                                # For LangGraph interrupts, we need to update the state and then continue streaming
                                # The interrupt contains namespace information for resuming
                                if (
                                    isinstance(interrupt_data, tuple)
                                    and len(interrupt_data) > 0
                                ):
                                    interrupt_obj = interrupt_data[0]
                                    logger.info(
                                        f"=== FLOW DEBUG: Interrupt object namespace: {getattr(interrupt_obj, 'ns', 'No ns attribute')}"
                                    )

                                    if (
                                        hasattr(interrupt_obj, "ns")
                                        and interrupt_obj.ns
                                    ):
                                        # Use the namespace from the interrupt for proper resuming
                                        node_name = (
                                            interrupt_obj.ns[0].split(":")[0]
                                            if ":" in interrupt_obj.ns[0]
                                            else interrupt_obj.ns[0]
                                        )
                                        logger.info(
                                            f"=== FLOW DEBUG: Updating state for node: {node_name}"
                                        )
                                        logger.info(
                                            f"=== FLOW DEBUG: User response: {user_response}"
                                        )

                                        flow.compiled_graph.update_state(
                                            config_dict,
                                            user_response,
                                            as_node=node_name,
                                        )
                                        logger.info(
                                            f"=== FLOW DEBUG: State updated for node {node_name} ==="
                                        )
                                    else:
                                        # Fallback: update state without specific node
                                        logger.info(
                                            "=== FLOW DEBUG: Updating state without specific node ==="
                                        )
                                        flow.compiled_graph.update_state(
                                            config_dict, user_response
                                        )
                                        logger.info(
                                            "=== FLOW DEBUG: State updated without node specification ==="
                                        )
                                else:
                                    # Fallback: update state without specific node
                                    logger.info(
                                        "=== FLOW DEBUG: Updating state without specific node (fallback) ==="
                                    )
                                    flow.compiled_graph.update_state(
                                        config_dict, user_response
                                    )
                                    logger.info(
                                        "=== FLOW DEBUG: State updated (fallback) ==="
                                    )

                                logger.info(
                                    "=== FLOW DEBUG: State update complete, execution should continue ==="
                                )
                                logger.info(
                                    "State updated successfully, starting new stream for continuation"
                                )

                                # After update_state(), we need to start a new stream from current state
                                # The original stream is exhausted, so we create a fresh stream
                                logger.info(
                                    "=== FLOW DEBUG: Starting fresh stream after state update ==="
                                )

                                # Restart spinner for continued execution
                                spinner = LoadingSpinner(
                                    "Continuing discussion flow..."
                                )
                                spinner.__enter__()

                                # Start new stream from current state - this will continue from where we left off
                                try:
                                    for continuation_update in flow.stream(config_dict):
                                        logger.info(
                                            f"=== FLOW DEBUG: Continuation update: {continuation_update}"
                                        )

                                        # Log which nodes are executing in continuation
                                        if isinstance(continuation_update, dict):
                                            for (
                                                node_name,
                                                node_data,
                                            ) in continuation_update.items():
                                                if node_name not in [
                                                    "__interrupt__",
                                                    "__end__",
                                                ]:
                                                    logger.info(
                                                        f"=== FLOW DEBUG: Continuation node '{node_name}' executed ==="
                                                    )
                                                    if (
                                                        isinstance(node_data, dict)
                                                        and node_data
                                                    ):
                                                        logger.info(
                                                            f"Continuation node '{node_name}' updates: {list(node_data.keys())}"
                                                        )

                                        # Handle potential nested interrupts
                                        if "__interrupt__" in continuation_update:
                                            logger.info(
                                                "=== FLOW DEBUG: Nested interrupt detected - handling recursively ==="
                                            )
                                            nested_interrupt_data = continuation_update[
                                                "__interrupt__"
                                            ]
                                            logger.info(
                                                f"Processing nested interrupt: {nested_interrupt_data}"
                                            )

                                            # Stop spinner for nested user input
                                            spinner.__exit__(None, None, None)

                                            # Process nested interrupt recursively with increased depth
                                            nested_user_response = (
                                                process_interrupt_recursive(
                                                    nested_interrupt_data,
                                                    config_dict,
                                                    depth=1,
                                                    stream_depth=1,
                                                )
                                            )

                                            if nested_user_response:
                                                logger.info(
                                                    f"=== FLOW DEBUG: Nested interrupt processed, updating state ==="
                                                )

                                                # Update state with nested interrupt response
                                                try:
                                                    if (
                                                        isinstance(
                                                            nested_interrupt_data, tuple
                                                        )
                                                        and len(nested_interrupt_data)
                                                        > 0
                                                    ):
                                                        nested_interrupt_obj = (
                                                            nested_interrupt_data[0]
                                                        )
                                                        if (
                                                            hasattr(
                                                                nested_interrupt_obj,
                                                                "ns",
                                                            )
                                                            and nested_interrupt_obj.ns
                                                        ):
                                                            nested_node_name = (
                                                                nested_interrupt_obj.ns[
                                                                    0
                                                                ].split(":")[0]
                                                                if ":"
                                                                in nested_interrupt_obj.ns[
                                                                    0
                                                                ]
                                                                else nested_interrupt_obj.ns[
                                                                    0
                                                                ]
                                                            )
                                                            flow.compiled_graph.update_state(
                                                                config_dict,
                                                                nested_user_response,
                                                                as_node=nested_node_name,
                                                            )
                                                        else:
                                                            flow.compiled_graph.update_state(
                                                                config_dict,
                                                                nested_user_response,
                                                            )
                                                    else:
                                                        flow.compiled_graph.update_state(
                                                            config_dict,
                                                            nested_user_response,
                                                        )

                                                    logger.info(
                                                        "=== FLOW DEBUG: Nested interrupt state updated successfully ==="
                                                    )

                                                    # Restart spinner and continue with a fresh stream to reach discussion phases
                                                    spinner = LoadingSpinner(
                                                        "Continuing to discussion phases..."
                                                    )
                                                    spinner.__enter__()

                                                    # Start fresh stream recursively to continue to discussion phases
                                                    logger.info(
                                                        "=== FLOW DEBUG: Starting recursive stream after nested interrupt ==="
                                                    )
                                                    try:
                                                        for (
                                                            final_continuation_update
                                                        ) in flow.stream(config_dict):
                                                            logger.info(
                                                                f"=== FLOW DEBUG: Final continuation update: {final_continuation_update}"
                                                            )

                                                            # Log which nodes are executing in final continuation
                                                            if isinstance(
                                                                final_continuation_update,
                                                                dict,
                                                            ):
                                                                for (
                                                                    node_name,
                                                                    node_data,
                                                                ) in (
                                                                    final_continuation_update.items()
                                                                ):
                                                                    if (
                                                                        node_name
                                                                        not in [
                                                                            "__interrupt__",
                                                                            "__end__",
                                                                        ]
                                                                    ):
                                                                        logger.info(
                                                                            f"=== FLOW DEBUG: Final continuation node '{node_name}' executed ==="
                                                                        )
                                                                        if (
                                                                            isinstance(
                                                                                node_data,
                                                                                dict,
                                                                            )
                                                                            and node_data
                                                                        ):
                                                                            logger.info(
                                                                                f"Final continuation node '{node_name}' updates: {list(node_data.keys())}"
                                                                            )

                                                            # Handle any further nested interrupts recursively
                                                            if (
                                                                "__interrupt__"
                                                                in final_continuation_update
                                                            ):
                                                                logger.info(
                                                                    "=== FLOW DEBUG: Additional nested interrupt detected - handling recursively ==="
                                                                )
                                                                additional_nested_interrupt_data = final_continuation_update[
                                                                    "__interrupt__"
                                                                ]
                                                                logger.info(
                                                                    f"Processing additional nested interrupt: {additional_nested_interrupt_data}"
                                                                )

                                                                # Stop spinner for additional nested user input
                                                                spinner.__exit__(
                                                                    None, None, None
                                                                )

                                                                # Process additional nested interrupt recursively with increased depth
                                                                additional_nested_user_response = process_interrupt_recursive(
                                                                    additional_nested_interrupt_data,
                                                                    config_dict,
                                                                    depth=2,
                                                                    stream_depth=2,
                                                                )

                                                                if additional_nested_user_response:
                                                                    logger.info(
                                                                        f"=== FLOW DEBUG: Additional nested interrupt processed ==="
                                                                    )

                                                                    # Update state with additional nested interrupt response
                                                                    try:
                                                                        if (
                                                                            isinstance(
                                                                                additional_nested_interrupt_data,
                                                                                tuple,
                                                                            )
                                                                            and len(
                                                                                additional_nested_interrupt_data
                                                                            )
                                                                            > 0
                                                                        ):
                                                                            additional_nested_interrupt_obj = additional_nested_interrupt_data[
                                                                                0
                                                                            ]
                                                                            if (
                                                                                hasattr(
                                                                                    additional_nested_interrupt_obj,
                                                                                    "ns",
                                                                                )
                                                                                and additional_nested_interrupt_obj.ns
                                                                            ):
                                                                                additional_nested_node_name = (
                                                                                    additional_nested_interrupt_obj.ns[
                                                                                        0
                                                                                    ].split(
                                                                                        ":"
                                                                                    )[
                                                                                        0
                                                                                    ]
                                                                                    if ":"
                                                                                    in additional_nested_interrupt_obj.ns[
                                                                                        0
                                                                                    ]
                                                                                    else additional_nested_interrupt_obj.ns[
                                                                                        0
                                                                                    ]
                                                                                )
                                                                                flow.compiled_graph.update_state(
                                                                                    config_dict,
                                                                                    additional_nested_user_response,
                                                                                    as_node=additional_nested_node_name,
                                                                                )
                                                                            else:
                                                                                flow.compiled_graph.update_state(
                                                                                    config_dict,
                                                                                    additional_nested_user_response,
                                                                                )
                                                                        else:
                                                                            flow.compiled_graph.update_state(
                                                                                config_dict,
                                                                                additional_nested_user_response,
                                                                            )

                                                                        logger.info(
                                                                            "=== FLOW DEBUG: Additional nested interrupt state updated successfully ==="
                                                                        )

                                                                        # Restart spinner and continue
                                                                        spinner = LoadingSpinner(
                                                                            "Continuing after additional nested interrupt..."
                                                                        )
                                                                        spinner.__enter__()

                                                                    except (
                                                                        Exception
                                                                    ) as additional_nested_error:
                                                                        logger.error(
                                                                            f"Error updating state for additional nested interrupt: {additional_nested_error}",
                                                                            exc_info=True,
                                                                        )
                                                                else:
                                                                    logger.warning(
                                                                        "=== FLOW DEBUG: Additional nested interrupt returned no response ==="
                                                                    )
                                                                    # Restart spinner anyway
                                                                    spinner = LoadingSpinner(
                                                                        "Continuing discussion flow..."
                                                                    )
                                                                    spinner.__enter__()

                                                            # Update UI with final continuation steps
                                                            current_state = (
                                                                flow_state_manager.state
                                                            )
                                                            ui_integration.update_ui_from_state(
                                                                current_state
                                                            )

                                                            # Create checkpoints for final continuation phases
                                                            if (
                                                                "current_phase"
                                                                in final_continuation_update
                                                            ):
                                                                recovery_manager.create_checkpoint(
                                                                    current_state,
                                                                    operation=f"final_continuation_phase_{final_continuation_update['current_phase']}",
                                                                    save_to_disk=True,
                                                                )

                                                        logger.info(
                                                            "=== FLOW DEBUG: Final continuation stream completed successfully ==="
                                                        )

                                                    except (
                                                        Exception
                                                    ) as final_continuation_error:
                                                        logger.error(
                                                            f"=== FLOW DEBUG: Error in final continuation stream: {final_continuation_error}",
                                                            exc_info=True,
                                                        )
                                                        spinner.__exit__(
                                                            None, None, None
                                                        )
                                                        raise

                                                except Exception as nested_error:
                                                    logger.error(
                                                        f"Error updating state for nested interrupt: {nested_error}",
                                                        exc_info=True,
                                                    )
                                            else:
                                                logger.warning(
                                                    "=== FLOW DEBUG: Nested interrupt returned no response ==="
                                                )
                                                # Restart spinner anyway
                                                spinner = LoadingSpinner(
                                                    "Continuing discussion flow..."
                                                )
                                                spinner.__enter__()

                                        # Update UI with continuation steps
                                        current_state = flow_state_manager.state
                                        ui_integration.update_ui_from_state(
                                            current_state
                                        )

                                        # Create checkpoints for continuation phases
                                        if "current_phase" in continuation_update:
                                            recovery_manager.create_checkpoint(
                                                current_state,
                                                operation=f"continuation_phase_{continuation_update['current_phase']}",
                                                save_to_disk=True,
                                            )

                                    logger.info(
                                        "=== FLOW DEBUG: Continuation stream completed successfully ==="
                                    )

                                except Exception as continuation_error:
                                    logger.error(
                                        f"=== FLOW DEBUG: Error in continuation stream: {continuation_error}",
                                        exc_info=True,
                                    )
                                    spinner.__exit__(None, None, None)
                                    raise

                            except Exception as e:
                                logger.error(
                                    f"Failed to resume with user input: {e}",
                                    exc_info=True,
                                )
                                logger.error(f"User response was: {user_response}")
                                logger.error(f"Config was: {config_dict}")
                                # Don't continue on error - let the stream naturally complete

                        # After handling interrupt and continuation, break out of the original stream
                        # since we've completed the execution in the continuation stream
                        logger.info(
                            "=== FLOW DEBUG: Breaking from original stream after continuation ==="
                        )
                        break

                    # Update UI with each step (for non-interrupt updates)
                    logger.info(
                        f"=== FLOW DEBUG: Processing non-interrupt update: {list(update.keys()) if isinstance(update, dict) else type(update)}"
                    )
                    current_state = flow_state_manager.state
                    ui_integration.update_ui_from_state(current_state)

                    # Create checkpoints at key phases
                    if "current_phase" in update:
                        recovery_manager.create_checkpoint(
                            current_state,
                            operation=f"phase_{update['current_phase']}",
                            save_to_disk=True,
                        )

                # Stop spinner when execution completes
                spinner.__exit__(None, None, None)
                logger.info("=== FLOW DEBUG: Main stream loop completed naturally ===")
                console.print("[green]Discussion completed successfully![/green]")

                # Get final state and show summary
                final_state = flow_state_manager.state
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
                # Ensure spinner is stopped even on error
                try:
                    spinner.__exit__(None, None, None)
                except:
                    pass

                logger.error(f"Discussion flow error: {e}", exc_info=True)
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error args: {e.args}")
                if hasattr(e, "__traceback__"):
                    import traceback

                    logger.error(
                        f"Full traceback: {traceback.format_exception(type(e), e, e.__traceback__)}"
                    )
                console.print(f"[red]Discussion flow error: {e}[/red]")

                # Attempt recovery
                if recovery_manager.emergency_recovery(flow_state_manager, e):
                    console.print("[yellow]Recovered from discussion error[/yellow]")
                else:
                    console.print("[red]Could not recover from discussion error[/red]")
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
