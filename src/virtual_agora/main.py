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
    get_continuation_approval,
    get_agenda_modifications,
    display_session_status,
)
from virtual_agora.ui.hitl_manager import (
    EnhancedHITLManager,
    HITLApprovalType,
    HITLInteraction,
)
from virtual_agora.ui.preferences import (
    get_preferences_manager,
    get_user_preferences,
)
from virtual_agora.ui.interrupt_handler import setup_interrupt_handlers
from virtual_agora.ui.components import create_header_panel
from virtual_agora.ui.console import get_console
from virtual_agora.ui.theme import get_current_theme, ProviderType
from virtual_agora.ui.accessibility import initialize_accessibility
from virtual_agora.ui.display_modes import initialize_display_manager
from virtual_agora.ui.langgraph_integration import (
    get_ui_integration,
    initialize_ui_integration,
    update_ui_from_state_change,
)
from virtual_agora.flow.interrupt_manager import get_interrupt_manager
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.app_v13 import VirtualAgoraApplicationV13


def process_interrupt_recursive(
    interrupt_data: Any,
    config_dict: Dict[str, Any],
    depth: int = 0,
    stream_depth: int = 0,
) -> Optional[Dict[str, Any]]:
    """Process LangGraph interrupts with proper context management.

    Args:
        interrupt_data: The interrupt data from LangGraph
        config_dict: Configuration dictionary for LangGraph operations
        depth: Legacy parameter (now managed by InterruptStackManager)
        stream_depth: Legacy parameter (now managed by InterruptStackManager)

    Returns:
        User response dictionary or None if processing fails
    """
    # Get interrupt manager for proper context management
    interrupt_manager = get_interrupt_manager()

    # Safety limits to prevent infinite recursion
    MAX_INTERRUPT_DEPTH = 5

    # Use managed depth from interrupt manager instead of parameters
    current_depth = interrupt_manager.get_current_depth()

    # Validate interrupt stack integrity before processing
    integrity_warnings = interrupt_manager.validate_stack_integrity()
    if integrity_warnings:
        logger.warning(
            f"Interrupt stack integrity issues detected: {integrity_warnings}"
        )
        # Consider emergency recovery if there are serious issues
        critical_issues = [
            w
            for w in integrity_warnings
            if "mismatch" in w.lower() or "excessive" in w.lower()
        ]
        if critical_issues:
            logger.error(f"Critical interrupt stack issues detected: {critical_issues}")
            recovery_info = interrupt_manager.emergency_recovery()
            logger.info(
                f"Emergency recovery performed: {recovery_info['recovered_successfully']}"
            )
            # Recalculate depth after recovery
            current_depth = interrupt_manager.get_current_depth()

    if current_depth >= MAX_INTERRUPT_DEPTH:
        logger.error(
            f"Maximum interrupt depth ({MAX_INTERRUPT_DEPTH}) exceeded, terminating"
        )
        logger.error(f"Stack info: {interrupt_manager.get_stack_info()}")
        # Attempt emergency recovery before giving up
        logger.info("Attempting emergency recovery due to depth limit")
        recovery_info = interrupt_manager.emergency_recovery()
        if recovery_info["recovered_successfully"]:
            current_depth = interrupt_manager.get_current_depth()
            logger.info(f"Recovery successful, new depth: {current_depth}")
            if current_depth >= MAX_INTERRUPT_DEPTH:
                return None
        else:
            return None

    # Log interrupt processing with managed depth
    logger.info(f"Processing interrupt at managed depth {current_depth}")

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

    # Push interrupt context to stack for proper management
    topic_name = interrupt_payload.get("completed_topic") or interrupt_payload.get(
        "topic"
    )
    interrupt_context = interrupt_manager.push_interrupt(
        interrupt_type=interrupt_type, payload=interrupt_payload, topic_name=topic_name
    )

    user_response = None

    try:
        # Handle different interrupt types
        if interrupt_type == "agenda_approval":
            user_response = handle_agenda_approval_interrupt(interrupt_payload)
        elif interrupt_type == "periodic_stop":
            user_response = handle_periodic_stop_interrupt(interrupt_payload)
        elif interrupt_type == "topic_continuation":
            user_response = handle_continuation_interrupt(interrupt_payload)
        else:
            logger.warning(
                f"Unknown interrupt type at managed depth {current_depth}: {interrupt_type}"
            )
            return None

        logger.info(
            f"Processed interrupt {interrupt_context.interrupt_id} at depth {current_depth}, type: {interrupt_type}"
        )

    finally:
        # Always pop interrupt context for cleanup
        try:
            popped_context = interrupt_manager.pop_interrupt()
            if (
                popped_context
                and popped_context.interrupt_id != interrupt_context.interrupt_id
            ):
                logger.error(
                    f"Interrupt stack corruption detected: expected {interrupt_context.interrupt_id}, got {popped_context.interrupt_id if popped_context else None}"
                )
                logger.error("This indicates serious interrupt stack corruption")
                # Perform emergency recovery
                recovery_info = interrupt_manager.emergency_recovery()
                logger.error(
                    f"Emergency recovery attempted: {recovery_info['recovered_successfully']}"
                )
                if not recovery_info["recovered_successfully"]:
                    logger.error(
                        "Emergency recovery failed - interrupt system may be unstable"
                    )
        except Exception as cleanup_error:
            logger.error(f"Error during interrupt cleanup: {cleanup_error}")
            logger.error("Attempting emergency recovery due to cleanup failure")
            try:
                recovery_info = interrupt_manager.emergency_recovery()
                logger.error(
                    f"Emergency recovery after cleanup error: {recovery_info['recovered_successfully']}"
                )
            except Exception as recovery_error:
                logger.error(f"Emergency recovery also failed: {recovery_error}")
                logger.error("Interrupt system may be in critical state")

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
    """Handle agenda approval interrupt from LangGraph using v1.3 HITLManager.

    Args:
        interrupt_payload: Interrupt data containing agenda and options

    Returns:
        State updates matching LangGraph schema
    """
    proposed_agenda = interrupt_payload.get("proposed_agenda", [])

    try:
        # Initialize HITLManager with console
        hitl_manager = EnhancedHITLManager(console)

        # Create HITLInteraction for agenda approval
        interaction = HITLInteraction(
            approval_type=HITLApprovalType.AGENDA_APPROVAL,
            prompt_message="Please review and approve the proposed discussion agenda.",
            context={"proposed_agenda": proposed_agenda},
        )

        # Use v1.3 HITLManager instead of v1.2 function
        response = hitl_manager.process_interaction(interaction)

        # Process HITLResponse and convert to LangGraph state updates
        if not response.approved:
            # User rejected the agenda
            return {
                "agenda_approved": False,
                "agenda_rejected": True,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                },
            }
        else:
            # User approved (with or without changes)
            final_agenda = (
                response.modified_data if response.modified_data else proposed_agenda
            )
            return {
                "agenda_approved": True,
                "topic_queue": final_agenda,
                "final_agenda": final_agenda,
                "hitl_state": {
                    "awaiting_approval": False,
                    "approval_type": "agenda_approval",
                    "approval_action": response.action,
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
    checkpoint_interval = interrupt_payload.get("checkpoint_interval", 3)

    try:
        console.clear()
        console.print(
            Panel(
                f"[bold yellow]{checkpoint_interval}-Round Checkpoint (Round {current_round})[/bold yellow]",
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
        "--debug",
        nargs="?",
        const="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Enable debug mode with optional log level (default: DEBUG)",
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

    parser.add_argument(
        "--hide-messages",
        action="store_true",
        help="Hide atmospheric UI messages during execution for debugging",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=3,
        help="Number of rounds between periodic user checkpoints (default: 3, min: 1, max: 20)",
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
        log_level = args.debug if args.debug else "WARNING"
        setup_logging(level=log_level)
        logger.info(f"Starting Virtual Agora v{__version__}")

        # Validate checkpoint interval
        if args.checkpoint_interval < 1 or args.checkpoint_interval > 20:
            raise ConfigurationError(
                f"Checkpoint interval must be between 1 and 20, got: {args.checkpoint_interval}"
            )
        logger.info(f"Using checkpoint interval: {args.checkpoint_interval} rounds")

        # Initialize display manager with hide_messages setting
        initialize_display_manager(hide_messages=args.hide_messages)
        if args.hide_messages:
            logger.info("Atmospheric UI messages hidden for debugging")

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
                current_log_level = args.debug if args.debug else "WARNING"
                if env_config.log_level != current_log_level:
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
            flow = VirtualAgoraV13Flow(
                config,
                enable_monitoring=True,
                checkpoint_interval=args.checkpoint_interval,
            )
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

            # Run the complete discussion workflow using clean architecture
            console.print("[cyan]Starting discussion workflow...[/cyan]")

            # Initialize the clean execution architecture
            from virtual_agora.execution import (
                SessionController,
                StreamCoordinator,
                UnifiedStateManager,
                ExecutionTracker,
            )

            # Create unified state manager
            unified_state_manager = UnifiedStateManager(session_id)

            # Initialize session state from existing flow state
            flow_state_dict = flow_state_manager.state
            unified_state_manager.update_flow_state(flow_state_dict)
            unified_state_manager.update_session_state(
                {
                    "main_topic": topic,
                    "current_phase": flow_state_dict.get("current_phase", 0),
                }
            )

            # Create session controller (single source of truth)
            session_controller = SessionController(flow, session_id)

            # Create stream coordinator (handles lifecycle without breaks)
            stream_coordinator = StreamCoordinator(flow, process_interrupt_recursive)

            # Create execution tracker (provides visibility)
            execution_tracker = ExecutionTracker(session_id)

            logger.info("Clean execution architecture initialized")

            # Status message instead of spinner (prevents Rich prompt interference)
            console.print("[cyan]Running discussion flow...[/cyan]")

            try:
                # Execute session using clean architecture
                config_dict = {"configurable": {"thread_id": session_id}}
                logger.info(
                    f"Starting clean session execution with config: {config_dict}"
                )

                # Use the stream coordinator instead of direct flow.stream()
                for update in stream_coordinator.coordinate_stream_execution(
                    config_dict
                ):
                    logger.debug(
                        f"Clean flow update: {list(update.keys()) if isinstance(update, dict) else type(update)}"
                    )

                    # Update unified state manager
                    if isinstance(update, dict):
                        # Extract state updates from the flow update
                        state_updates = {}
                        for node_name, node_data in update.items():
                            if node_name not in [
                                "__interrupt__",
                                "__end__",
                            ] and isinstance(node_data, dict):
                                state_updates.update(node_data)
                                # Track node execution
                                execution_tracker.track_node_execution(
                                    node_name,
                                    unified_state_manager.get_legacy_state(),
                                    state_updates,
                                )

                        if state_updates:
                            unified_state_manager.update_flow_state(state_updates)

                    # Handle interrupts through the clean architecture
                    if "__interrupt__" in update:
                        # No spinner to stop (prevents Rich prompt interference)

                        interrupt_start = datetime.now()
                        interrupt_data = update["__interrupt__"]
                        logger.info(
                            f"Handling interrupt via clean architecture: {interrupt_data}"
                        )

                        # Extract interrupt type for tracking
                        interrupt_type = "unknown"
                        if (
                            isinstance(interrupt_data, tuple)
                            and len(interrupt_data) > 0
                        ):
                            interrupt_obj = interrupt_data[0]
                            if hasattr(interrupt_obj, "value") and isinstance(
                                interrupt_obj.value, dict
                            ):
                                interrupt_type = interrupt_obj.value.get(
                                    "type", "unknown"
                                )

                        # The StreamCoordinator handles the interrupt processing
                        # We just need to track it and restart the spinner
                        interrupt_end = datetime.now()
                        execution_tracker.track_interrupt_handling(
                            interrupt_type, interrupt_start, interrupt_end
                        )

                        # Status message for continued execution (prevents Rich prompt interference)
                        console.print("[cyan]Continuing discussion flow...[/cyan]")

                    # Handle natural completion
                    elif "__end__" in update:
                        logger.info(
                            "Session completed naturally via clean architecture"
                        )
                        break

                    # Update UI with current state
                    current_state = unified_state_manager.get_legacy_state()
                    ui_integration.update_ui_from_state(current_state)

                    # Create checkpoints at key phases
                    if isinstance(update, dict):
                        for node_name, node_data in update.items():
                            if (
                                isinstance(node_data, dict)
                                and "current_phase" in node_data
                            ):
                                recovery_manager.create_checkpoint(
                                    current_state,
                                    operation=f"phase_{node_data['current_phase']}",
                                    save_to_disk=True,
                                )
                                execution_tracker.track_phase_transition(
                                    current_state.get("current_phase", 0),
                                    node_data["current_phase"],
                                    {"node": node_name, "update": node_data},
                                )

                # Execution completed (no spinner to stop)
                logger.info(
                    "Clean architecture session execution completed successfully"
                )
                console.print("[green]Discussion completed successfully![/green]")

                # Get final state and statistics from unified state manager
                final_unified_state = unified_state_manager.get_unified_state()
                final_statistics = unified_state_manager.statistics
                execution_report = execution_tracker.get_performance_report()

                # LOG: Debug session completion with clean architecture
                logger.info(f"=== CLEAN ARCHITECTURE SESSION COMPLETION ===")
                logger.info(f"Session completed successfully using clean architecture")
                logger.info(f"Session ID: {session_id}")
                logger.info(f"Topics completed: {final_statistics.topics_discussed}")
                logger.info(f"Total messages: {final_statistics.total_messages}")
                logger.info(
                    f"Session duration: {final_statistics.session_duration:.1f}s"
                )
                logger.info(
                    f"Execution events tracked: {execution_report['total_events']}"
                )

                # Validate state consistency
                validation_errors = unified_state_manager.validate_state_consistency()
                if validation_errors:
                    logger.warning(
                        f"State validation found {len(validation_errors)} issues:"
                    )
                    for error in validation_errors:
                        logger.warning(f"  - {error}")
                else:
                    logger.info("State validation passed - all state layers consistent")

                # Display accurate statistics
                console.print(f"[dim]Final session ID: {session_id}[/dim]")
                console.print(
                    f"[dim]Topics discussed: {final_statistics.topics_discussed}[/dim]"
                )
                console.print(
                    f"[dim]Total messages: {final_statistics.total_messages}[/dim]"
                )
                console.print(
                    f"[dim]Session duration: {final_statistics.session_duration:.1f}s[/dim]"
                )
                console.print(
                    f"[dim]Execution events: {execution_report['total_events']}[/dim]"
                )

            except Exception as e:
                # No spinner to stop on error

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
