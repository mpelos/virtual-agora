"""Virtual Agora Application class with v1.3 HITL integration.

This module provides the main application class that integrates all v1.3
components including the enhanced HITL system.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.live import Live

from virtual_agora.config.models import Config as VirtualAgoraConfig
from virtual_agora.ui.hitl_manager import EnhancedHITLManager
from virtual_agora.ui.hitl_state import (
    HITLInteraction,
    HITLContext,
    HITLApprovalType,
)
from virtual_agora.ui.components import EnhancedDashboard
from virtual_agora.ui.session_control import SessionController, CheckpointManager
from virtual_agora.ui.console import get_console
from virtual_agora.ui.human_in_the_loop import get_initial_topic
from virtual_agora.ui.assembly_dashboard import get_assembly_dashboard, AssemblyPhase
from virtual_agora.ui.display_modes import is_assembly_mode, get_display_manager
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class VirtualAgoraApplicationV13:
    """Main application class with v1.3 enhancements."""

    def __init__(self, config: VirtualAgoraConfig):
        """Initialize the v1.3 application.

        Args:
            config: Virtual Agora configuration
        """
        self.config = config
        self.console = get_console().rich_console
        self.hitl_manager = EnhancedHITLManager(self.console)
        self.dashboard = EnhancedDashboard(self.console)
        self.session_controller = SessionController(self.console)
        self.checkpoint_manager = CheckpointManager()

        # Initialize v1.3 flow
        self.flow = VirtualAgoraV13Flow(config, enable_monitoring=True)
        self.state_manager = self.flow.get_state_manager()
        self.recovery_manager = StateRecoveryManager()

        # Initialize specialized agents from flow
        self.specialized_agents = self.flow.specialized_agents
        self.discussing_agents = self.flow.discussing_agents

        # Set up interrupt callback
        self.session_controller.set_interrupt_callback(self._handle_interrupt)

        self.session_id = None
        self.is_running = False
        self.assembly_dashboard = get_assembly_dashboard()

    def _handle_interrupt(self, interrupt_data: Dict[str, Any]) -> None:
        """Handle interrupt from session controller."""
        action = interrupt_data.get("action")

        if action == "end_topic":
            # Force topic conclusion
            state_updates = {
                "user_forced_conclusion": True,
                "force_reason": interrupt_data.get("reason", "User interrupt"),
            }
            self.state_manager.update_state(state_updates)

        elif action == "skip_to_report":
            # Skip remaining topics
            state_updates = {
                "topic_queue": [],
                "skip_to_report": True,
            }
            self.state_manager.update_state(state_updates)

        elif action == "pause":
            # Save checkpoint
            checkpoint = interrupt_data.get("checkpoint", {})
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                self.session_id, self.state_manager.state, metadata=checkpoint
            )
            logger.info(f"Saved checkpoint: {checkpoint_id}")

    def handle_hitl_gate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HITL gates with enhanced UI.

        This method is called by graph nodes when HITL interaction is needed.
        """
        hitl_state = state.get("hitl_state", {})

        if not hitl_state.get("awaiting_approval"):
            return {}

        # Create context from state
        context = HITLContext(
            current_round=state.get("current_round"),
            active_topic=state.get("active_topic"),
            completed_topics=state.get("completed_topics", []),
            remaining_topics=state.get("topic_queue", []),
            proposed_agenda=state.get("proposed_agenda", []),
            session_stats={
                "total_messages": len(state.get("messages", [])),
                "topics_completed": len(state.get("completed_topics", [])),
                "session_duration": self._format_duration(state.get("start_time")),
            },
            agent_vote_result=state.get("conclusion_vote"),
        )

        # Create interaction object
        interaction = HITLInteraction(
            approval_type=HITLApprovalType(hitl_state["approval_type"]),
            prompt_message=hitl_state.get("prompt_message", ""),
            options=hitl_state.get("options"),
            context=context.to_dict(),
        )

        # Check for periodic stop
        if (
            interaction.approval_type == HITLApprovalType.PERIODIC_STOP
            and self.session_controller.check_periodic_control(
                state.get("current_round", 0)
            )
        ):
            self.session_controller.display_checkpoint_notification(
                state.get("current_round", 0),
                state.get("active_topic", "Unknown"),
                state,
            )

        # Process through HITL manager
        response = self.hitl_manager.process_interaction(interaction)

        # Update state based on response
        return self._process_hitl_response(interaction.approval_type, response, state)

    def _process_hitl_response(
        self, approval_type: HITLApprovalType, response, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process HITL response and return state updates."""

        state_updates = {"hitl_state": {"awaiting_approval": False}}

        if approval_type == HITLApprovalType.PERIODIC_STOP:
            if response.action == "force_topic_end":
                state_updates.update(
                    {
                        "user_forced_conclusion": True,
                        "force_reason": response.reason,
                    }
                )
            elif response.action == "skip_to_report":
                state_updates.update(
                    {
                        "topic_queue": [],
                        "skip_to_report": True,
                    }
                )

        elif approval_type == HITLApprovalType.AGENDA_APPROVAL:
            if response.approved:
                state_updates.update(
                    {
                        "agenda": {
                            "topics": response.modified_data,
                            "current_topic_index": 0,
                            "completed_topics": [],
                        },
                        "topic_queue": response.modified_data,
                        "agenda_approved": True,
                        "agenda_edited": response.action == "edited",
                    }
                )
            else:
                state_updates["agenda_approved"] = False

        elif approval_type == HITLApprovalType.TOPIC_CONTINUATION:
            if response.action == "continue":
                state_updates["continue_session"] = True
            elif response.action == "modify_agenda":
                state_updates.update(
                    {
                        "continue_session": True,
                        "user_requested_modification": True,
                    }
                )
            else:
                state_updates["continue_session"] = False

        elif approval_type == HITLApprovalType.SESSION_CONTINUATION:
            state_updates["user_approves_continuation"] = (
                response.action == "continue_session"
            )

        elif approval_type == HITLApprovalType.FINAL_REPORT_APPROVAL:
            state_updates["generate_final_report"] = response.approved

        # Add response to HITL history
        if "hitl_state" not in state_updates:
            state_updates["hitl_state"] = {}

        state_updates["hitl_state"]["approval_history"] = state.get(
            "hitl_state", {}
        ).get("approval_history", []) + [
            {
                "type": approval_type.value,
                "result": response.action,
                "timestamp": datetime.now(),
                "approved": response.approved,
                "reason": response.reason,
            }
        ]

        return state_updates

    def _format_duration(self, start_time: Optional[datetime]) -> str:
        """Format session duration."""
        if not start_time:
            return "Unknown"

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        duration = datetime.now() - start_time
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def show_agent_invocation(
        self, agent_type: str, task: str, context: Dict[str, Any]
    ) -> None:
        """Display agent invocation in UI."""
        self.dashboard.agent_display.show_agent_invocation(
            agent_type, task, context, status="active"
        )

    def show_agent_result(
        self, agent_type: str, result: str, execution_time: float
    ) -> None:
        """Display agent result in UI."""
        self.dashboard.agent_display.show_agent_result(
            agent_type, result, execution_time
        )

    def _initialize_assembly_dashboard(self, session_id: str, topic: str) -> None:
        """Initialize the assembly dashboard with session information.

        Args:
            session_id: Session identifier
            topic: Main discussion topic
        """
        # Extract participant info from flow
        participants = []
        for agent in self.flow.discussing_agents:
            # Get provider type from agent ID
            from virtual_agora.flow.nodes_v13 import get_provider_type_from_agent_id

            provider = get_provider_type_from_agent_id(agent.agent_id)

            participants.append({"agent_id": agent.agent_id, "provider": provider})

        # Get agenda items if available (we'll start with empty and update later)
        agenda_items = []
        if hasattr(self, "state_manager") and self.state_manager.state:
            agenda_items = self.state_manager.state.get("topic_queue", [])

        # Initialize the assembly dashboard
        self.assembly_dashboard.initialize_session(
            session_id=session_id,
            main_topic=topic,
            participants=participants,
            agenda_items=agenda_items,
        )

        # Set initial phase
        self.assembly_dashboard.update_phase(AssemblyPhase.INITIALIZATION)

        # Display the session header
        self.assembly_dashboard.display_session_header()

        logger.info(f"Assembly dashboard initialized for session {session_id}")

    def update_dashboard(self) -> None:
        """Update the dashboard with current state."""
        if hasattr(self, "state_manager"):
            # Update traditional dashboard
            self.dashboard.update_session_info(self.state_manager.state)

            # Update assembly dashboard if in assembly mode
            if is_assembly_mode() and hasattr(self, "assembly_dashboard"):
                self._update_assembly_dashboard()

    def _update_assembly_dashboard(self) -> None:
        """Update the assembly dashboard with current state information."""
        if not hasattr(self, "state_manager") or not self.state_manager.state:
            return

        state = self.state_manager.state

        # Update phase based on current phase
        current_phase_str = state.get("current_phase", "")
        if "agenda_proposal" in current_phase_str.lower():
            phase = AssemblyPhase.TOPIC_PROPOSAL
        elif "agenda_voting" in current_phase_str.lower():
            phase = AssemblyPhase.AGENDA_VOTING
        elif "discussion" in current_phase_str.lower():
            phase = AssemblyPhase.DISCUSSION
        elif "voting" in current_phase_str.lower():
            phase = AssemblyPhase.VOTING
        else:
            phase = AssemblyPhase.INITIALIZATION

        # Update phase if it has changed
        if (
            hasattr(self.assembly_dashboard, "session_info")
            and self.assembly_dashboard.session_info
        ):
            if self.assembly_dashboard.session_info.current_phase != phase:
                self.assembly_dashboard.update_phase(phase)

        # Update agenda progress
        current_topic = state.get("active_topic")
        if current_topic:
            completed_count = len(state.get("completed_topics", []))
            self.assembly_dashboard.update_agenda_progress(
                current_topic, completed_count
            )

        # Update current round if available
        current_round = state.get("current_round", 0)
        if (
            hasattr(self.assembly_dashboard, "session_info")
            and self.assembly_dashboard.session_info
        ):
            self.assembly_dashboard.session_info.current_round = current_round

    def run_session(self, topic: Optional[str] = None) -> int:
        """Run a complete Virtual Agora session.

        Args:
            topic: Optional pre-specified topic (otherwise will prompt)

        Returns:
            Exit code (0 for success)
        """
        try:
            # Welcome message
            self.console.clear()
            self.console.print(
                "[bold cyan]Welcome to Virtual Agora v1.3[/bold cyan]\n"
                "[dim]Enhanced with specialized agents and improved HITL controls[/dim]\n"
            )

            # Get topic if not provided
            if not topic:
                topic_data = get_initial_topic()
                topic = topic_data["topic"]
                user_defines_topics = topic_data["user_defines_topics"]
            else:
                user_defines_topics = False  # Default when topic provided externally

            # Create session
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_id = self.flow.create_session(
                session_id=self.session_id,
                main_topic=topic,
                user_defines_topics=user_defines_topics,
            )

            logger.info(f"Created v1.3 session: {session_id}")
            self.is_running = True

            # Initialize assembly dashboard if in assembly mode
            if is_assembly_mode():
                self._initialize_assembly_dashboard(session_id, topic)

            # Show initial dashboard
            self.update_dashboard()

            # Run the flow with live dashboard updates
            self.console.print("[cyan]Initializing discussion flow...[/cyan]")
            self.flow.compile()

            self.console.print("[green]Flow initialized successfully![/green]\n")

            # Stream execution with dashboard updates
            config_dict = {"configurable": {"thread_id": session_id}}

            with Live(self.dashboard.render(), refresh_per_second=2) as live:
                try:
                    for update in self.flow.stream(config_dict):
                        logger.debug(f"Flow update: {update}")

                        # Update dashboard
                        self.update_dashboard()
                        live.update(self.dashboard.render())

                        # Handle HITL gates
                        current_state = self.state_manager.state
                        if current_state.get("hitl_state", {}).get("awaiting_approval"):
                            # Exit live display for HITL interaction
                            live.stop()

                            # Handle HITL
                            hitl_updates = self.handle_hitl_gate(current_state)
                            if hitl_updates:
                                self.state_manager.update_state(hitl_updates)

                            # Resume live display
                            live.start()

                        # Create checkpoints at phase transitions
                        if "current_phase" in update:
                            self.checkpoint_manager.save_checkpoint(
                                session_id,
                                current_state,
                                metadata={
                                    "phase": update["current_phase"],
                                    "operation": f"phase_{update['current_phase']}",
                                },
                            )

                except Exception as e:
                    logger.error(f"Discussion flow error: {e}")
                    live.stop()
                    self.console.print(f"[red]Error during discussion: {e}[/red]")

                    # Attempt recovery
                    if self.recovery_manager.emergency_recovery(self.state_manager, e):
                        self.console.print("[yellow]Recovered from error[/yellow]")
                        live.start()
                    else:
                        raise

            # Show final summary
            self._show_final_summary()

            return 0

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Session interrupted by user[/yellow]")
            return 1

        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            self.console.print(f"\n[red]Session error: {e}[/red]")
            return 1

        finally:
            self.is_running = False

    def _show_final_summary(self) -> None:
        """Show final session summary."""
        state = self.state_manager.state

        self.console.print("\n" + "=" * 60 + "\n")
        self.console.print("[bold green]Session Complete![/bold green]\n")

        # Session stats
        stats = {
            "Session ID": state.get("session_id", "Unknown"),
            "Duration": self._format_duration(state.get("start_time")),
            "Topics Discussed": len(state.get("completed_topics", [])),
            "Total Messages": len(state.get("messages", [])),
            "Checkpoints": len(self.session_controller.checkpoint_history),
            "HITL Interactions": len(self.hitl_manager.state_tracker.approval_history),
        }

        for key, value in stats.items():
            self.console.print(f"[bold]{key}:[/bold] {value}")

        # Topics summary
        if state.get("completed_topics"):
            self.console.print("\n[bold]Topics Discussed:[/bold]")
            for i, topic in enumerate(state["completed_topics"], 1):
                self.console.print(f"  {i}. {topic}")

        # Reports generated
        if state.get("final_report_directory"):
            self.console.print(
                f"\n[bold]Final Report:[/bold] {state['final_report_directory']}"
            )

        self.console.print("\n" + "=" * 60)

    def resume_session(self, checkpoint_id: str) -> int:
        """Resume a session from checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to resume from

        Returns:
            Exit code
        """
        try:
            # Load checkpoint
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)

            self.console.print(
                f"[cyan]Resuming session from checkpoint {checkpoint_id}...[/cyan]"
            )

            # Restore state
            self.state_manager.state = checkpoint_data["state"]
            self.session_id = checkpoint_data["session_id"]

            # Resume flow execution
            return self.run_session()

        except Exception as e:
            logger.error(f"Failed to resume session: {e}")
            self.console.print(f"[red]Failed to resume session: {e}[/red]")
            return 1
