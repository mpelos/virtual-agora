# Phase 4: HITL & UI Enhancement Guide

## Purpose
This document provides detailed instructions for implementing the enhanced Human-in-the-Loop (HITL) gates and updating the UI components to support the v1.3 node-centric architecture. The new design introduces periodic user control points and more granular session management.

## Prerequisites
- Completed Phases 1-3 (Config, Agents, Graph)
- Understanding of current HITL implementation
- Familiarity with Rich library for terminal UI
- Access to v1.3 HITL specifications

## Enhanced HITL Requirements

### v1.1 HITL Gates (Current)
1. Initial theme input
2. Agenda approval
3. Topic continuation approval
4. Final report trigger

### v1.3 HITL Gates (Target)
1. Initial theme input
2. Agenda approval with editing
3. **NEW: Periodic 5-round stops**
4. **NEW: Topic conclusion override**
5. Topic continuation permission
6. **NEW: Agent poll override**
7. Session continuation approval
8. Final report generation approval

## HITL State Management Updates

### Step 1: Enhance HITL State Structure

```python
# src/virtual_agora/ui/hitl_state.py

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

class HITLApprovalType(Enum):
    """All HITL interaction types in v1.3."""
    
    # Existing types
    THEME_INPUT = "theme_input"
    AGENDA_APPROVAL = "agenda_approval"
    
    # New v1.3 types
    PERIODIC_STOP = "periodic_stop"  # Every 5 rounds
    TOPIC_OVERRIDE = "topic_override"  # Force topic end
    TOPIC_CONTINUATION = "topic_continuation"
    AGENT_POLL_OVERRIDE = "agent_poll_override"
    SESSION_CONTINUATION = "session_continuation"
    FINAL_REPORT_APPROVAL = "final_report_approval"

class HITLInteraction:
    """Represents a single HITL interaction."""
    
    def __init__(
        self,
        approval_type: HITLApprovalType,
        prompt_message: str,
        options: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ):
        self.approval_type = approval_type
        self.prompt_message = prompt_message
        self.options = options or []
        self.context = context or {}
        self.timeout_seconds = timeout_seconds
        self.timestamp = datetime.now()
        self.response = None
        self.response_time = None
    
    def record_response(self, response: Any) -> None:
        """Record user response with timing."""
        self.response = response
        self.response_time = datetime.now()
        self.duration_seconds = (
            self.response_time - self.timestamp
        ).total_seconds()
```

### Step 2: Implement Enhanced HITL Manager

```python
# src/virtual_agora/ui/hitl_manager.py

from typing import Dict, Any, Optional, Callable
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

class EnhancedHITLManager:
    """Manages all HITL interactions for v1.3."""
    
    def __init__(self, console: Console):
        self.console = console
        self.interaction_history = []
        self.handlers = self._register_handlers()
    
    def _register_handlers(self) -> Dict[HITLApprovalType, Callable]:
        """Register specific handlers for each HITL type."""
        return {
            HITLApprovalType.THEME_INPUT: self._handle_theme_input,
            HITLApprovalType.AGENDA_APPROVAL: self._handle_agenda_approval,
            HITLApprovalType.PERIODIC_STOP: self._handle_periodic_stop,
            HITLApprovalType.TOPIC_OVERRIDE: self._handle_topic_override,
            # ... register all handlers
        }
    
    def process_interaction(
        self, 
        interaction: HITLInteraction
    ) -> Dict[str, Any]:
        """Process a HITL interaction and return response."""
        
        # Display context if available
        if interaction.context:
            self._display_context(interaction.context)
        
        # Get appropriate handler
        handler = self.handlers.get(
            interaction.approval_type,
            self._default_handler
        )
        
        # Process interaction
        response = handler(interaction)
        
        # Record response
        interaction.record_response(response)
        self.interaction_history.append(interaction)
        
        return response
    
    def _handle_periodic_stop(
        self, 
        interaction: HITLInteraction
    ) -> Dict[str, Any]:
        """Handle 5-round periodic stop check.
        
        New in v1.3 - gives user periodic control.
        """
        # Display round information
        round_num = interaction.context.get("current_round", 0)
        topic = interaction.context.get("active_topic", "Unknown")
        
        panel = Panel(
            f"[bold yellow]Round {round_num} Checkpoint[/bold yellow]\n\n"
            f"Current topic: [cyan]{topic}[/cyan]\n"
            f"You've reached a 5-round checkpoint.\n\n"
            f"{interaction.prompt_message}",
            title="🛑 Periodic Stop",
            border_style="yellow"
        )
        self.console.print(panel)
        
        # Get user decision
        force_end = Confirm.ask(
            "Do you want to end discussion of this topic now?",
            default=False
        )
        
        if force_end:
            # Ask for reason (optional)
            reason = Prompt.ask(
                "Reason for ending (optional)",
                default="User decision at periodic checkpoint"
            )
            
            return {
                "force_topic_end": True,
                "reason": reason,
                "checkpoint_round": round_num
            }
        
        return {
            "force_topic_end": False,
            "continue_discussion": True
        }
    
    def _handle_agenda_approval(
        self, 
        interaction: HITLInteraction
    ) -> Dict[str, Any]:
        """Handle agenda approval with editing capability.
        
        Enhanced in v1.3 to support inline editing.
        """
        agenda = interaction.context.get("proposed_agenda", [])
        
        # Display proposed agenda
        table = Table(title="Proposed Discussion Agenda")
        table.add_column("Order", style="cyan", width=6)
        table.add_column("Topic", style="white")
        
        for i, topic in enumerate(agenda, 1):
            table.add_row(str(i), topic)
        
        self.console.print(table)
        
        # Ask for approval
        choices = ["Approve", "Edit", "Reorder", "Cancel"]
        action = Prompt.ask(
            "What would you like to do?",
            choices=choices,
            default="Approve"
        )
        
        if action == "Approve":
            return {"approved": True, "agenda": agenda}
        
        elif action == "Edit":
            return self._edit_agenda(agenda)
        
        elif action == "Reorder":
            return self._reorder_agenda(agenda)
        
        else:  # Cancel
            return {"approved": False, "cancelled": True}
    
    def _edit_agenda(self, agenda: List[str]) -> Dict[str, Any]:
        """Allow user to edit agenda items."""
        edited_agenda = []
        
        self.console.print("\n[yellow]Edit each topic (Enter to keep):[/yellow]")
        
        for i, topic in enumerate(agenda, 1):
            new_topic = Prompt.ask(
                f"{i}. {topic}",
                default=topic
            )
            if new_topic.strip():  # Skip empty entries
                edited_agenda.append(new_topic)
        
        # Option to add new topics
        while True:
            new_topic = Prompt.ask(
                "Add new topic (Enter to finish)",
                default=""
            )
            if not new_topic:
                break
            edited_agenda.append(new_topic)
        
        return {
            "approved": True,
            "agenda": edited_agenda,
            "edited": True
        }
```

### Step 3: UI Component Updates

```python
# src/virtual_agora/ui/components.py

from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, Any

class SpecializedAgentDisplay:
    """Display component for specialized agent activities."""
    
    def __init__(self, console: Console):
        self.console = console
        self.agent_colors = {
            "moderator": "cyan",
            "summarizer": "magenta", 
            "topic_report": "green",
            "ecclesia_report": "blue"
        }
    
    def show_agent_invocation(
        self,
        agent_type: str,
        task: str,
        context: Dict[str, Any]
    ) -> None:
        """Display when a specialized agent is invoked."""
        color = self.agent_colors.get(agent_type, "white")
        
        self.console.print(
            f"\n[{color}]🤖 {agent_type.title()} Agent Activated[/{color}]"
        )
        self.console.print(f"   Task: {task}")
        
        if context:
            self.console.print("   Context:")
            for key, value in context.items():
                self.console.print(f"     • {key}: {value}")

class ProgressTracker:
    """Enhanced progress tracking for v1.3 workflow."""
    
    def __init__(self, console: Console):
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        self.tasks = {}
    
    def start_phase(self, phase_name: str, total_steps: int) -> None:
        """Start tracking a new phase."""
        task_id = self.progress.add_task(
            f"Phase: {phase_name}",
            total=total_steps
        )
        self.tasks[phase_name] = task_id
    
    def update_phase(self, phase_name: str, completed: int) -> None:
        """Update phase progress."""
        if phase_name in self.tasks:
            self.progress.update(
                self.tasks[phase_name],
                completed=completed
            )
    
    def show_round_progress(
        self,
        round_num: int,
        topic: str,
        is_checkpoint: bool = False
    ) -> None:
        """Display round progress with checkpoint indicator."""
        checkpoint_marker = "🛑" if is_checkpoint else "  "
        
        self.console.print(
            f"\n{checkpoint_marker} [bold]Round {round_num}[/bold] - "
            f"Topic: [cyan]{topic}[/cyan]"
        )
        
        if is_checkpoint:
            self.console.print(
                "[yellow]   ⚠️  5-Round Checkpoint - User control available[/yellow]"
            )

class EnhancedDashboard:
    """Main dashboard for v1.3 session monitoring."""
    
    def __init__(self, console: Console):
        self.console = console
        self.layout = self._create_layout()
        self.agent_display = SpecializedAgentDisplay(console)
        self.progress_tracker = ProgressTracker(console)
    
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar")
        )
        
        return layout
    
    def update_session_info(self, state: Dict[str, Any]) -> None:
        """Update dashboard with current session state."""
        # Header: Session info
        self.layout["header"].update(
            Panel(
                f"Theme: {state.get('main_topic', 'Not set')}\n"
                f"Phase: {state.get('current_phase', 0)}",
                title="Virtual Agora Session"
            )
        )
        
        # Sidebar: Agent status
        agent_status = self._build_agent_status(state)
        self.layout["sidebar"].update(agent_status)
        
        # Main: Current activity
        current_activity = self._build_activity_display(state)
        self.layout["main"].update(current_activity)
    
    def _build_agent_status(self, state: Dict[str, Any]) -> Panel:
        """Build agent status display."""
        specialized = state.get("specialized_agents", {})
        
        status_lines = ["[bold]Specialized Agents:[/bold]"]
        for agent_type, agent_id in specialized.items():
            color = self.agent_display.agent_colors.get(agent_type, "white")
            status = "🟢 Active" if agent_id else "🔴 Not initialized"
            status_lines.append(f"[{color}]{agent_type}[/{color}]: {status}")
        
        discussing_count = len(state.get("agents", {}))
        status_lines.append(f"\n[bold]Discussing Agents:[/bold] {discussing_count}")
        
        return Panel(
            "\n".join(status_lines),
            title="Agent Status",
            border_style="blue"
        )
```

### Step 4: Main Application Integration

```python
# Updates to src/virtual_agora/main.py

from virtual_agora.ui.hitl_manager import EnhancedHITLManager
from virtual_agora.ui.components import EnhancedDashboard

class VirtualAgoraApplication:
    """Main application class with v1.3 enhancements."""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.hitl_manager = EnhancedHITLManager(self.console)
        self.dashboard = EnhancedDashboard(self.console)
        
        # Initialize all specialized agents
        self.specialized_agents = self._initialize_specialized_agents()
        self.discussing_agents = self._initialize_discussing_agents()
    
    def _initialize_specialized_agents(self) -> Dict[str, LLMAgent]:
        """Initialize all 5 specialized agents."""
        return {
            "moderator": create_moderator(self.config.moderator),
            "summarizer": create_summarizer(self.config.summarizer),
            "topic_report": create_topic_report_agent(self.config.topic_report),
            "ecclesia_report": create_ecclesia_report_agent(
                self.config.ecclesia_report
            ),
        }
    
    def handle_hitl_gate(self, state: VirtualAgoraState) -> Dict[str, Any]:
        """Handle HITL gates with enhanced UI."""
        hitl_state = state.get("hitl_state", {})
        
        if not hitl_state.get("awaiting_approval"):
            return {}
        
        # Create interaction object
        interaction = HITLInteraction(
            approval_type=HITLApprovalType(hitl_state["approval_type"]),
            prompt_message=hitl_state["prompt_message"],
            options=hitl_state.get("options"),
            context={
                "current_round": state.get("current_round"),
                "active_topic": state.get("active_topic"),
                "proposed_agenda": state.get("proposed_agenda"),
                # ... other relevant context
            }
        )
        
        # Process through HITL manager
        response = self.hitl_manager.process_interaction(interaction)
        
        # Update state based on response
        return self._process_hitl_response(
            interaction.approval_type,
            response,
            state
        )
    
    def _process_hitl_response(
        self,
        approval_type: HITLApprovalType,
        response: Dict[str, Any],
        state: VirtualAgoraState
    ) -> Dict[str, Any]:
        """Process HITL response and return state updates."""
        
        if approval_type == HITLApprovalType.PERIODIC_STOP:
            if response.get("force_topic_end"):
                return {
                    "user_forced_conclusion": True,
                    "force_reason": response.get("reason"),
                    "hitl_state": {"awaiting_approval": False}
                }
        
        elif approval_type == HITLApprovalType.AGENDA_APPROVAL:
            if response.get("approved"):
                return {
                    "agenda": {
                        "topics": response["agenda"],
                        "current_topic_index": 0,
                        "completed_topics": []
                    },
                    "agenda_edited": response.get("edited", False),
                    "hitl_state": {"awaiting_approval": False}
                }
        
        # ... handle other approval types
        
        return {"hitl_state": {"awaiting_approval": False}}
```

### Step 5: Session Control Enhancements

```python
# src/virtual_agora/ui/session_control.py

class SessionController:
    """Enhanced session control for v1.3."""
    
    def __init__(self, console: Console):
        self.console = console
        self.interrupt_handler = self._setup_interrupt_handler()
    
    def _setup_interrupt_handler(self) -> None:
        """Setup keyboard interrupt handling."""
        import signal
        
        def handle_interrupt(signum, frame):
            self._show_interrupt_menu()
        
        signal.signal(signal.SIGINT, handle_interrupt)
    
    def _show_interrupt_menu(self) -> None:
        """Show menu when user interrupts."""
        self.console.print("\n[yellow]Session interrupted![/yellow]")
        
        choices = [
            "Resume session",
            "End current topic",
            "Skip to final report",
            "Save and exit",
            "Exit without saving"
        ]
        
        choice = Prompt.ask(
            "What would you like to do?",
            choices=choices,
            default="Resume session"
        )
        
        self._handle_interrupt_choice(choice)
    
    def check_periodic_control(
        self, 
        round_num: int,
        checkpoint_interval: int = 5
    ) -> bool:
        """Check if periodic control point reached."""
        return round_num > 0 and round_num % checkpoint_interval == 0
    
    def display_checkpoint_notification(
        self,
        round_num: int,
        topic: str
    ) -> None:
        """Display checkpoint notification."""
        self.console.bell()  # Audio notification
        
        self.console.print(
            Panel(
                f"[bold yellow]5-Round Checkpoint Reached![/bold yellow]\n\n"
                f"Round: {round_num}\n"
                f"Topic: {topic}\n\n"
                f"You now have the opportunity to:\n"
                f"• End the current topic discussion\n"
                f"• Continue for another 5 rounds\n"
                f"• Modify the discussion parameters",
                title="🛑 User Control Point",
                border_style="yellow",
                expand=False
            )
        )
```

## Implementation Instructions

### Development Workflow

1. **Update HITL State Management**
   - Implement new approval types
   - Create interaction tracking
   - Add response recording

2. **Enhance UI Components**
   - Create specialized agent displays
   - Implement progress tracking
   - Build enhanced dashboard

3. **Integrate with Main Application**
   - Update initialization
   - Wire HITL handling
   - Connect UI updates

4. **Test User Interactions**
   - Test each HITL gate
   - Validate UI responsiveness
   - Check error handling

### UI Design Patterns

```python
# Pattern 1: Consistent HITL prompting
def prompt_user(message: str, options: List[str]) -> str:
    """Consistent prompt pattern."""
    return Prompt.ask(message, choices=options)

# Pattern 2: Context-aware displays
def show_context(console: Console, context: Dict[str, Any]):
    """Show relevant context before prompting."""
    for key, value in context.items():
        console.print(f"{key}: {value}")

# Pattern 3: Progress indication
def show_progress(console: Console, phase: str, percent: float):
    """Show progress for long operations."""
    console.print(f"{phase}: {percent:.0%} complete")
```

### Testing Requirements

1. **HITL Gate Tests**
   ```python
   def test_periodic_stop_gate():
       """Test 5-round periodic stop."""
       manager = EnhancedHITLManager(Console())
       
       interaction = HITLInteraction(
           HITLApprovalType.PERIODIC_STOP,
           "End topic discussion?",
           context={"current_round": 5}
       )
       
       # Mock user response
       with patch("rich.prompt.Confirm.ask", return_value=True):
           response = manager.process_interaction(interaction)
       
       assert response["force_topic_end"] == True
   ```

2. **UI Component Tests**
   ```python
   def test_agent_display():
       """Test specialized agent display."""
       console = Console(file=StringIO())
       display = SpecializedAgentDisplay(console)
       
       display.show_agent_invocation(
           "summarizer",
           "Compress round 3",
           {"round": 3, "topic": "AI Safety"}
       )
       
       output = console.file.getvalue()
       assert "Summarizer Agent Activated" in output
   ```

## Common Challenges and Solutions

### Challenge 1: UI Responsiveness
**Problem**: UI blocks during LLM calls
**Solution**: Use async updates and progress indicators

### Challenge 2: Context Overload
**Problem**: Too much information shown to user
**Solution**: Progressive disclosure, collapsible sections

### Challenge 3: Interrupt Handling
**Problem**: User interrupts during critical operations
**Solution**: Safe interrupt points, state preservation

### Challenge 4: Terminal Compatibility
**Problem**: Rich features not supported on all terminals
**Solution**: Fallback modes, feature detection

## Validation Checklist

- [ ] All v1.3 HITL gates implemented
- [ ] Periodic 5-round stops working
- [ ] Agenda editing functional
- [ ] Agent status displays correct
- [ ] Progress tracking accurate
- [ ] Interrupt handling safe
- [ ] UI responsive and clear
- [ ] All interactions logged

## References

- v1.3 HITL Specification: Section 2 of `docs/project_spec_2.md`
- Current HITL: `src/virtual_agora/ui/human_in_the_loop.py`
- Rich Documentation: https://rich.readthedocs.io/
- UI Components: `src/virtual_agora/ui/components.py`

## Next Phase

Once HITL and UI enhancements are complete, proceed to Phase 5: Testing & Validation.

---

**Document Version**: 1.0
**Phase**: 4 of 5
**Status**: Implementation Guide