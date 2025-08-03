"""GraphInterrupt simulation for deterministic LLM testing.

This module provides mechanisms for simulating GraphInterrupt scenarios
in deterministic LLMs, allowing for testing of user input and interaction
flows without requiring real user input.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import uuid

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InterruptSimulation:
    """Configuration for simulating a GraphInterrupt scenario."""

    interrupt_type: str
    trigger_condition: Callable[[Dict[str, Any]], bool]
    interrupt_data: Dict[str, Any]
    user_response: Dict[str, Any]
    description: str = ""


class InterruptSimulator:
    """Manages GraphInterrupt simulation for deterministic testing.

    This class provides mechanisms for triggering and handling GraphInterrupts
    in a controlled, deterministic manner for testing purposes.
    """

    def __init__(self):
        self.simulations = {}
        self.triggered_interrupts = []
        self.interrupt_responses = {}
        self.auto_respond = True

    def register_simulation(self, simulation: InterruptSimulation) -> None:
        """Register an interrupt simulation scenario.

        Args:
            simulation: InterruptSimulation configuration
        """
        self.simulations[simulation.interrupt_type] = simulation
        logger.debug(f"Registered interrupt simulation: {simulation.interrupt_type}")

    def set_interrupt_response(
        self, interrupt_type: str, response: Dict[str, Any]
    ) -> None:
        """Set a predetermined response for an interrupt type.

        Args:
            interrupt_type: Type of interrupt
            response: Response data to return when interrupt is triggered
        """
        self.interrupt_responses[interrupt_type] = response
        logger.debug(
            f"Set interrupt response for {interrupt_type}: {list(response.keys())}"
        )

    def should_trigger_interrupt(
        self, context: Dict[str, Any], interrupt_type: str
    ) -> bool:
        """Check if an interrupt should be triggered based on context.

        Args:
            context: Current conversation context
            interrupt_type: Type of interrupt to check

        Returns:
            True if interrupt should be triggered
        """
        if interrupt_type not in self.simulations:
            return False

        simulation = self.simulations[interrupt_type]
        return simulation.trigger_condition(context)

    def create_interrupt_data(
        self, interrupt_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create interrupt data for a specific interrupt type.

        Args:
            interrupt_type: Type of interrupt
            context: Current conversation context

        Returns:
            Dictionary containing interrupt data
        """
        if interrupt_type not in self.simulations:
            return {}

        simulation = self.simulations[interrupt_type]

        # Merge base interrupt data with context-specific data
        interrupt_data = simulation.interrupt_data.copy()
        interrupt_data.update(
            {
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "simulation_id": str(uuid.uuid4()),
            }
        )

        return interrupt_data

    def get_interrupt_response(
        self, interrupt_type: str, interrupt_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get the predetermined response for an interrupt.

        Args:
            interrupt_type: Type of interrupt
            interrupt_data: Data from the interrupt

        Returns:
            Response data to continue execution
        """
        if interrupt_type in self.interrupt_responses:
            response = self.interrupt_responses[interrupt_type].copy()
        elif interrupt_type in self.simulations:
            response = self.simulations[interrupt_type].user_response.copy()
        else:
            response = {"status": "continue"}  # Default response

        # Record this interrupt
        self.triggered_interrupts.append(
            {
                "type": interrupt_type,
                "data": interrupt_data,
                "response": response,
                "timestamp": datetime.now(),
            }
        )

        logger.info(
            f"Simulated interrupt {interrupt_type} with response: {list(response.keys())}"
        )

        return response

    def get_interrupt_history(self) -> List[Dict[str, Any]]:
        """Get history of all triggered interrupts.

        Returns:
            List of interrupt records
        """
        return self.triggered_interrupts.copy()

    def reset_simulator(self) -> None:
        """Reset the simulator state."""
        self.triggered_interrupts.clear()
        self.interrupt_responses.clear()
        logger.debug("Reset interrupt simulator state")


# Default interrupt simulations for Virtual Agora


def create_agenda_approval_simulation() -> InterruptSimulation:
    """Create simulation for agenda approval interrupt."""

    def trigger_condition(context: Dict[str, Any]) -> bool:
        return context.get("call_count", 0) <= 3 and any(
            keyword in context.get("last_user_message", "").lower()
            for keyword in ["agenda", "propose", "structure"]
        )

    return InterruptSimulation(
        interrupt_type="agenda_approval",
        trigger_condition=trigger_condition,
        interrupt_data={
            "type": "agenda_approval",
            "proposed_agenda": [
                "Core Concepts and Definitions",
                "Current State Analysis",
                "Key Challenges and Opportunities",
                "Potential Solutions and Approaches",
                "Implementation Considerations",
            ],
        },
        user_response={
            "agenda_approved": True,
            "topic_queue": [
                "Core Concepts and Definitions",
                "Current State Analysis",
                "Key Challenges and Opportunities",
                "Potential Solutions and Approaches",
                "Implementation Considerations",
            ],
            "final_agenda": [
                "Core Concepts and Definitions",
                "Current State Analysis",
                "Key Challenges and Opportunities",
                "Potential Solutions and Approaches",
                "Implementation Considerations",
            ],
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "agenda_approval",
                "approval_action": "approve",
            },
        },
        description="Simulates user approving proposed discussion agenda",
    )


def create_user_participation_simulation() -> InterruptSimulation:
    """Create simulation for user turn participation interrupt."""

    def trigger_condition(context: Dict[str, Any]) -> bool:
        return (
            context.get("call_count", 0) >= 3
            and context.get("call_count", 0) % 3 == 0  # Every 3rd call
        )

    return InterruptSimulation(
        interrupt_type="user_turn_participation",
        trigger_condition=trigger_condition,
        interrupt_data={
            "type": "user_turn_participation",
            "current_round": 3,
            "current_topic": "Discussion Topic",
            "previous_summary": "Previous round summary",
        },
        user_response={
            "user_turn_decision": "continue",
            "user_participation_message": "Thank you for the discussion. I'd like to add that this topic is very important for our community.",
        },
        description="Simulates user participating in discussion round",
    )


def create_topic_continuation_simulation() -> InterruptSimulation:
    """Create simulation for topic continuation interrupt."""

    def trigger_condition(context: Dict[str, Any]) -> bool:
        return context.get("call_count", 0) >= 5 and any(
            keyword in context.get("last_user_message", "").lower()
            for keyword in ["continue", "next", "proceed", "topic"]
        )

    return InterruptSimulation(
        interrupt_type="topic_continuation",
        trigger_condition=trigger_condition,
        interrupt_data={
            "type": "topic_continuation",
            "completed_topic": "Current Topic",
            "remaining_topics": ["Next Topic 1", "Next Topic 2"],
            "agent_recommendation": "continue",
        },
        user_response={
            "user_approves_continuation": True,
            "user_requests_end": False,
            "user_requested_modification": False,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "topic_continuation",
            },
        },
        description="Simulates user approving continuation to next topic",
    )


def create_periodic_stop_simulation() -> InterruptSimulation:
    """Create simulation for periodic checkpoint interrupt."""

    def trigger_condition(context: Dict[str, Any]) -> bool:
        return (
            context.get("call_count", 0) % 5 == 0  # Every 5th call
            and context.get("call_count", 0) > 0
        )

    return InterruptSimulation(
        interrupt_type="periodic_stop",
        trigger_condition=trigger_condition,
        interrupt_data={
            "type": "periodic_stop",
            "current_round": 5,
            "current_topic": "Discussion Topic",
            "checkpoint_interval": 3,
        },
        user_response={
            "user_periodic_decision": "continue",
            "periodic_stop_counter": 1,
            "hitl_state": {
                "awaiting_approval": False,
                "approval_type": "periodic_stop",
            },
        },
        description="Simulates user choosing to continue at periodic checkpoint",
    )


# Global interrupt simulator
_global_simulator = InterruptSimulator()


def get_interrupt_simulator() -> InterruptSimulator:
    """Get the global interrupt simulator instance.

    Returns:
        Global InterruptSimulator instance
    """
    return _global_simulator


def setup_default_interrupt_simulations() -> None:
    """Set up default interrupt simulations for Virtual Agora."""
    simulator = get_interrupt_simulator()

    # Register all default simulations
    simulator.register_simulation(create_agenda_approval_simulation())
    simulator.register_simulation(create_user_participation_simulation())
    simulator.register_simulation(create_topic_continuation_simulation())
    simulator.register_simulation(create_periodic_stop_simulation())

    logger.info("Set up default interrupt simulations")


def reset_interrupt_simulator() -> None:
    """Reset the global interrupt simulator."""
    _global_simulator.reset_simulator()


class InterruptTriggeringMixin:
    """Mixin for LLMs that can trigger GraphInterrupts.

    This mixin provides functionality for deterministic LLMs to trigger
    GraphInterrupt scenarios based on context and predefined conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interrupt_simulator = get_interrupt_simulator()

    def check_and_trigger_interrupts(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if any interrupts should be triggered and handle them.

        Args:
            context: Current conversation context

        Returns:
            Interrupt response data if interrupt was triggered, None otherwise
        """
        for interrupt_type in self.interrupt_simulator.simulations:
            if self.interrupt_simulator.should_trigger_interrupt(
                context, interrupt_type
            ):

                # Create interrupt data
                interrupt_data = self.interrupt_simulator.create_interrupt_data(
                    interrupt_type, context
                )

                # Get predetermined response
                response = self.interrupt_simulator.get_interrupt_response(
                    interrupt_type, interrupt_data
                )

                # In real testing, this would trigger actual GraphInterrupt
                # For now, we just return the response data
                logger.info(f"Would trigger GraphInterrupt: {interrupt_type}")

                return response

        return None
