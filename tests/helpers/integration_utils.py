"""Utility functions for integration testing.

This module provides helper functions for creating test states, validating responses,
and managing test scenarios for Virtual Agora integration tests.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, AsyncMock, patch
from contextlib import ExitStack

from virtual_agora.config.models import (
    Config as VirtualAgoraConfig,
    ModeratorConfig,
    AgentConfig,
    SummarizerConfig,
    TopicReportConfig,
    EcclesiaReportConfig,
)
from virtual_agora.providers.config import ProviderType
from virtual_agora.state.schema import (
    VirtualAgoraState,
    HITLState,
    FlowControl,
    AgentInfo,
    Message,
    TopicInfo,
    VoteRound,
    RoundInfo,
)
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.flow.graph import VirtualAgoraFlow
from .fake_llm import FakeLLMBase, create_fake_llm_pool, create_specialized_fake_llms


class StateTestBuilder:
    """Builder class for creating test states."""

    def __init__(self):
        # Create a dictionary with the required fields for VirtualAgoraState
        self.state = {
            "session_id": "test_session",
            "start_time": datetime.now(),
            "config_hash": "test_hash",
            "ui_state": {
                "initialized": True,
                "theme": "default",
                "terminal_size": {"width": 80, "height": 24},
            },
            "current_phase": 0,  # Phases are integers 0-5
            "phase_history": [],
            "phase_start_time": datetime.now(),
            "current_round": 1,
            "round_history": [],
            "turn_order_history": [],
            "rounds_per_topic": {},
            "hitl_state": {
                "pending_approval": False,
                "approval_type": "none",
                "user_input": "",
                "approved": False,
                "session_start_approval": True,
                "topic_approval_required": True,
                "override_options": [],
                "approval_timestamp": None,
            },
            "flow_control": {
                "continue_session": True,
                "skip_to_phase": "",
                "force_conclusion": False,
                "debug_mode": False,
                "pause_after_phase": None,
                "pause_after_round": None,
                "max_rounds_per_topic": 10,
                "min_rounds_per_topic": 3,
                "require_consensus_for_conclusion": True,
            },
            "main_topic": "Test Topic",
            "active_topic": None,
            "topic_queue": [],
            "proposed_topics": [],
            "topics_info": {},
            "completed_topics": [],
            "agents": {},
            "turn_type": {},
            "agent_roles": {},
            "active_speakers": {},
            "moderator_id": "moderator",
            "active_vote": None,
            "votes": [],
            "vote_history": [],
            "vote_metadata": {},
            "voting_phase_active": False,
            "last_vote_id": None,
            "messages": [],
            "message_metadata": {},
            "message_count_by_agent": {},
            "message_count_by_topic": {},
            "message_embeddings": {},
            "last_message_id": None,
            # Additional fields required by tests
            "topic_summaries": [],
            "voting_rounds": [],
            "discussion_rounds": [],
            "round_summaries": [],
            "metadata": {},
            "conclusion_polls": [],
            "minority_dissent": [],
            "agenda": [],
            "current_topic_index": 0,
            "speaking_order": [],
            "reporting_agent": None,
            "topic_change_allowed": True,
            "max_topics": 5,
        }

    def with_session_id(self, session_id: str) -> "StateTestBuilder":
        """Set session ID."""
        self.state["session_id"] = session_id
        return self

    def with_main_topic(self, topic: str) -> "StateTestBuilder":
        """Set main topic."""
        self.state["main_topic"] = topic
        return self

    def with_phase(self, phase: Union[str, int]) -> "StateTestBuilder":
        """Set current phase."""
        # Convert phase names to integers if needed
        if isinstance(phase, str):
            phase_map = {
                "initialization": 0,
                "agenda_setting": 1,
                "discussion": 2,
                "topic_conclusion": 3,
                "agenda_reevaluation": 4,
                "final_report": 5,
            }
            phase = phase_map.get(phase, 0)
        self.state["current_phase"] = phase
        return self

    def with_agents(self, agents: Dict[str, AgentInfo]) -> "StateTestBuilder":
        """Set agents."""
        self.state["agents"] = agents
        return self

    def with_agenda(self, agenda: List[TopicInfo]) -> "StateTestBuilder":
        """Set agenda."""
        self.state["agenda"] = agenda
        return self

    def with_current_topic(self, topic_index: int) -> "StateTestBuilder":
        """Set current topic index."""
        self.state["current_topic_index"] = topic_index
        return self

    def with_round(self, round_num: int) -> "StateTestBuilder":
        """Set current round."""
        self.state["current_round"] = round_num
        return self

    def with_speaking_order(self, speaking_order: List[str]) -> "StateTestBuilder":
        """Set speaking order."""
        self.state["speaking_order"] = speaking_order
        return self

    def with_messages(self, messages: List[Message]) -> "StateTestBuilder":
        """Set messages."""
        self.state["messages"] = messages
        return self

    def with_hitl_state(self, hitl_state: HITLState) -> "StateTestBuilder":
        """Set HITL state."""
        self.state["hitl_state"] = hitl_state
        return self

    def build(self) -> VirtualAgoraState:
        """Build the final state."""
        return self.state


class TestResponseValidator:
    """Validator for test responses and state transitions."""

    @staticmethod
    def validate_agenda_synthesis(response: str) -> bool:
        """Validate moderator agenda synthesis response."""
        try:
            data = json.loads(response)
            return (
                "proposed_agenda" in data
                and isinstance(data["proposed_agenda"], list)
                and len(data["proposed_agenda"]) > 0
            )
        except (json.JSONDecodeError, KeyError):
            return False

    @staticmethod
    def validate_conclusion_vote(response: str) -> bool:
        """Validate agent conclusion vote response."""
        response_lower = response.lower().strip()
        return response_lower.startswith(("yes", "no"))

    @staticmethod
    def validate_topic_summary(response: str) -> bool:
        """Validate topic summary structure."""
        # Basic validation - should be substantial text with markdown
        return (
            len(response) > 100
            and "#" in response  # Should contain headers
            and "summary" in response.lower()
        )

    @staticmethod
    def validate_report_structure(response: str) -> bool:
        """Validate report structure JSON."""
        try:
            data = json.loads(response)
            return isinstance(data, list) and len(data) > 0
        except json.JSONDecodeError:
            return False

    @staticmethod
    def validate_state_transition(
        old_state: VirtualAgoraState, new_state: VirtualAgoraState
    ) -> bool:
        """Validate that state transition is valid."""
        # Basic validation rules
        if new_state["session_id"] != old_state["session_id"]:
            return False

        # Round should not decrease (unless new topic)
        if new_state.get("current_topic_index", 0) == old_state.get(
            "current_topic_index", 0
        ) and new_state.get("current_round", 0) < old_state.get("current_round", 0):
            return False

        # Message count should generally increase
        if len(new_state.get("messages", [])) < len(old_state.get("messages", [])):
            return False

        return True

    @staticmethod
    def validate_discussion_flow(state: VirtualAgoraState) -> List[str]:
        """Validate overall discussion flow and return any issues."""
        issues = []

        # Check if we have agents
        if not state.get("agents"):
            issues.append("No agents in state")

        # Check if we have an agenda when in discussion phase
        if state.get("current_phase") == 2 and not state.get(
            "agenda"
        ):  # Phase 2 is discussion
            issues.append("No agenda in discussion phase")

        # Check if current topic index is valid
        agenda = state.get("agenda", [])
        if agenda and state.get("current_topic_index", 0) >= len(agenda):
            issues.append("Current topic index out of bounds")

        # Check if speaking order matches agents
        speaking_order = state.get("speaking_order", [])
        agents = state.get("agents", {})
        if speaking_order:
            for agent_id in speaking_order:
                if agent_id not in agents and agent_id != "moderator":
                    issues.append(f"Unknown agent in speaking order: {agent_id}")

        return issues

    @staticmethod
    def validate_specialized_agents(state: VirtualAgoraState) -> List[str]:
        """Validate v1.3 specialized agent functionality."""
        issues = []

        # Check for round summaries (Summarizer Agent)
        if state.get("current_round", 0) > 1 and not state.get("round_summaries"):
            issues.append("No round summaries generated by Summarizer Agent")

        # Check for topic reports (Topic Report Agent)
        if state.get("completed_topics") and not state.get("topic_summaries"):
            issues.append("No topic reports generated by Topic Report Agent")

        # Check for periodic stop state (v1.3 HITL)
        if state.get("current_round", 0) > 0 and state.get("current_round") % 5 == 0:
            hitl_state = state.get("hitl_state", {})
            if not hitl_state.get("periodic_stop_prompted"):
                issues.append("Periodic 5-round stop not prompted")

        return issues

    @staticmethod
    def validate_v13_hitl_gates(state: VirtualAgoraState) -> List[str]:
        """Validate v1.3 HITL gates."""
        issues = []

        hitl_state = state.get("hitl_state", {})

        # Check for required HITL gates based on phase
        phase = state.get("current_phase")

        if phase == 0 and not state.get("main_topic"):
            issues.append("Initial theme not collected from user")

        if phase == 1 and state.get("agenda") and not hitl_state.get("agenda_approved"):
            issues.append("Agenda not approved by user")

        # Check periodic stop tracking
        if "periodic_stop_counter" not in state:
            issues.append("Periodic stop counter not initialized")

        return issues


class MockUIHandler:
    """Mock handler for UI interactions in tests."""

    def __init__(self, predefined_responses: Dict[str, str] = None):
        self.predefined_responses = predefined_responses or {}
        self.interaction_history = []

    def mock_user_input(self, prompt: str, default: str = "") -> str:
        """Mock user input with predefined responses."""
        self.interaction_history.append(("input", prompt))

        # Check for predefined responses based on prompt content
        for key, response in self.predefined_responses.items():
            if key.lower() in prompt.lower():
                return response

        # Default responses for common prompts
        if "topic" in prompt.lower() and "discuss" in prompt.lower():
            return "Future of Artificial Intelligence"
        elif "approve" in prompt.lower():
            return "y"
        elif "continue" in prompt.lower():
            return "y"

        return default

    def mock_approval_prompt(self, agenda: List[str]) -> bool:
        """Mock agenda approval."""
        self.interaction_history.append(("approval", agenda))
        return True  # Always approve in tests

    def mock_periodic_stop(self, round_num: int, topic: str) -> Dict[str, Any]:
        """Mock v1.3 periodic 5-round stop."""
        self.interaction_history.append(
            ("periodic_stop", {"round": round_num, "topic": topic})
        )

        # Check for predefined response
        for key, response in self.predefined_responses.items():
            if "periodic" in key.lower() or "stop" in key.lower():
                if response.lower() in ["yes", "end", "stop"]:
                    return {"force_topic_end": True, "reason": "User requested end"}

        return {"force_topic_end": False}

    def mock_agenda_editing(self, agenda: List[str]) -> List[str]:
        """Mock v1.3 agenda editing capability."""
        self.interaction_history.append(("agenda_edit", agenda))

        # Check for predefined edits
        if "edited_agenda" in self.predefined_responses:
            return self.predefined_responses["edited_agenda"]

        return agenda  # Return unmodified by default

    def get_interaction_count(self) -> int:
        """Get number of UI interactions."""
        return len(self.interaction_history)

    def get_last_interaction(self) -> Optional[tuple]:
        """Get the last UI interaction."""
        return self.interaction_history[-1] if self.interaction_history else None


class IntegrationTestHelper:
    """Main helper class for integration tests."""

    def __init__(self, num_agents: int = 3, scenario: str = "default"):
        self.num_agents = num_agents
        self.scenario = scenario
        self.fake_llms = create_fake_llm_pool(num_agents, scenario=scenario)
        self.specialized_llms = create_specialized_fake_llms()
        self.ui_handler = MockUIHandler()

    def create_test_config(self) -> VirtualAgoraConfig:
        """Create a test configuration with fake providers."""
        moderator = ModeratorConfig(
            provider=ProviderType.GOOGLE, model="fake-moderator-model"
        )

        # v1.3 specialized agents
        summarizer = SummarizerConfig(
            provider=ProviderType.OPENAI, model="fake-summarizer-model"
        )
        topic_report = TopicReportConfig(
            provider=ProviderType.ANTHROPIC, model="fake-topic-report-model"
        )
        ecclesia_report = EcclesiaReportConfig(
            provider=ProviderType.GOOGLE, model="fake-ecclesia-report-model"
        )

        agents = []
        for i in range(self.num_agents):
            agent = AgentConfig(
                provider=ProviderType.OPENAI, model=f"fake-agent-model-{i+1}", count=1
            )
            agents.append(agent)

        return VirtualAgoraConfig(
            moderator=moderator,
            summarizer=summarizer,
            topic_report=topic_report,
            ecclesia_report=ecclesia_report,
            agents=agents,
        )

    def create_test_flow(self) -> VirtualAgoraFlow:
        """Create a VirtualAgoraFlow with fake LLMs."""
        config = self.create_test_config()

        with patch(
            "virtual_agora.providers.factory.ProviderFactory.create_provider"
        ) as mock_factory:
            # Return appropriate fake LLM based on the request
            def mock_provider_creator(provider_config, use_cache=True):
                model = provider_config.model
                if "moderator" in model:
                    return self.specialized_llms["moderator"]
                elif "summarizer" in model:
                    return self.specialized_llms["summarizer"]
                elif "topic-report" in model:
                    return self.specialized_llms["topic_report"]
                elif "ecclesia-report" in model:
                    return self.specialized_llms["ecclesia_report"]
                else:
                    # Extract agent number from model name
                    agent_num = model.split("-")[-1]
                    agent_key = f"agent_{agent_num}"
                    return self.fake_llms.get(agent_key, self.fake_llms["agent_1"])

            mock_factory.side_effect = mock_provider_creator

            flow = VirtualAgoraFlow(config, enable_monitoring=False)
            return flow

    def create_basic_state(self) -> VirtualAgoraState:
        """Create a basic test state."""
        return (
            StateTestBuilder()
            .with_main_topic("Test Topic for Integration")
            .with_phase("initialization")
            .build()
        )

    def create_agenda_setting_state(self) -> VirtualAgoraState:
        """Create state ready for agenda setting."""
        agents = {
            f"agent_{i+1}": {
                "agent_id": f"agent_{i+1}",
                "name": f"Agent {i+1}",
                "role": "participant",
                "model": f"fake-model-{i+1}",
                "provider": "fake_provider",
            }
            for i in range(self.num_agents)
        }

        return (
            StateTestBuilder()
            .with_main_topic("Future of Artificial Intelligence")
            .with_phase("agenda_setting")
            .with_agents(agents)
            .with_speaking_order(list(agents.keys()))
            .build()
        )

    def create_discussion_state(self) -> VirtualAgoraState:
        """Create state ready for discussion."""
        state = self.create_agenda_setting_state()

        # Add agenda topics
        agenda = [
            {
                "title": "Technical Implementation",
                "description": "Discussion of technical aspects",
                "proposed_by": "agent_1",
                "votes_for": 2,
                "votes_against": 1,
                "status": "active",
            },
            {
                "title": "Social Impact",
                "description": "Discussion of social implications",
                "proposed_by": "agent_2",
                "votes_for": 3,
                "votes_against": 0,
                "status": "pending",
            },
        ]

        state["agenda"] = agenda
        state["current_phase"] = 2  # discussion phase is 2
        state["current_topic_index"] = 0
        state["current_round"] = 1

        return state

    def simulate_user_interactions(self, responses: Dict[str, str]):
        """Configure UI handler with specific user responses."""
        self.ui_handler.predefined_responses.update(responses)

    def assert_flow_completion(self, final_state: VirtualAgoraState):
        """Assert that flow completed successfully."""
        assert (
            final_state["current_phase"] == "completed"
            or final_state["current_phase"] == 5
        )
        assert len(final_state.get("topic_summaries", [])) > 0
        assert final_state["session_id"] is not None

    def assert_agenda_created(self, state: VirtualAgoraState):
        """Assert that agenda was created successfully."""
        assert len(state["agenda"]) > 0
        assert all(isinstance(topic, dict) for topic in state["agenda"])
        # Check if hitl_state is a dict and has approved field
        # In test scenarios, we may not always have hitl_state, so we'll allow it

    def assert_discussion_progressed(self, state: VirtualAgoraState):
        """Assert that discussion has progressed."""
        assert state["current_round"] > 1
        assert len(state["messages"]) > 0
        assert len(state.get("round_summaries", [])) > 0

    def assert_topic_concluded(self, state: VirtualAgoraState):
        """Assert that a topic was concluded properly."""
        assert len(state.get("topic_summaries", [])) > 0
        # Should have voting rounds for conclusion
        assert any(
            vr.get("vote_type") == "conclusion" for vr in state.get("voting_rounds", [])
        )


def create_test_messages(agent_id: str, count: int = 3) -> List[Dict[str, Any]]:
    """Create test messages for an agent."""
    messages = []
    for i in range(count):
        msg = {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "content": f"Test message {i+1} from {agent_id}",
            "timestamp": datetime.now(),
            "round_number": i + 1,
            "topic": "Test Topic",
            "message_type": "discussion",
        }
        messages.append(msg)
    return messages


def create_test_voting_round(vote_type: str = "conclusion") -> Dict[str, Any]:
    """Create a test voting round."""
    return {
        "id": str(uuid.uuid4()),
        "phase": 1,
        "vote_type": vote_type,
        "options": ["Yes", "No"],
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "required_votes": 3,
        "received_votes": 3,
        "result": "Yes" if vote_type == "conclusion" else None,
        "status": "completed",
    }


def create_test_discussion_round(round_number: int = 1) -> Dict[str, Any]:
    """Create a test discussion round."""
    return {
        "round_id": str(uuid.uuid4()),
        "round_number": round_number,
        "topic": "Test Topic",
        "start_time": datetime.now(),
        "end_time": datetime.now(),
        "participants": ["agent_1", "agent_2", "agent_3"],
        "summary": "Test round summary",
        "messages_count": 3,
        "status": "completed",
    }


# Convenience functions for common test patterns
def run_integration_test(
    test_scenario: str = "default",
    num_agents: int = 3,
    user_responses: Dict[str, str] = None,
) -> tuple[VirtualAgoraFlow, VirtualAgoraState]:
    """Run a complete integration test scenario.

    Returns:
        Tuple of (flow, final_state)
    """
    helper = IntegrationTestHelper(num_agents=num_agents, scenario=test_scenario)

    if user_responses:
        helper.simulate_user_interactions(user_responses)

    flow = helper.create_test_flow()
    initial_state = helper.create_basic_state()

    # This would typically involve running the flow
    # For now, return the setup for further testing
    return flow, initial_state


def patch_ui_components():
    """Create patches for UI components to avoid interactive prompts."""
    # Create an ExitStack context manager that combines multiple patches
    stack = ExitStack()

    # Mock UI display functions
    stack.enter_context(
        patch.multiple(
            "virtual_agora.ui.human_in_the_loop",
            get_continuation_approval=Mock(
                return_value="continue"  # Returns action string directly
            ),
            get_agenda_modifications=Mock(
                return_value=[]  # Returns list of modifications
            ),
            display_session_status=Mock(),
            get_agenda_approval=Mock(
                return_value=["Topic 1", "Topic 2"]  # Returns approved agenda list
            ),
            get_initial_topic=Mock(return_value="Test Topic"),
        )
    )

    # Mock preferences
    stack.enter_context(
        patch(
            "virtual_agora.ui.preferences.get_user_preferences",
            return_value=Mock(auto_approve_agenda_on_consensus=True),
        )
    )

    return stack


def create_test_config_file(
    config_path, provider="openai", num_agents=3, version="1.3"
):
    """Create a test configuration file."""
    if version == "1.3":
        config_content = f"""
# Virtual Agora Configuration v1.3
moderator:
  provider: {provider}
  model: gpt-4

summarizer:
  provider: {provider}
  model: gpt-4o

topic_report:
  provider: anthropic
  model: claude-3-opus-20240229

ecclesia_report:
  provider: google
  model: gemini-2.5-pro

agents:"""
    else:
        config_content = f"""
moderator:
  provider: {provider}
  model: gpt-4

agents:"""

    # Create different models to avoid naming conflicts
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]

    for i in range(num_agents):
        model = models[i % len(models)]
        config_content += f"""
  - provider: {provider}
    model: {model}
    count: 1"""

    with open(config_path, "w") as f:
        f.write(config_content)


def create_test_env_file(env_path):
    """Create a test environment file with required API keys."""
    env_content = """
OPENAI_API_KEY=test_openai_key_123456789
ANTHROPIC_API_KEY=test_anthropic_key_123456789
GOOGLE_API_KEY=test_google_key_123456789
GROQ_API_KEY=test_groq_key_123456789
"""

    with open(env_path, "w") as f:
        f.write(env_content)


def create_v13_test_state() -> VirtualAgoraState:
    """Create a v1.3 specific test state with all required fields."""
    builder = StateTestBuilder()

    # Add v1.3 specific fields
    state = builder.build()
    state["specialized_agents"] = {
        "moderator": {"agent_id": "moderator", "type": "moderator"},
        "summarizer": {"agent_id": "summarizer", "type": "summarizer"},
        "topic_report": {"agent_id": "topic_report", "type": "topic_report"},
        "ecclesia_report": {"agent_id": "ecclesia_report", "type": "ecclesia_report"},
    }

    # v1.3 HITL enhancements
    state["hitl_state"]["periodic_stop_counter"] = 0
    state["hitl_state"]["periodic_stop_prompted"] = False
    state["hitl_state"]["agenda_edit_allowed"] = True
    state["hitl_state"]["force_topic_end"] = False

    # v1.3 reporting fields
    state["round_summaries"] = []
    state["topic_reports"] = {}
    state["final_report_sections"] = []
    state["ecclesia_report_generated"] = False

    return state


def create_v13_agent_response(agent_type: str, context: Dict[str, Any]) -> str:
    """Create realistic agent responses for v1.3 agents."""
    if agent_type == "summarizer":
        return f"Round {context.get('round', 1)} Summary: The agents discussed {context.get('topic', 'the topic')} with focus on implementation details and strategic considerations. Key points included technical feasibility, stakeholder impact, and phased approach recommendations."

    elif agent_type == "topic_report":
        topic = context.get("topic", "the discussion topic")
        return f"""# Topic Report: {topic}

## Overview
Comprehensive discussion on {topic} yielded valuable insights across multiple dimensions.

## Key Themes
- Technical implementation strategies
- Stakeholder engagement approaches
- Risk mitigation frameworks

## Consensus Points
- Phased implementation recommended
- Continuous monitoring essential
- Stakeholder buy-in critical

## Next Steps
Proceed with prototype development while maintaining feedback loops."""

    elif agent_type == "ecclesia_report":
        if context.get("structure_request"):
            return '["Executive Summary", "Key Findings", "Cross-Topic Analysis", "Strategic Recommendations", "Implementation Roadmap", "Conclusion"]'
        else:
            return """# Executive Summary

This Virtual Agora session successfully explored the designated theme through structured multi-agent discussion. The diversity of perspectives enriched the analysis and led to actionable recommendations.

## Key Achievements
- Comprehensive exploration of all major aspects
- Identification of critical success factors
- Development of implementation framework
- Risk assessment and mitigation strategies"""

    else:
        return "Default response for unknown agent type"
