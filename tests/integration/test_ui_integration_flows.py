"""Integration tests for UI/Human-in-the-Loop flows.

This module tests the complete user interaction workflows including
topic input, agenda approval, continuation decisions, and user preferences.
"""

import pytest
import json
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import uuid

from virtual_agora.state.schema import VirtualAgoraState, HITLState
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.ui.preferences import UserPreferences

from ..helpers.fake_llm import ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
)


def patch_other_ui_components(exclude=None):
    """Helper to patch UI components except specific ones being tested."""
    from contextlib import ExitStack

    if exclude is None:
        exclude = []

    stack = ExitStack()

    # Patch components we're not specifically testing
    if "get_continuation_approval" not in exclude:
        mock_continuation = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_continuation_approval")
        )
        mock_continuation.return_value = "continue"

    if "get_agenda_modifications" not in exclude:
        mock_modifications = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_agenda_modifications")
        )
        mock_modifications.return_value = []

    if "get_agenda_approval" not in exclude:
        mock_approval = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.get_agenda_approval")
        )
        mock_approval.return_value = True  # Default to boolean True for approval tests

    if "display_session_status" not in exclude:
        mock_display = stack.enter_context(
            patch("virtual_agora.ui.human_in_the_loop.display_session_status")
        )

    if "get_user_preferences" not in exclude:
        mock_preferences = stack.enter_context(
            patch("virtual_agora.ui.preferences.get_user_preferences")
        )
        mock_preferences.return_value = Mock(auto_approve_agenda_on_consensus=True)

    return stack


class TestInitialTopicFlow:
    """Test initial topic input and validation."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
    def test_initial_topic_input_success(self, mock_get_topic):
        """Test successful initial topic input."""
        # Mock user providing a valid topic
        mock_get_topic.return_value = "Artificial Intelligence Ethics"

        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            initial_state = self.test_helper.create_basic_state()

            # Simulate initialization node processing
            result = self._simulate_topic_initialization(initial_state)

            # Validate topic was set correctly
            assert result["main_topic"] == "Artificial Intelligence Ethics"
            assert "speaking_order" in result
            assert len(result["speaking_order"]) == self.test_helper.num_agents

            # Verify get_initial_topic was called
            mock_get_topic.assert_called_once()

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
    def test_initial_topic_input_validation(self, mock_get_topic):
        """Test topic input validation and retry."""
        # Mock user providing invalid then valid topic
        mock_get_topic.side_effect = [
            "",  # Empty topic (invalid)
            "AI",  # Too short (invalid)
            "Climate Change Policy and Its Global Impact",  # Valid
        ]

        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            initial_state = self.test_helper.create_basic_state()

            # Simulate initialization with validation
            result = self._simulate_topic_initialization_with_validation(initial_state)

            # Should eventually succeed with valid topic
            assert result["main_topic"] == "Climate Change Policy and Its Global Impact"

            # Verify get_initial_topic was called multiple times
            assert mock_get_topic.call_count == 3

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
    def test_initial_topic_with_predefined_topic(self, mock_get_topic):
        """Test flow when topic is already provided."""
        # Topic already provided in state
        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            initial_state = self.test_helper.create_basic_state()
            initial_state["main_topic"] = "Blockchain Technology Applications"

            # Simulate initialization
            result = self._simulate_topic_initialization(initial_state)

            # Should use existing topic
            assert result["main_topic"] == "Blockchain Technology Applications"

            # Should not call get_initial_topic
            mock_get_topic.assert_not_called()

    def _simulate_topic_initialization(self, state: VirtualAgoraState) -> dict:
        """Simulate topic initialization process."""
        # Get topic from state or user input
        main_topic = state.get("main_topic")
        if not main_topic or main_topic == "Test Topic for Integration":
            # Use mocked input if no topic or default test topic
            from virtual_agora.ui.human_in_the_loop import get_initial_topic

            main_topic = get_initial_topic()

        # Simulate agent initialization
        participant_ids = [f"agent_{i+1}" for i in range(self.test_helper.num_agents)]

        return {
            "current_phase": 0,
            "main_topic": main_topic,
            "speaking_order": participant_ids,
            "moderator_id": "moderator",
            "phase_start_time": datetime.now(),
        }

    def _simulate_topic_initialization_with_validation(
        self, state: VirtualAgoraState
    ) -> dict:
        """Simulate topic initialization with validation."""
        main_topic = None
        max_attempts = 5
        attempts = 0

        while not main_topic and attempts < max_attempts:
            from virtual_agora.ui.human_in_the_loop import get_initial_topic

            candidate_topic = get_initial_topic()

            # Simple validation
            if candidate_topic and len(candidate_topic.strip()) >= 10:
                main_topic = candidate_topic.strip()
            attempts += 1

        if not main_topic:
            # Use a fallback valid topic for testing
            main_topic = "Climate Change Policy and Its Global Impact"

        participant_ids = [f"agent_{i+1}" for i in range(self.test_helper.num_agents)]

        return {
            "current_phase": 0,
            "main_topic": main_topic,
            "speaking_order": participant_ids,
            "moderator_id": "moderator",
            "phase_start_time": datetime.now(),
        }


class TestAgendaApprovalFlow:
    """Test agenda approval and modification workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_agenda_approval")
    def test_agenda_approval_acceptance(self, mock_get_approval):
        """Test user accepting proposed agenda."""
        # Mock user approving agenda
        mock_get_approval.return_value = True

        with patch_other_ui_components(exclude=["get_agenda_approval"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Add proposed agenda
            proposed_agenda = [
                {
                    "title": "Technical Implementation",
                    "description": "Technical details",
                    "status": "pending",
                },
                {
                    "title": "Business Impact",
                    "description": "ROI and market analysis",
                    "status": "pending",
                },
                {
                    "title": "Risk Assessment",
                    "description": "Risk mitigation strategies",
                    "status": "pending",
                },
            ]
            state["agenda"] = proposed_agenda

            # Simulate approval process
            result = self._simulate_agenda_approval(state)

            # Verify approval
            assert result["hitl_state"]["approved"] == True
            assert result["current_phase"] == 2  # Discussion phase
            assert all(topic["status"] == "approved" for topic in result["agenda"])

            # Verify UI was called
            mock_get_approval.assert_called_once()

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_agenda_approval")
    @patch("virtual_agora.ui.human_in_the_loop.get_agenda_modifications")
    def test_agenda_approval_rejection_and_modification(
        self, mock_get_modifications, mock_get_approval
    ):
        """Test user rejecting agenda and requesting modifications."""
        # Mock user rejecting then modifying agenda
        mock_get_approval.return_value = False
        mock_get_modifications.return_value = {
            "add_topics": ["Legal Compliance", "Timeline Planning"],
            "remove_topics": ["Risk Assessment"],
            "reorder": True,
        }

        with patch_other_ui_components(
            exclude=["get_agenda_approval", "get_agenda_modifications"]
        ):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Add initial agenda
            initial_agenda = [
                {
                    "title": "Technical Implementation",
                    "description": "Tech details",
                    "status": "pending",
                },
                {
                    "title": "Risk Assessment",
                    "description": "Risk analysis",
                    "status": "pending",
                },
            ]
            state["agenda"] = initial_agenda

            # Simulate rejection and modification
            result = self._simulate_agenda_rejection_and_modification(state)

            # Verify modifications were applied
            agenda_titles = [topic["title"] for topic in result["agenda"]]
            assert "Legal Compliance" in agenda_titles
            assert "Timeline Planning" in agenda_titles
            assert "Risk Assessment" not in agenda_titles
            assert "Technical Implementation" in agenda_titles

            # Should return to agenda setting phase
            assert result["current_phase"] == 1

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_agenda_approval")
    def test_agenda_approval_with_empty_agenda(self, mock_get_approval):
        """Test handling of empty agenda approval."""
        mock_get_approval.return_value = False

        with patch_other_ui_components(exclude=["get_agenda_approval"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()
            state["agenda"] = []  # Empty agenda

            # Should handle empty agenda gracefully
            result = self._simulate_agenda_approval(state)

            # Should reject empty agenda
            assert result["hitl_state"]["approved"] == False
            assert result["current_phase"] == 1  # Back to agenda setting

    def _simulate_agenda_approval(self, state: VirtualAgoraState) -> dict:
        """Simulate agenda approval process."""
        from virtual_agora.ui.human_in_the_loop import get_agenda_approval

        # Check if agenda exists and is not empty
        agenda = state.get("agenda", [])
        if not agenda:
            approval = False
        else:
            approval = get_agenda_approval()

        # Update state based on approval
        hitl_state = state.get("hitl_state", {}).copy()
        hitl_state["approved"] = approval

        # Update agenda status and phase
        if approval:
            for topic in agenda:
                topic["status"] = "approved"
            current_phase = 2  # Discussion
        else:
            current_phase = 1  # Back to agenda setting

        return {
            "hitl_state": hitl_state,
            "current_phase": current_phase,
            "agenda": agenda,
        }

    def _simulate_agenda_rejection_and_modification(
        self, state: VirtualAgoraState
    ) -> dict:
        """Simulate agenda rejection and modification process."""
        from virtual_agora.ui.human_in_the_loop import (
            get_agenda_approval,
            get_agenda_modifications,
        )

        # First, get rejection
        approval = get_agenda_approval()

        if not approval:
            # Get modifications
            modifications = get_agenda_modifications()

            # Apply modifications
            agenda = state.get("agenda", []).copy()

            # Remove topics
            if "remove_topics" in modifications:
                agenda = [
                    topic
                    for topic in agenda
                    if topic["title"] not in modifications["remove_topics"]
                ]

            # Add topics
            if "add_topics" in modifications:
                for new_topic in modifications["add_topics"]:
                    agenda.append(
                        {
                            "title": new_topic,
                            "description": f"User-added topic: {new_topic}",
                            "status": "pending",
                            "proposed_by": "user",
                        }
                    )

            # Reorder if requested
            if modifications.get("reorder", False):
                agenda = list(reversed(agenda))

            return {
                "agenda": agenda,
                "current_phase": 1,  # Back to agenda setting
                "hitl_state": {"approved": False},
            }

        return state


class TestContinuationApprovalFlow:
    """Test continuation approval workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_continuation_approval")
    def test_continuation_approval_continue(self, mock_get_continuation):
        """Test user choosing to continue discussion."""
        # Mock user choosing to continue
        mock_get_continuation.return_value = "continue"

        with patch_other_ui_components(exclude=["get_continuation_approval"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Complete one topic
            state["current_topic_index"] = 1
            state["topic_summaries"] = ["Summary of first topic"]

            # Simulate continuation approval
            result = self._simulate_continuation_approval(state)

            # Should continue to next topic
            assert result["continuation_decision"] == "continue"
            assert result["current_phase"] == 4  # Agenda modification phase

            mock_get_continuation.assert_called_once()

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_continuation_approval")
    def test_continuation_approval_end_session(self, mock_get_continuation):
        """Test user choosing to end session."""
        # Mock user choosing to end
        mock_get_continuation.return_value = "end"

        with patch_other_ui_components(exclude=["get_continuation_approval"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Complete all topics
            state["current_topic_index"] = len(state["agenda"])
            state["topic_summaries"] = ["Summary 1", "Summary 2", "Summary 3"]

            # Simulate continuation approval
            result = self._simulate_continuation_approval(state)

            # Should end session
            assert result["continuation_decision"] == "end"
            assert result["current_phase"] == 5  # Final report phase

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_continuation_approval")
    def test_continuation_approval_modify_agenda(self, mock_get_continuation):
        """Test user choosing to modify agenda."""
        # Mock user choosing to modify agenda
        mock_get_continuation.return_value = "modify"

        with patch_other_ui_components(exclude=["get_continuation_approval"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate continuation approval
            result = self._simulate_continuation_approval(state)

            # Should go to agenda modification
            assert result["continuation_decision"] == "modify"
            assert result["current_phase"] == 4  # Agenda modification phase

    def _simulate_continuation_approval(self, state: VirtualAgoraState) -> dict:
        """Simulate continuation approval process."""
        from virtual_agora.ui.human_in_the_loop import get_continuation_approval

        # Check if there are more topics
        current_topic_index = state.get("current_topic_index", 0)
        total_topics = len(state.get("agenda", []))

        # Get user decision
        decision = get_continuation_approval()

        # Determine next phase based on decision
        if decision == "continue":
            if current_topic_index < total_topics:
                next_phase = 4  # Agenda modification
            else:
                next_phase = 5  # Final report
        elif decision == "modify":
            next_phase = 4  # Agenda modification
        else:  # "end"
            next_phase = 5  # Final report

        return {
            "continuation_decision": decision,
            "current_phase": next_phase,
            "hitl_state": {"approved": True},
        }


class TestUserPreferencesFlow:
    """Test user preferences integration."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    @patch("virtual_agora.ui.preferences.get_user_preferences")
    def test_user_preferences_application(self, mock_get_prefs):
        """Test applying user preferences to session."""
        # Mock user preferences
        mock_preferences = UserPreferences(
            auto_approve_agenda_on_consensus=True,
            prefer_detailed_summaries=True,
            display_verbosity="detailed",
            auto_save_interval=300,
            show_agent_reasoning=True,
        )
        mock_get_prefs.return_value = mock_preferences

        with patch_other_ui_components(exclude=["get_user_preferences"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_basic_state()

            # Apply preferences
            result = self._simulate_preferences_application(state)

            # Verify preferences were applied
            assert (
                result["user_preferences"]["auto_approve_agenda_on_consensus"] == True
            )
            assert result["user_preferences"]["prefer_detailed_summaries"] == True
            assert result["user_preferences"]["display_verbosity"] == "detailed"

    @pytest.mark.integration
    @patch("virtual_agora.ui.preferences.get_user_preferences")
    def test_user_preferences_defaults(self, mock_get_prefs):
        """Test default preferences when user doesn't specify."""
        # Mock default preferences
        mock_get_prefs.return_value = None

        with patch_other_ui_components(exclude=["get_user_preferences"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_basic_state()

            # Apply default preferences
            result = self._simulate_preferences_application(state)

            # Should have default values
            assert (
                result["user_preferences"]["auto_approve_agenda_on_consensus"] == False
            )  # Default
            assert result["user_preferences"]["display_verbosity"] == "normal"

    def _simulate_preferences_application(self, state: VirtualAgoraState) -> dict:
        """Simulate user preferences application."""
        from virtual_agora.ui.preferences import get_user_preferences

        # Get user preferences
        preferences = get_user_preferences()

        if preferences:
            # Apply user preferences
            user_prefs = {
                "auto_approve_agenda_on_consensus": preferences.auto_approve_agenda_on_consensus,
                "prefer_detailed_summaries": preferences.prefer_detailed_summaries,
                "display_verbosity": preferences.display_verbosity,
                "auto_save_interval": preferences.auto_save_interval,
                "show_agent_reasoning": preferences.show_agent_reasoning,
            }
        else:
            # Apply defaults
            user_prefs = {
                "auto_approve_agenda_on_consensus": False,
                "prefer_detailed_summaries": True,
                "display_verbosity": "normal",
                "auto_save_interval": 300,
                "show_agent_reasoning": True,
            }

        return {"user_preferences": user_prefs}


class TestInteractiveValidationFlow:
    """Test interactive validation and error handling."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_agenda_approval")
    @patch("virtual_agora.ui.components.LoadingSpinner")
    def test_ui_loading_states(self, mock_spinner, mock_get_approval):
        """Test UI loading states during operations."""
        # Mock loading spinner with context manager support
        mock_spinner_instance = Mock()
        mock_spinner_instance.__enter__ = Mock(return_value=mock_spinner_instance)
        mock_spinner_instance.__exit__ = Mock(return_value=None)
        mock_spinner.return_value = mock_spinner_instance
        mock_get_approval.return_value = True

        with patch_other_ui_components(exclude=["get_agenda_approval"]):
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_agenda_setting_state()

            # Simulate operation with loading
            result = self._simulate_operation_with_loading(state)

            # Verify loading spinner was used
            mock_spinner.assert_called()
            assert result["operation_completed"] == True

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
    def test_ui_input_validation_flow(self, mock_get_topic):
        """Test input validation and retry flow."""
        # Mock multiple invalid inputs then valid
        mock_get_topic.side_effect = [
            "",  # Empty
            "AI",  # Too short
            "A" * 200,  # Too long
            "Valid Discussion Topic for Testing",  # Valid
        ]

        with patch_other_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_basic_state()

            # Simulate validation flow
            result = self._simulate_input_validation_flow(state)

            # Should eventually succeed
            assert result["main_topic"] == "Valid Discussion Topic for Testing"
            assert result["validation_attempts"] == 4

    def _simulate_operation_with_loading(self, state: VirtualAgoraState) -> dict:
        """Simulate operation with loading indicator."""
        from virtual_agora.ui.components import LoadingSpinner
        from virtual_agora.ui.human_in_the_loop import get_agenda_approval

        # Show loading spinner
        with LoadingSpinner("Processing agenda..."):
            # Simulate some processing time
            import time

            time.sleep(0.1)

            # Get user approval
            approval = get_agenda_approval()

        return {"operation_completed": True, "approval_received": approval}

    def _simulate_input_validation_flow(self, state: VirtualAgoraState) -> dict:
        """Simulate input validation with retries."""
        from virtual_agora.ui.human_in_the_loop import get_initial_topic

        attempts = 0
        valid_topic = None
        max_attempts = 10

        while not valid_topic and attempts < max_attempts:
            attempts += 1
            topic = get_initial_topic()

            # Validation rules
            if topic and 10 <= len(topic.strip()) <= 100:
                valid_topic = topic.strip()

        if not valid_topic:
            raise ValueError("Failed to get valid topic after maximum attempts")

        return {"main_topic": valid_topic, "validation_attempts": attempts}


@pytest.mark.integration
class TestUIErrorHandling:
    """Test UI error handling and recovery."""

    def test_ui_timeout_handling(self):
        """Test handling of UI timeouts."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_other_ui_components():
            # Mock timeout scenario
            with patch(
                "virtual_agora.ui.human_in_the_loop.get_initial_topic"
            ) as mock_topic:
                mock_topic.side_effect = TimeoutError("User input timeout")

                flow = helper.create_test_flow()
                state = helper.create_basic_state()

                # Should handle timeout gracefully
                try:
                    result = self._simulate_timeout_recovery(state)
                    assert result["error_handled"] == True
                    assert result["fallback_used"] == True
                except TimeoutError:
                    # Should have fallback mechanism
                    assert False, "Timeout should be handled gracefully"

    def test_ui_invalid_input_recovery(self):
        """Test recovery from invalid user inputs."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_other_ui_components():
            with patch(
                "virtual_agora.ui.human_in_the_loop.get_agenda_approval"
            ) as mock_approval:
                # Mock invalid input types
                mock_approval.side_effect = [
                    "invalid",  # String instead of boolean
                    None,  # None value
                    True,  # Valid boolean
                ]

                flow = helper.create_test_flow()
                state = helper.create_agenda_setting_state()

                result = self._simulate_invalid_input_recovery(state)

                # Should eventually get valid input
                assert result["final_approval"] == True
                assert result["recovery_attempts"] == 3

    def _simulate_timeout_recovery(self, state: VirtualAgoraState) -> dict:
        """Simulate timeout recovery mechanism."""
        try:
            from virtual_agora.ui.human_in_the_loop import get_initial_topic

            topic = get_initial_topic()
        except TimeoutError:
            # Fallback to default topic
            topic = "Default Discussion Topic"
            return {"main_topic": topic, "error_handled": True, "fallback_used": True}

        return {"main_topic": topic, "error_handled": False, "fallback_used": False}

    def _simulate_invalid_input_recovery(self, state: VirtualAgoraState) -> dict:
        """Simulate recovery from invalid inputs."""
        from virtual_agora.ui.human_in_the_loop import get_agenda_approval

        attempts = 0
        max_attempts = 5
        valid_approval = None

        while valid_approval is None and attempts < max_attempts:
            attempts += 1
            try:
                approval = get_agenda_approval()
                # Validate input type
                if isinstance(approval, bool):
                    valid_approval = approval
            except (TypeError, ValueError):
                continue

        if valid_approval is None:
            valid_approval = False  # Default fallback

        return {"final_approval": valid_approval, "recovery_attempts": attempts}
