"""Debug test for UI integration."""

import pytest
from unittest.mock import patch, Mock

from ..helpers.integration_utils import IntegrationTestHelper, patch_ui_components


class TestUIDebug:
    """Debug UI integration issues."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
    def test_simple_mock(self, mock_get_topic):
        """Test simple mock behavior."""
        mock_get_topic.return_value = "Mocked Topic"

        from virtual_agora.ui.human_in_the_loop import get_initial_topic

        result = get_initial_topic()

        assert result == "Mocked Topic"
        mock_get_topic.assert_called_once()

    @pytest.mark.integration
    @patch("virtual_agora.ui.human_in_the_loop.get_initial_topic")
    def test_validation_flow_mock(self, mock_get_topic):
        """Test validation flow with mock."""
        mock_get_topic.side_effect = [
            "",  # Empty topic (invalid)
            "AI",  # Too short (invalid)
            "Climate Change Policy and Its Global Impact",  # Valid
        ]

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

        assert main_topic == "Climate Change Policy and Its Global Impact"
        assert mock_get_topic.call_count == 3
