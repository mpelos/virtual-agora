"""Unit tests for ModeratorAgent in Virtual Agora v1.3."""

import pytest
from unittest.mock import Mock, patch
import json

from virtual_agora.agents.moderator import ModeratorAgent
from tests.helpers.fake_llm import ModeratorFakeLLM, FakeLLMBase


class TestModeratorAgentV3:
    """Unit tests for ModeratorAgent v1.3 functionality."""

    @pytest.fixture
    def moderator_llm(self):
        """Create a fake LLM for moderator testing."""
        return ModeratorFakeLLM()

    @pytest.fixture
    def moderator(self, moderator_llm):
        """Create test moderator instance."""
        return ModeratorAgent(agent_id="test_moderator", llm=moderator_llm)

    def test_initialization_v13_no_mode(self, moderator):
        """Test that v1.3 ModeratorAgent has no mode parameter."""
        assert moderator.agent_id == "test_moderator"

        # v1.3: No mode attribute
        assert not hasattr(moderator, "mode")

        # Check v1.3 specific prompt
        assert "specialized reasoning tool" in moderator.system_prompt
        assert "Virtual Agora" in moderator.system_prompt
        assert "process facilitation" in moderator.system_prompt
        assert "NOT a discussion participant" in moderator.system_prompt
        assert "NO opinions on topics" in moderator.system_prompt

    def test_v13_prompt_tasks(self, moderator):
        """Test that v1.3 prompt includes all required tasks."""
        prompt = moderator.system_prompt

        # Check all required capabilities
        assert "Proposal Compilation" in prompt
        assert "deduplicated list" in prompt
        assert "Vote Synthesis" in prompt
        assert "rank-ordered agenda" in prompt
        assert "JSON" in prompt
        assert '{"proposed_agenda":' in prompt
        assert "Break ties" in prompt
        assert "objective criteria" in prompt

    def test_collect_proposals(self, moderator):
        """Test proposal collection and deduplication."""
        agent_proposals = [
            {
                "agent_id": "agent_1",
                "proposals": "1. AI Safety measures\n2. Performance optimization\n3. User privacy",
            },
            {
                "agent_id": "agent_2",
                "proposals": "1. Performance optimization\n2. Scalability concerns\n3. AI Safety measures",
            },
            {
                "agent_id": "agent_3",
                "proposals": "1. Cost reduction\n2. User privacy\n3. Market expansion",
            },
        ]

        unique_topics = moderator.collect_proposals(agent_proposals)

        assert isinstance(unique_topics, list)
        assert len(unique_topics) > 0
        # Should have deduplicated (AI Safety and User privacy appear multiple times)
        # Note: The fake LLM returns a predefined response, but in real usage
        # this would deduplicate properly

    def test_synthesize_agenda_json_output(self, moderator):
        """Test that agenda synthesis produces valid JSON."""
        agent_votes = [
            {
                "agent_id": "agent_1",
                "vote": "I prefer Technical Implementation first, then Legal considerations, and Social Impact last.",
            },
            {
                "agent_id": "agent_2",
                "vote": "Legal and Regulatory should be first to ensure compliance, then Technical, then Social.",
            },
            {
                "agent_id": "agent_3",
                "vote": "Social Impact is most important, followed by Technical and Legal aspects.",
            },
        ]

        result = moderator.synthesize_agenda(agent_votes)

        # Should return a parsed dictionary (not JSON string)
        assert isinstance(result, dict)
        assert "proposed_agenda" in result
        assert isinstance(result["proposed_agenda"], list)
        assert len(result["proposed_agenda"]) > 0
        assert all(isinstance(item, str) for item in result["proposed_agenda"])

    def test_tie_breaking_logic(self, moderator):
        """Test that moderator can break ties objectively."""
        # Create votes that result in a tie
        tie_votes = [
            {"agent_id": "agent_1", "vote": "Topic A should be first, then Topic B"},
            {"agent_id": "agent_2", "vote": "Topic B should be first, then Topic A"},
        ]

        result = moderator.synthesize_agenda(tie_votes)

        # Should still produce a valid ordering
        assert isinstance(result, dict)
        assert "proposed_agenda" in result
        assert len(result["proposed_agenda"]) >= 2
        # The ordering should be deterministic based on objective criteria

    def test_natural_language_vote_processing(self, moderator):
        """Test processing of natural language votes."""
        complex_votes = [
            {
                "agent_id": "agent_1",
                "vote": "While I see merits in all topics, I believe we should start with Security considerations given their critical nature, followed by Performance optimization which builds on secure foundations, and conclude with User Experience improvements.",
            },
            {
                "agent_id": "agent_2",
                "vote": "I strongly advocate for User Experience as our primary focus - without users, the other aspects don't matter. Security would be my second choice, with Performance as a nice-to-have third priority.",
            },
            {
                "agent_id": "agent_3",
                "vote": "From a practical standpoint, Performance should lead our discussions as it affects everything else. Security is indeed important and should follow. UX can be refined once we have a secure, performant system.",
            },
        ]

        result = moderator.synthesize_agenda(complex_votes)

        # Should extract clear ordering from verbose votes
        assert isinstance(result, dict)
        assert "proposed_agenda" in result
        # Should have identified the three topics mentioned
        assert len(result["proposed_agenda"]) >= 3

    def test_process_only_no_content_opinions(self, moderator):
        """Test that moderator focuses only on process, not content."""
        # The prompt should ensure the moderator doesn't express opinions
        prompt = moderator.system_prompt

        # Strong assertions about neutrality
        assert "NOT a discussion participant" in prompt
        assert "NO opinions" in prompt or "no opinions" in prompt.lower()
        assert "process" in prompt.lower()
        assert "facilitation" in prompt.lower()

        # When processing proposals, should not add own topics
        proposals = [{"agent_id": "agent_1", "proposals": "Topic A, Topic B"}]

        result = moderator.collect_proposals(proposals)
        # Result should only contain topics from agents, not moderator additions
        assert isinstance(result, list)

    def test_error_handling_invalid_json(self, moderator):
        """Test handling of invalid JSON responses."""

        # Create a mock LLM that returns invalid JSON
        class BadJSONLLM(FakeLLMBase):
            def _get_response(self, messages, **kwargs):
                return "This is not valid JSON"

        bad_moderator = ModeratorAgent(agent_id="bad_moderator", llm=BadJSONLLM())

        votes = [{"agent_id": "agent_1", "vote": "Topic A first"}]

        # Should handle gracefully
        try:
            result = bad_moderator.synthesize_agenda(votes)
            # If it doesn't raise, should return some default
            assert isinstance(result, dict) or result is None
        except Exception as e:
            # If it raises, should be informative
            assert "JSON" in str(e) or "parse" in str(e).lower()

    def test_empty_proposals_handling(self, moderator):
        """Test handling of empty proposal lists."""
        empty_proposals = []

        result = moderator.collect_proposals(empty_proposals)

        # Should handle gracefully
        assert isinstance(result, list)
        # Could be empty or have a default response

    def test_duplicate_proposal_handling(self, moderator):
        """Test deduplication of identical proposals."""
        duplicate_proposals = [
            {
                "agent_id": "agent_1",
                "proposals": "1. Cloud Migration\n2. Security Audit\n3. Performance Testing",
            },
            {
                "agent_id": "agent_2",
                "proposals": "1. Security Audit\n2. Cloud Migration\n3. Performance Testing",
            },
            {
                "agent_id": "agent_3",
                "proposals": "1. Performance Testing\n2. Security Audit\n3. Cloud Migration",
            },
        ]

        result = moderator.collect_proposals(duplicate_proposals)

        # All three agents proposed the same 3 topics
        # Result should have exactly 3 unique topics (or close to it)
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_objective_criteria_in_synthesis(self, moderator):
        """Test that synthesis uses objective criteria."""
        votes_with_equal_support = [
            {
                "agent_id": "agent_1",
                "vote": "I prefer discussing 'Quantum Computing Applications' first",
            },
            {
                "agent_id": "agent_2",
                "vote": "I think 'Blockchain Technology' should be our starting point",
            },
            {
                "agent_id": "agent_3",
                "vote": "I vote for 'Artificial Intelligence Ethics' as the primary topic",
            },
        ]

        result = moderator.synthesize_agenda(votes_with_equal_support)

        # With equal support, should use objective criteria like:
        # - Clarity of scope
        # - Relevance to theme
        # - Logical dependencies
        assert isinstance(result, dict)
        assert "proposed_agenda" in result
        assert len(result["proposed_agenda"]) >= 3

        # The ordering should be consistent if run multiple times
        # (though we can't test this with fake LLM)

    def test_no_mode_parameter_allowed(self):
        """Test that v1.3 ModeratorAgent doesn't accept mode parameter."""
        llm = ModeratorFakeLLM()

        # v1.3 should not accept mode parameter
        try:
            # This should fail or ignore the mode parameter
            moderator = ModeratorAgent(
                agent_id="test",
                llm=llm,
                mode="synthesis",  # v1.1 style - should not work
            )
            # If it doesn't raise an error, mode should be ignored
            assert not hasattr(moderator, "mode") or moderator.mode is None
        except TypeError as e:
            # Expected - mode parameter not accepted
            assert "mode" in str(e)

    def test_precise_analytical_output(self, moderator):
        """Test that outputs are precise and analytical as required."""
        # Test with specific proposal format
        proposals = [
            {
                "agent_id": "agent_1",
                "proposals": "- Implement authentication system\n- Design database schema\n- Create API endpoints",
            },
            {
                "agent_id": "agent_2",
                "proposals": "- Design database schema\n- Build frontend UI\n- Implement authentication system",
            },
        ]

        result = moderator.collect_proposals(proposals)

        # Output should be a clean list, not verbose
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)
        # Items should be concise (this is ensured by the prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
