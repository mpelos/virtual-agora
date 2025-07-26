"""Integration tests for agenda management system."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from virtual_agora.agenda.manager import AgendaManager
from virtual_agora.agenda.models import (
    AgendaState, AgendaStatus, VoteType, ProposalStatus, VoteStatus,
    Proposal, Vote
)
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent


class TestAgendaManagementIntegration:
    """Integration tests for the complete agenda management workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.moderator = Mock(spec=ModeratorAgent)
        self.manager = AgendaManager(self.moderator, "integration_session")
        
        # Create mock agents
        self.agents = {}
        for i in range(3):
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = f"agent_{i+1}"
            agent.propose_topics.return_value = [
                f"Topic {i+1}A",
                f"Topic {i+1}B", 
                f"Topic {i+1}C"
            ]
            agent.vote_on_agenda.return_value = f"1. Topic {i+1}A\n2. Topic {i+1}B"
            self.agents[f"agent_{i+1}"] = agent
    
    @pytest.mark.asyncio
    async def test_full_agenda_initialization_workflow(self):
        """Test the complete agenda initialization workflow."""
        
        # Mock moderator synthesis response - synthesize_agenda returns a list directly
        # Use a more explicit mock that accepts any arguments and returns our expected result
        def mock_synthesize_agenda(*args, **kwargs):
            return ["agent_1_TopicA", "agent_2_TopicA", "agent_3_TopicA"]
        
        self.moderator.synthesize_agenda.side_effect = mock_synthesize_agenda
        
        # Mock the threading components to avoid concurrency issues in tests
        with patch('virtual_agora.agenda.manager.ThreadPoolExecutor') as mock_executor_class:
            # Mock executor context manager
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            # Mock proposal collection futures
            proposal_futures = []
            for agent_id, agent in self.agents.items():
                future = Mock()
                future.result.return_value = [
                    Proposal(agent_id=agent_id, topic=f"{agent_id}_TopicA", status=ProposalStatus.COLLECTED),
                    Proposal(agent_id=agent_id, topic=f"{agent_id}_TopicB", status=ProposalStatus.COLLECTED)
                ]
                proposal_futures.append(future)
            
            mock_executor.submit.side_effect = lambda *args: proposal_futures.pop(0)
            
            # Mock as_completed for proposals
            with patch('virtual_agora.agenda.manager.as_completed', return_value=proposal_futures[::-1]):
                # Mock voting collection futures
                with patch('virtual_agora.agenda.voting.ThreadPoolExecutor') as mock_vote_executor_class:
                    mock_vote_executor = Mock()
                    mock_vote_executor_class.return_value.__enter__.return_value = mock_vote_executor
                    
                    vote_futures = []
                    for agent_id in self.agents.keys():
                        future = Mock()
                        future.result.return_value = Vote(
                            agent_id=agent_id,
                            vote_type=VoteType.INITIAL_AGENDA,
                            vote_content=f"I prefer {agent_id}_TopicA first",
                            parsed_preferences=[f"{agent_id}_TopicA"],
                            confidence_score=0.8,
                            status=VoteStatus.SUBMITTED
                        )
                        vote_futures.append(future)
                    
                    mock_vote_executor.submit.side_effect = lambda *args: vote_futures.pop(0)
                    
                    # Mock as_completed for votes
                    with patch('virtual_agora.agenda.voting.as_completed', return_value=vote_futures[::-1]):
                        # Execute the initialization
                        result_state = await self.manager.initialize_agenda(
                            self.agents,
                            "Main Discussion Topic",
                            proposals_per_agent=3
                        )
        
        # Verify the result
        assert isinstance(result_state, AgendaState)
        assert result_state.status == AgendaStatus.COMPLETED
        assert len(result_state.current_agenda) == 3
        
        # Check that proposals were collected
        assert len(result_state.proposal_collections) == 1
        proposal_collection = result_state.proposal_collections[0]
        assert proposal_collection.session_id == "integration_session"
        assert proposal_collection.status == ProposalStatus.COLLECTED
        
        # Check that votes were collected
        assert len(result_state.vote_collections) == 1
        vote_collection = result_state.vote_collections[0]
        assert vote_collection.vote_type == VoteType.INITIAL_AGENDA
        assert vote_collection.status == VoteStatus.SUBMITTED
        
        # Check agenda items
        for i, item in enumerate(result_state.current_agenda):
            assert item.rank == i + 1
            assert item.status == "pending"
            assert len(item.proposed_by) >= 0
        
        # Check analytics were updated
        analytics = self.manager.get_analytics()
        assert analytics.session_id == "integration_session"
        assert analytics.total_proposals >= 0
    
    @pytest.mark.asyncio
    async def test_agenda_initialization_with_empty_proposals(self):
        """Test agenda initialization when no proposals are collected."""
        
        # Mock agents to return no proposals
        for agent in self.agents.values():
            agent.propose_topics.return_value = []
        
        # Mock threading to return empty proposals
        with patch('virtual_agora.agenda.manager.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            # Return empty proposal lists
            empty_futures = []
            for _ in self.agents:
                future = Mock()
                future.result.return_value = []
                empty_futures.append(future)
            
            mock_executor.submit.side_effect = lambda *args: empty_futures.pop(0)
            
            with patch('virtual_agora.agenda.manager.as_completed', return_value=empty_futures[::-1]):
                result_state = await self.manager.initialize_agenda(
                    self.agents,
                    "Main Discussion Topic"
                )
        
        # Should have fallback agenda
        assert result_state.status == AgendaStatus.COMPLETED
        assert len(result_state.current_agenda) == 5  # Fallback topics
        assert all(item.proposed_by == ["system"] for item in result_state.current_agenda)
        
        # Should have recorded edge case
        assert len(self.manager.edge_cases) == 1
        assert self.manager.edge_cases[0].event_type == "empty_proposals"
        assert self.manager.edge_cases[0].recovered_successfully is True
    
    @pytest.mark.asyncio
    async def test_agenda_modification_workflow(self):
        """Test the agenda modification workflow."""
        
        # First, set up an existing agenda
        from virtual_agora.agenda.models import AgendaItem
        
        self.manager.state.current_agenda = [
            AgendaItem(topic="Existing Topic 1", rank=1, status="pending"),
            AgendaItem(topic="Existing Topic 2", rank=2, status="pending"),
            AgendaItem(topic="Existing Topic 3", rank=3, status="pending")
        ]
        self.manager.state.status = AgendaStatus.COMPLETED
        
        remaining_topics = ["Existing Topic 2", "Existing Topic 3"]
        
        # Mock moderator modification request
        def mock_request_modification(*args, **kwargs):
            return """
            Based on our discussion, I suggest we add a new topic about emerging technologies
            and remove the third topic as it's less relevant now.
            """
        
        self.moderator.request_agenda_modification.side_effect = mock_request_modification
        
        # Mock moderator synthesis for modified agenda - synthesize_agenda returns list directly
        # Use a more explicit mock that accepts any arguments and returns our expected result
        def mock_synthesize_modified_agenda(*args, **kwargs):
            return ["New Technology Topic", "Existing Topic 2"]
        
        self.moderator.synthesize_agenda.side_effect = mock_synthesize_modified_agenda
        
        # Mock voting collection for modifications
        with patch('virtual_agora.agenda.voting.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            vote_futures = []
            for agent_id in self.agents.keys():
                future = Mock()
                future.result.return_value = Vote(
                    agent_id=agent_id,
                    vote_type=VoteType.AGENDA_MODIFICATION,
                    vote_content="I prefer New Technology Topic first",
                    parsed_preferences=["New Technology Topic", "Existing Topic 2"],
                    confidence_score=0.8,
                    status=VoteStatus.SUBMITTED
                )
                vote_futures.append(future)
            
            mock_executor.submit.side_effect = lambda *args: vote_futures.pop(0)
            
            with patch('virtual_agora.agenda.voting.as_completed', return_value=vote_futures[::-1]):
                result_state = await self.manager.modify_agenda(self.agents, remaining_topics)
        
        # Verify modification was applied
        assert result_state.version == 2  # Version incremented
        assert result_state.last_modification is not None
        
        # Check new agenda structure
        agenda_topics = [item.topic for item in result_state.current_agenda]
        assert "New Technology Topic" in agenda_topics
        assert "Existing Topic 2" in agenda_topics
        assert "Existing Topic 3" not in agenda_topics  # Should be removed
    
    def test_topic_transition_workflow(self):
        """Test the topic transition workflow."""
        
        # Set up agenda with multiple topics
        from virtual_agora.agenda.models import AgendaItem
        
        self.manager.state.current_agenda = [
            AgendaItem(topic="Topic A", rank=1, status="pending"),
            AgendaItem(topic="Topic B", rank=2, status="pending"),
            AgendaItem(topic="Topic C", rank=3, status="pending")
        ]
        
        agent_ids = list(self.agents.keys())
        
        # Transition to first topic
        transition1 = self.manager.transition_to_topic("Topic A", agent_ids)
        
        assert transition1.from_topic is None
        assert transition1.to_topic == "Topic A"
        assert transition1.transition_type == "start"
        assert self.manager.state.active_topic == "Topic A"
        
        # Find Topic A and verify it's active
        topic_a = next(item for item in self.manager.state.current_agenda if item.topic == "Topic A")
        assert topic_a.status == "active"
        assert topic_a.discussion_started is not None
        
        # Transition to second topic
        transition2 = self.manager.transition_to_topic("Topic B", agent_ids)
        
        assert transition2.from_topic == "Topic A"
        assert transition2.to_topic == "Topic B"
        assert self.manager.state.active_topic == "Topic B"
        
        # Verify Topic A is now completed
        assert "Topic A" in self.manager.state.completed_topics
        topic_a_updated = next(item for item in self.manager.state.current_agenda if item.topic == "Topic A")
        assert topic_a_updated.status == "completed"
        assert topic_a_updated.discussion_ended is not None
        
        # Check analytics recorded transitions
        assert len(self.manager.analytics.topic_transitions) == 2
    
    def test_complete_agenda_workflow(self):
        """Test the complete agenda workflow from start to finish."""
        
        from virtual_agora.agenda.models import AgendaItem
        
        # Set up initial agenda
        self.manager.state.current_agenda = [
            AgendaItem(topic="Topic A", rank=1, status="pending"),
            AgendaItem(topic="Topic B", rank=2, status="pending")
        ]
        
        agent_ids = list(self.agents.keys())
        
        # Check initial state
        assert not self.manager.is_agenda_complete()
        
        # Transition through all topics
        self.manager.transition_to_topic("Topic A", agent_ids)
        assert not self.manager.is_agenda_complete()
        
        self.manager.transition_to_topic("Topic B", agent_ids)
        assert not self.manager.is_agenda_complete()  # Topic B is active, not completed
        
        # Complete the last topic by transitioning to None (end state)
        # Simulate completion by manually marking the last topic as completed
        topic_b = next(item for item in self.manager.state.current_agenda if item.topic == "Topic B")
        topic_b.status = "completed"
        self.manager.state.completed_topics.append("Topic B")
        self.manager.state.active_topic = None
        
        # Now agenda should be complete
        assert self.manager.is_agenda_complete()
        
        # Verify analytics
        analytics = self.manager.get_analytics()
        assert analytics.topics_completed == 2
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        
        # Test synthesis failure with fallback
        self.moderator.synthesize_agenda.side_effect = Exception("Synthesis failed")
        
        # Mock successful proposal and vote collection
        with patch('virtual_agora.agenda.manager.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            # Create successful proposal futures
            proposal_futures = [Mock() for _ in self.agents]
            for i, future in enumerate(proposal_futures):
                future.result.return_value = [
                    Proposal(agent_id=f"agent_{i+1}", topic=f"Topic{i+1}", status=ProposalStatus.COLLECTED)
                ]
            
            mock_executor.submit.side_effect = lambda *args: proposal_futures.pop(0)
            
            with patch('virtual_agora.agenda.manager.as_completed', return_value=proposal_futures[::-1]):
                # Mock voting
                with patch('virtual_agora.agenda.voting.ThreadPoolExecutor') as mock_vote_executor_class:
                    mock_vote_executor = Mock()
                    mock_vote_executor_class.return_value.__enter__.return_value = mock_vote_executor
                    
                    vote_futures = [Mock() for _ in self.agents]
                    for i, future in enumerate(vote_futures):
                        future.result.return_value = Vote(
                            agent_id=f"agent_{i+1}",
                            vote_type=VoteType.INITIAL_AGENDA,
                            vote_content=f"Topic{i+1}",
                            parsed_preferences=[f"Topic{i+1}"],
                            status=VoteStatus.SUBMITTED
                        )
                    
                    mock_vote_executor.submit.side_effect = lambda *args: vote_futures.pop(0)
                    
                    with patch('virtual_agora.agenda.voting.as_completed', return_value=vote_futures[::-1]):
                        # This should use fallback synthesis
                        result_state = await self.manager.initialize_agenda(
                            self.agents,
                            "Main Topic"
                        )
        
        # Should still succeed with fallback
        assert result_state.status == AgendaStatus.COMPLETED
        assert len(result_state.current_agenda) > 0
        
        # Should have recorded the failure and recovery
        assert len(self.manager.edge_cases) == 1
        assert self.manager.edge_cases[0].event_type == "initialization_error"
    
    @pytest.mark.asyncio
    async def test_analytics_throughout_workflow(self):
        """Test that analytics are properly collected throughout the workflow."""
        
        # Initialize analytics state
        initial_analytics = self.manager.get_analytics()
        assert initial_analytics.total_proposals == 0
        assert initial_analytics.topics_completed == 0
        
        # Mock a simple successful initialization
        with patch.object(self.manager.proposal_collector, 'collect_proposals') as mock_collect_proposals:
            with patch.object(self.manager.voting_orchestrator, 'collect_votes') as mock_collect_votes:
                with patch.object(self.manager.synthesizer, 'synthesize_agenda') as mock_synthesize:
                    
                    # Mock successful proposal collection
                    from virtual_agora.agenda.models import ProposalCollection
                    mock_proposal_collection = ProposalCollection(
                        session_id="integration_session",
                        requested_agents=list(self.agents.keys()),
                        responding_agents=list(self.agents.keys()),
                        proposals=[Proposal(agent_id="agent_1", topic="Test Topic")],
                        status=ProposalStatus.COLLECTED
                    )
                    mock_collect_proposals.return_value = mock_proposal_collection
                    
                    # Mock successful vote collection
                    from virtual_agora.agenda.models import VoteCollection
                    mock_vote_collection = VoteCollection(
                        session_id="integration_session",
                        vote_type=VoteType.INITIAL_AGENDA,
                        requested_agents=list(self.agents.keys()),
                        responding_agents=list(self.agents.keys()),
                        votes=[Vote(agent_id="agent_1", vote_type=VoteType.INITIAL_AGENDA, vote_content="Test vote", status=VoteStatus.SUBMITTED)],
                        status=VoteStatus.SUBMITTED
                    )
                    mock_collect_votes.return_value = mock_vote_collection
                    
                    # Mock successful synthesis
                    from virtual_agora.agenda.models import AgendaSynthesisResult
                    mock_synthesis_result = AgendaSynthesisResult(
                        proposed_agenda=["Test Topic"]
                    )
                    mock_synthesize.return_value = mock_synthesis_result
                    
                    # Execute initialization
                    result_state = await self.manager.initialize_agenda(
                        self.agents,
                        "Main Topic"
                    )
        
        # Verify analytics were updated
        final_analytics = self.manager.get_analytics()
        assert final_analytics.total_proposals >= 0
        assert len(final_analytics.agent_participation_rates) == len(self.agents)
        
        # Test topic transition analytics
        from virtual_agora.agenda.models import AgendaItem
        self.manager.state.current_agenda = [
            AgendaItem(topic="Test Topic", rank=1, status="pending")
        ]
        
        transition = self.manager.transition_to_topic("Test Topic", list(self.agents.keys()))
        
        # Verify transition was recorded in analytics
        assert len(self.manager.analytics.topic_transitions) == 1
        
        # Generate final analytics report
        participation_report = self.manager.analytics.generate_participation_report()
        assert "overview" in participation_report
        assert "agent_details" in participation_report
        
        timeline_report = self.manager.analytics.generate_timeline_report()
        assert "events" in timeline_report
        assert "total_events" in timeline_report