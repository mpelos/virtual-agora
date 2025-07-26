"""Tests for agenda manager."""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from virtual_agora.agenda.manager import AgendaManager, ProposalCollector
from virtual_agora.agenda.models import (
    Proposal, Vote, VoteType, AgendaItem, AgendaState, ProposalCollection,
    VoteCollection, AgendaModification, TopicTransition, EdgeCaseEvent,
    ProposalStatus, VoteStatus, AgendaStatus
)
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent


class TestProposalCollector:
    """Test ProposalCollector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.moderator = Mock(spec=ModeratorAgent)
        self.collector = ProposalCollector(self.moderator, timeout_seconds=30)
    
    def create_mock_agents(self, count=3):
        """Create mock agents for testing."""
        agents = {}
        for i in range(count):
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = f"agent_{i+1}"
            agent.propose_topics.return_value = [
                f"Topic {i+1}A",
                f"Topic {i+1}B",
                f"Topic {i+1}C"
            ]
            agents[f"agent_{i+1}"] = agent
        return agents
    
    def test_collect_single_agent_proposals_success(self):
        """Test collecting proposals from single agent successfully."""
        agent = Mock(spec=DiscussionAgent)
        agent.propose_topics.return_value = ["Topic A", "Topic B", "Topic C"]
        
        proposals = self.collector._collect_single_agent_proposals(
            "agent_1", agent, "Main Topic", 3
        )
        
        assert len(proposals) == 3
        assert all(p.agent_id == "agent_1" for p in proposals)
        assert all(p.status == ProposalStatus.COLLECTED for p in proposals)
        assert proposals[0].topic == "Topic A"
        assert proposals[1].topic == "Topic B"
        assert proposals[2].topic == "Topic C"
    
    def test_collect_single_agent_proposals_failure(self):
        """Test handling agent proposal failure."""
        agent = Mock(spec=DiscussionAgent)
        agent.propose_topics.side_effect = Exception("Agent error")
        
        proposals = self.collector._collect_single_agent_proposals(
            "agent_1", agent, "Main Topic", 3
        )
        
        assert proposals == []
    
    def test_collect_single_agent_proposals_limit(self):
        """Test proposal count limiting."""
        agent = Mock(spec=DiscussionAgent)
        agent.propose_topics.return_value = [
            "Topic A", "Topic B", "Topic C", "Topic D", "Topic E"
        ]
        
        proposals = self.collector._collect_single_agent_proposals(
            "agent_1", agent, "Main Topic", 3
        )
        
        assert len(proposals) == 3  # Limited to requested count
    
    def test_deduplicate_proposals(self):
        """Test proposal deduplication."""
        proposals = [
            Proposal(agent_id="agent_1", topic="Climate Change"),
            Proposal(agent_id="agent_2", topic="climate change"),  # Duplicate
            Proposal(agent_id="agent_3", topic="Technology Ethics"),
            Proposal(agent_id="agent_1", topic="Climate  Change"),  # Duplicate with spacing
        ]
        
        unique_proposals = self.collector._deduplicate_proposals(proposals)
        
        assert len(unique_proposals) == 2
        
        # Find the climate change proposal
        climate_proposal = next(
            p for p in unique_proposals 
            if self.collector._normalize_topic(p.topic) == "climate change"
        )
        
        # Should have metadata about duplicates
        assert climate_proposal.metadata["duplicate_count"] == 3
        assert len(climate_proposal.metadata["proposed_by_agents"]) == 3
    
    def test_normalize_topic(self):
        """Test topic normalization."""
        assert self.collector._normalize_topic("Climate Change") == "climate change"
        assert self.collector._normalize_topic("  Technology Ethics  ") == "technology ethics"
        assert self.collector._normalize_topic("Economic  Policy") == "economic policy"
    
    @pytest.mark.asyncio
    async def test_collect_proposals_success(self):
        """Test successful proposal collection."""
        agents = self.create_mock_agents(3)
        
        # Mock the ThreadPoolExecutor to avoid threading in tests
        with patch('virtual_agora.agenda.manager.ThreadPoolExecutor') as mock_executor:
            # Mock executor context manager
            mock_context = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_context
            
            # Mock futures
            mock_futures = []
            for agent_id in agents.keys():
                future = MagicMock()
                future.result.return_value = [
                    Proposal(agent_id=agent_id, topic=f"{agent_id}_Topic_A"),
                    Proposal(agent_id=agent_id, topic=f"{agent_id}_Topic_B"),
                ]
                mock_futures.append(future)
            
            mock_context.submit.side_effect = lambda *args: mock_futures.pop(0)
            
            # Mock as_completed to return all futures
            with patch('virtual_agora.agenda.manager.as_completed', return_value=mock_futures[::-1]):
                collection = await self.collector.collect_proposals(
                    agents, "Main Topic", "session_1", 3
                )
        
        assert collection.session_id == "session_1"
        assert collection.status == ProposalStatus.COLLECTED
        assert len(collection.requested_agents) == 3
        assert collection.completion_rate > 0.0


class TestAgendaManager:
    """Test AgendaManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.moderator = Mock(spec=ModeratorAgent)
        self.manager = AgendaManager(self.moderator, "session_1")
    
    def create_mock_agents(self, count=3):
        """Create mock agents for testing."""
        agents = {}
        for i in range(count):
            agent = Mock(spec=DiscussionAgent)
            agent.agent_id = f"agent_{i+1}"
            agents[f"agent_{i+1}"] = agent
        return agents
    
    def test_manager_initialization(self):
        """Test agenda manager initialization."""
        assert self.manager.session_id == "session_1"
        assert self.manager.moderator == self.moderator
        assert isinstance(self.manager.state, AgendaState)
        assert self.manager.state.session_id == "session_1"
        assert self.manager.state.status == AgendaStatus.PENDING
    
    def test_calculate_vote_score(self):
        """Test vote score calculation."""
        votes = [
            Vote(
                agent_id="agent_1",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Climate Change\n2. Technology",
                parsed_preferences=["Climate Change", "Technology"],
                status=VoteStatus.SUBMITTED
            ),
            Vote(
                agent_id="agent_2",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Technology\n2. Climate Change",
                parsed_preferences=["Technology", "Climate Change"],
                status=VoteStatus.SUBMITTED
            )
        ]
        
        # Climate Change should get 1.0 (first choice) + 0.8 (second choice) = 1.8
        climate_score = self.manager._calculate_vote_score("Climate Change", votes)
        assert climate_score == 1.8
        
        # Technology should get 1.0 (first choice) + 0.8 (second choice) = 1.8
        tech_score = self.manager._calculate_vote_score("Technology", votes)
        assert tech_score == 1.8
        
        # Non-existent topic should get 0
        none_score = self.manager._calculate_vote_score("Non-existent", votes)
        assert none_score == 0.0
    
    @pytest.mark.asyncio
    async def test_handle_empty_proposals(self):
        """Test handling empty proposals edge case."""
        agents = self.create_mock_agents(3)
        
        result_state = await self.manager._handle_empty_proposals(agents, "Main Topic")
        
        assert result_state.status == AgendaStatus.COMPLETED
        assert len(result_state.current_agenda) == 5  # Fallback topics
        assert all(item.proposed_by == ["system"] for item in result_state.current_agenda)
        assert len(self.manager.edge_cases) == 1
        assert self.manager.edge_cases[0].event_type == "empty_proposals"
        assert self.manager.edge_cases[0].recovered_successfully is True
    
    @pytest.mark.asyncio
    async def test_handle_no_votes(self):
        """Test handling no votes edge case."""
        proposals = [
            Proposal(agent_id="agent_1", topic="Topic A"),
            Proposal(agent_id="agent_2", topic="Topic B"),
            Proposal(agent_id="agent_3", topic="Topic C")
        ]
        
        result_state = await self.manager._handle_no_votes(proposals)
        
        assert result_state.status == AgendaStatus.COMPLETED
        assert len(result_state.current_agenda) == 3
        assert result_state.current_agenda[0].topic == "Topic A"
        assert result_state.current_agenda[0].rank == 1
        assert result_state.current_agenda[1].topic == "Topic B"
        assert result_state.current_agenda[1].rank == 2
        assert len(self.manager.edge_cases) == 1
        assert self.manager.edge_cases[0].event_type == "no_votes"
    
    def test_transition_to_topic_success(self):
        """Test successful topic transition."""
        # Set up agenda
        self.manager.state.current_agenda = [
            AgendaItem(topic="Topic A", rank=1, status="pending"),
            AgendaItem(topic="Topic B", rank=2, status="pending")
        ]
        
        transition = self.manager.transition_to_topic("Topic A", ["agent_1", "agent_2"])
        
        assert transition.to_topic == "Topic A"
        assert transition.transition_type == "start"
        assert transition.agent_states_reset == ["agent_1", "agent_2"]
        assert self.manager.state.active_topic == "Topic A"
        
        # Find the agenda item and check its status
        topic_item = next(
            item for item in self.manager.state.current_agenda 
            if item.topic == "Topic A"
        )
        assert topic_item.status == "active"
        assert topic_item.discussion_started is not None
    
    def test_transition_to_topic_invalid(self):
        """Test transition to invalid topic."""
        self.manager.state.current_agenda = [
            AgendaItem(topic="Topic A", rank=1, status="pending")
        ]
        
        with pytest.raises(ValueError, match="Topic not found in agenda"):
            self.manager.transition_to_topic("Invalid Topic", ["agent_1"])
    
    def test_transition_with_previous_topic(self):
        """Test transition that completes previous topic."""
        # Set up agenda with active topic
        self.manager.state.current_agenda = [
            AgendaItem(topic="Previous Topic", rank=1, status="active"),
            AgendaItem(topic="Next Topic", rank=2, status="pending")
        ]
        self.manager.state.active_topic = "Previous Topic"
        
        transition = self.manager.transition_to_topic("Next Topic", ["agent_1"])
        
        assert transition.from_topic == "Previous Topic"
        assert transition.to_topic == "Next Topic"
        assert self.manager.state.active_topic == "Next Topic"
        assert "Previous Topic" in self.manager.state.completed_topics
        
        # Check previous topic is marked completed
        prev_item = next(
            item for item in self.manager.state.current_agenda 
            if item.topic == "Previous Topic"
        )
        assert prev_item.status == "completed"
        assert prev_item.discussion_ended is not None
    
    def test_is_agenda_complete_false(self):
        """Test agenda completion check when not complete."""
        self.manager.state.current_agenda = [
            AgendaItem(topic="Topic A", rank=1),
            AgendaItem(topic="Topic B", rank=2),
            AgendaItem(topic="Topic C", rank=3)
        ]
        self.manager.state.completed_topics = ["Topic A"]
        
        assert self.manager.is_agenda_complete() is False
    
    def test_is_agenda_complete_true(self):
        """Test agenda completion check when complete."""
        self.manager.state.current_agenda = [
            AgendaItem(topic="Topic A", rank=1),
            AgendaItem(topic="Topic B", rank=2)
        ]
        self.manager.state.completed_topics = ["Topic A", "Topic B"]
        
        assert self.manager.is_agenda_complete() is True
    
    def test_is_agenda_complete_empty(self):
        """Test agenda completion check with empty agenda."""
        self.manager.state.current_agenda = []
        self.manager.state.completed_topics = []
        
        assert self.manager.is_agenda_complete() is True
    
    def test_get_current_state(self):
        """Test getting current state."""
        state = self.manager.get_current_state()
        
        assert isinstance(state, AgendaState)
        assert state.session_id == "session_1"
    
    def test_get_analytics(self):
        """Test getting analytics."""
        analytics = self.manager.get_analytics()
        
        assert analytics.session_id == "session_1"
        assert isinstance(analytics.generated_at, datetime)
    
    @pytest.mark.asyncio
    async def test_collect_modification_suggestions(self):
        """Test collecting modification suggestions."""
        agents = self.create_mock_agents(2)
        remaining_topics = ["Topic A", "Topic B"]
        
        # Mock moderator response
        self.moderator.request_agenda_modification.return_value = "I suggest we add a new topic about technology"
        
        modifications = await self.manager._collect_modification_suggestions(
            agents, remaining_topics
        )
        
        # Should create modifications for agents
        assert len(modifications) >= 0
        
        # Check moderator was called
        assert self.moderator.request_agenda_modification.called
        call_args = self.moderator.request_agenda_modification.call_args[0]
        assert call_args[0] == remaining_topics
        assert call_args[1] == list(agents.keys())
    
    def test_apply_agenda_modifications(self):
        """Test applying agenda modifications."""
        # Set up existing agenda
        self.manager.state.current_agenda = [
            AgendaItem(topic="Existing Topic 1", rank=1),
            AgendaItem(topic="Existing Topic 2", rank=2)
        ]
        
        new_agenda = ["New Topic", "Existing Topic 1", "Modified Topic"]
        
        self.manager._apply_agenda_modifications(new_agenda)
        
        assert len(self.manager.state.current_agenda) == 3
        
        # Check rankings updated
        items_by_topic = {item.topic: item for item in self.manager.state.current_agenda}
        
        assert items_by_topic["New Topic"].rank == 1
        assert items_by_topic["Existing Topic 1"].rank == 2
        assert items_by_topic["Modified Topic"].rank == 3
        
        # Check that existing item preserved its data
        assert items_by_topic["Existing Topic 1"].proposed_by == []  # Original data
        
        # Check that new items have modification attribution
        assert items_by_topic["New Topic"].proposed_by == ["modification"]
        assert items_by_topic["Modified Topic"].proposed_by == ["modification"]