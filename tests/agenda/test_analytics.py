"""Tests for agenda analytics and reporting."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from virtual_agora.agenda.analytics import AgendaAnalyticsCollector
from virtual_agora.agenda.models import (
    Proposal, Vote, VoteType, VoteCollection, ProposalCollection,
    AgendaState, AgendaAnalytics, TopicTransition, AgendaModification,
    EdgeCaseEvent, ProposalStatus, VoteStatus, AgendaStatus
)


class TestAgendaAnalyticsCollector:
    """Test AgendaAnalyticsCollector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = AgendaAnalyticsCollector("session_1")
    
    def create_sample_proposal_collection(self):
        """Create sample proposal collection."""
        proposals = [
            Proposal(agent_id="agent_1", topic="Topic A"),
            Proposal(agent_id="agent_2", topic="Topic B"),
            Proposal(agent_id="agent_3", topic="Topic A")  # Duplicate
        ]
        
        return ProposalCollection(
            session_id="session_1",
            proposals=proposals,
            requested_agents=["agent_1", "agent_2", "agent_3"],
            responding_agents=["agent_1", "agent_2", "agent_3"],
            timeout_agents=[],
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now() - timedelta(minutes=2),
            status=ProposalStatus.COLLECTED
        )
    
    def create_sample_vote_collection(self):
        """Create sample vote collection."""
        votes = [
            Vote(
                agent_id="agent_1",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Topic A\n2. Topic B",
                parsed_preferences=["Topic A", "Topic B"],
                confidence_score=0.9,
                status=VoteStatus.SUBMITTED
            ),
            Vote(
                agent_id="agent_2",
                vote_type=VoteType.INITIAL_AGENDA,
                vote_content="1. Topic B\n2. Topic A",
                parsed_preferences=["Topic B", "Topic A"],
                confidence_score=0.8,
                status=VoteStatus.SUBMITTED
            )
        ]
        
        return VoteCollection(
            session_id="session_1",
            vote_type=VoteType.INITIAL_AGENDA,
            votes=votes,
            topic_options=["Topic A", "Topic B"],
            requested_agents=["agent_1", "agent_2", "agent_3"],
            responding_agents=["agent_1", "agent_2"],
            timeout_agents=["agent_3"],
            start_time=datetime.now() - timedelta(minutes=2),
            end_time=datetime.now(),
            status=VoteStatus.SUBMITTED
        )
    
    def create_sample_agenda_state(self):
        """Create sample agenda state."""
        from virtual_agora.agenda.models import AgendaItem
        
        return AgendaState(
            session_id="session_1",
            version=1,
            status=AgendaStatus.COMPLETED,
            current_agenda=[
                AgendaItem(topic="Topic A", rank=1, proposed_by=["agent_1", "agent_3"]),
                AgendaItem(topic="Topic B", rank=2, proposed_by=["agent_2"])
            ],
            completed_topics=["Topic A"],
            active_topic="Topic B"
        )
    
    @pytest.mark.asyncio
    async def test_record_agenda_initialization(self):
        """Test recording agenda initialization."""
        proposal_collection = self.create_sample_proposal_collection()
        vote_collection = self.create_sample_vote_collection()
        agenda_state = self.create_sample_agenda_state()
        
        await self.collector.record_agenda_initialization(
            proposal_collection, vote_collection, agenda_state
        )
        
        # Check that collections were stored
        assert len(self.collector.proposal_collections) == 1
        assert len(self.collector.vote_collections) == 1
        
        # Check timeline events were recorded
        timeline_events = self.collector.timeline_events
        assert len(timeline_events) >= 5  # Start/end for proposals, votes, synthesis
        
        event_types = [event["event_type"] for event in timeline_events]
        assert "proposal_collection_start" in event_types
        assert "proposal_collection_end" in event_types
        assert "vote_collection_start" in event_types
        assert "vote_collection_end" in event_types
        assert "agenda_synthesized" in event_types
    
    def test_record_topic_transition(self):
        """Test recording topic transition."""
        transition = TopicTransition(
            session_id="session_1",
            from_topic="Topic A",
            to_topic="Topic B",
            transition_type="start",
            duration_seconds=300.0,
            message_count=15,
            participant_count=3
        )
        
        self.collector.record_topic_transition(transition)
        
        # Check transition was stored
        assert len(self.collector.topic_transitions) == 1
        assert self.collector.topic_transitions[0] == transition
        
        # Check timeline event was recorded
        timeline_events = self.collector.timeline_events
        assert len(timeline_events) == 1
        assert timeline_events[0]["event_type"] == "topic_start"
        assert timeline_events[0]["details"]["from_topic"] == "Topic A"
        assert timeline_events[0]["details"]["to_topic"] == "Topic B"
    
    @pytest.mark.asyncio
    async def test_record_agenda_modification(self):
        """Test recording agenda modification."""
        modifications = [
            AgendaModification(
                agent_id="agent_1",
                modification_type="add",
                new_topic="New Topic"
            ),
            AgendaModification(
                agent_id="agent_2",
                modification_type="remove",
                target_topic="Old Topic"
            )
        ]
        
        vote_collection = self.create_sample_vote_collection()
        agenda_state = self.create_sample_agenda_state()
        agenda_state.version = 2  # Updated version
        
        await self.collector.record_agenda_modification(
            modifications, vote_collection, agenda_state
        )
        
        # Check modifications were stored
        assert len(self.collector.modifications) == 2
        assert len(self.collector.vote_collections) == 1
        
        # Check timeline event
        timeline_events = self.collector.timeline_events
        assert len(timeline_events) == 1
        assert timeline_events[0]["event_type"] == "agenda_modification"
        assert timeline_events[0]["details"]["modifications_count"] == 2
        assert timeline_events[0]["details"]["new_version"] == 2
    
    def test_record_edge_case(self):
        """Test recording edge case."""
        edge_case = EdgeCaseEvent(
            session_id="session_1",
            event_type="empty_proposals",
            description="No proposals were collected",
            resolution_strategy="fallback_agenda",
            system_response="Created fallback agenda",
            affected_agents=["agent_1", "agent_2"],
            recovered_successfully=True
        )
        
        self.collector.record_edge_case(edge_case)
        
        # Check edge case was stored
        assert len(self.collector.edge_cases) == 1
        assert self.collector.edge_cases[0] == edge_case
        
        # Check timeline event
        timeline_events = self.collector.timeline_events
        assert len(timeline_events) == 1
        assert timeline_events[0]["event_type"] == "edge_case"
        assert timeline_events[0]["details"]["case_type"] == "empty_proposals"
        assert timeline_events[0]["details"]["resolved"] is True
    
    def test_calculate_agent_participation(self):
        """Test agent participation calculation."""
        # Add proposal collection
        proposal_collection = self.create_sample_proposal_collection()
        self.collector.proposal_collections.append(proposal_collection)
        
        # Add vote collection
        vote_collection = self.create_sample_vote_collection()
        self.collector.vote_collections.append(vote_collection)
        
        participation = self.collector._calculate_agent_participation()
        
        # agent_1: participated in both proposals and votes = 100%
        assert participation["agent_1"] == 1.0
        
        # agent_2: participated in both proposals and votes = 100%
        assert participation["agent_2"] == 1.0
        
        # agent_3: participated in proposals but not votes = 50%
        assert participation["agent_3"] == 0.5
    
    def test_calculate_proposal_distribution(self):
        """Test proposal distribution calculation."""
        proposal_collection = self.create_sample_proposal_collection()
        self.collector.proposal_collections.append(proposal_collection)
        
        distribution = self.collector._calculate_proposal_distribution()
        
        assert distribution["agent_1"] == 1
        assert distribution["agent_2"] == 1
        assert distribution["agent_3"] == 1
    
    def test_analyze_voting_patterns(self):
        """Test voting patterns analysis."""
        vote_collection = self.create_sample_vote_collection()
        self.collector.vote_collections.append(vote_collection)
        
        patterns = self.collector._analyze_voting_patterns()
        
        assert patterns["total_votes"] == 2
        assert patterns["valid_votes"] == 2
        assert patterns["average_preferences_per_vote"] == 2.0
        
        # Check vote quality distribution
        quality_dist = patterns["vote_quality_distribution"]
        assert quality_dist["high"] == 2  # Both votes have high confidence
        
        # Should have consensus/controversy analysis
        assert "most_consensus_topic" in patterns
        assert "most_controversial_topic" in patterns
    
    def test_analyze_modification_patterns(self):
        """Test modification patterns analysis."""
        modifications = [
            AgendaModification(agent_id="agent_1", modification_type="add"),
            AgendaModification(agent_id="agent_2", modification_type="add"),
            AgendaModification(agent_id="agent_3", modification_type="remove")
        ]
        self.collector.modifications.extend(modifications)
        
        patterns = self.collector._analyze_modification_patterns()
        
        assert patterns["add"] == 2
        assert patterns["remove"] == 1
        assert patterns["total"] == 3
    
    def test_generate_analytics(self):
        """Test comprehensive analytics generation."""
        # Set up data
        proposal_collection = self.create_sample_proposal_collection()
        vote_collection = self.create_sample_vote_collection()
        agenda_state = self.create_sample_agenda_state()
        edge_cases = []
        
        self.collector.proposal_collections.append(proposal_collection)
        self.collector.vote_collections.append(vote_collection)
        
        analytics = self.collector.generate_analytics(agenda_state, edge_cases)
        
        assert isinstance(analytics, AgendaAnalytics)
        assert analytics.session_id == "session_1"
        assert analytics.total_proposals == 3
        assert analytics.unique_topics_proposed == 2  # Topic A and Topic B
        assert analytics.topics_completed == 1  # Topic A completed
        
        # Check participation rates
        assert len(analytics.agent_participation_rates) == 3
        assert analytics.average_vote_participation > 0.0
        
        # Check distributions
        assert len(analytics.topic_proposal_distribution) == 3
        assert analytics.voting_patterns["total_votes"] == 2
        
        # Check timeline events
        assert len(analytics.timeline_events) == 0  # No events recorded in this test
    
    def test_generate_participation_report(self):
        """Test participation report generation."""
        # Set up data
        proposal_collection = self.create_sample_proposal_collection()
        vote_collection = self.create_sample_vote_collection()
        
        self.collector.proposal_collections.append(proposal_collection)
        self.collector.vote_collections.append(vote_collection)
        
        report = self.collector.generate_participation_report()
        
        assert "overview" in report
        assert "agent_details" in report
        assert "recommendations" in report
        
        # Check overview
        overview = report["overview"]
        assert overview["total_agents"] == 3
        assert overview["total_opportunities"] == 6  # 3 agents * 2 opportunities each
        assert overview["total_participations"] == 5  # agent_1: 2, agent_2: 2, agent_3: 1
        
        # Check agent details
        agent_details = report["agent_details"]
        assert len(agent_details) == 3
        assert agent_details["agent_1"]["participation_rate"] == 1.0
        assert agent_details["agent_1"]["performance_category"] == "excellent"
        assert agent_details["agent_3"]["participation_rate"] == 0.5
        assert agent_details["agent_3"]["performance_category"] == "moderate"
    
    def test_categorize_participation(self):
        """Test participation categorization."""
        assert self.collector._categorize_participation(0.95) == "excellent"
        assert self.collector._categorize_participation(0.8) == "good"
        assert self.collector._categorize_participation(0.6) == "moderate"
        assert self.collector._categorize_participation(0.3) == "poor"
    
    def test_generate_timeline_report(self):
        """Test timeline report generation."""
        # Add some timeline events
        self.collector.timeline_events = [
            {
                "timestamp": datetime.now() - timedelta(minutes=5),
                "event_type": "proposal_collection_start",
                "details": {"requested_agents": 3}
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=2),
                "event_type": "proposal_collection_end",
                "details": {"proposals_collected": 3}
            },
            {
                "timestamp": datetime.now(),
                "event_type": "agenda_synthesized",
                "details": {"agenda_length": 2}
            }
        ]
        
        report = self.collector.generate_timeline_report()
        
        assert "events" in report
        assert "durations" in report
        assert "total_events" in report
        
        assert len(report["events"]) == 3
        assert report["total_events"] == 3
        
        # Check event structure
        event = report["events"][0]
        assert "timestamp" in event
        assert "event_type" in event
        assert "details" in event
    
    def test_export_analytics_data(self):
        """Test analytics data export."""
        # Set up some data
        proposal_collection = self.create_sample_proposal_collection()
        vote_collection = self.create_sample_vote_collection()
        transition = TopicTransition(
            session_id="session_1",
            from_topic="Topic A",
            to_topic="Topic B",
            transition_type="start"
        )
        modification = AgendaModification(
            agent_id="agent_1",
            modification_type="add",
            new_topic="New Topic"
        )
        edge_case = EdgeCaseEvent(
            session_id="session_1",
            event_type="test_case",
            description="Test edge case",
            resolution_strategy="test_strategy",
            system_response="Test response"
        )
        
        self.collector.proposal_collections.append(proposal_collection)
        self.collector.vote_collections.append(vote_collection)
        self.collector.topic_transitions.append(transition)
        self.collector.modifications.append(modification)
        self.collector.edge_cases.append(edge_case)
        
        export_data = self.collector.export_analytics_data()
        
        assert export_data["session_id"] == "session_1"
        assert "export_timestamp" in export_data
        
        # Check all data types are included
        assert len(export_data["proposal_collections"]) == 1
        assert len(export_data["vote_collections"]) == 1
        assert len(export_data["topic_transitions"]) == 1
        assert len(export_data["modifications"]) == 1
        assert len(export_data["edge_cases"]) == 1
        
        # Check data structure
        pc_data = export_data["proposal_collections"][0]
        assert "id" in pc_data
        assert "start_time" in pc_data
        assert "proposals_count" in pc_data
        assert "completion_rate" in pc_data