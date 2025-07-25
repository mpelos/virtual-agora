"""Unit tests for tool integration module."""

import pytest
from unittest.mock import Mock, patch
import json

from virtual_agora.tools.tool_integration import (
    ProposalTool,
    VotingTool,
    SummaryTool,
    create_discussion_tools,
    count_votes,
    format_agenda,
)


class TestProposalTool:
    """Test ProposalTool functionality."""
    
    def test_proposal_tool_creation(self):
        """Test creating a proposal tool."""
        tool = ProposalTool()
        assert tool.name == "propose_topics"
        assert "3-5 discussion topics" in tool.description
        assert tool.return_direct is False
    
    def test_proposal_tool_valid_input(self):
        """Test proposal tool with valid input."""
        tool = ProposalTool()
        result = tool._run(
            topics=["AI Ethics", "Climate Change", "Space Exploration"],
            rationale="These are important global topics"
        )
        
        assert "I propose the following discussion topics:" in result
        assert "1. AI Ethics" in result
        assert "2. Climate Change" in result
        assert "3. Space Exploration" in result
        assert "Rationale: These are important global topics" in result
    
    def test_proposal_tool_invalid_topic_count(self):
        """Test proposal tool with wrong number of topics."""
        tool = ProposalTool()
        
        # Too few topics
        result = tool._run(topics=["Only one"])
        assert "Error: Please propose between 3 and 5 topics" in result
        
        # Too many topics
        result = tool._run(topics=["T1", "T2", "T3", "T4", "T5", "T6"])
        assert "Error: Please propose between 3 and 5 topics" in result
    
    def test_proposal_tool_without_rationale(self):
        """Test proposal tool without rationale."""
        tool = ProposalTool()
        result = tool._run(
            topics=["Topic 1", "Topic 2", "Topic 3"]
        )
        
        assert "I propose the following discussion topics:" in result
        assert "Rationale:" not in result
    
    @pytest.mark.asyncio
    async def test_proposal_tool_async(self):
        """Test async version of proposal tool."""
        tool = ProposalTool()
        result = await tool._arun(
            topics=["Async 1", "Async 2", "Async 3"]
        )
        
        assert "1. Async 1" in result
        assert "2. Async 2" in result
        assert "3. Async 3" in result


class TestVotingTool:
    """Test VotingTool functionality."""
    
    def test_voting_tool_creation(self):
        """Test creating a voting tool."""
        tool = VotingTool()
        assert tool.name == "vote"
        assert "'yes', 'no', or 'abstain'" in tool.description
        assert tool.return_direct is False
    
    def test_voting_tool_yes_vote(self):
        """Test voting yes."""
        tool = VotingTool()
        result = tool._run(
            topic="AI Ethics",
            vote="yes",
            reasoning="This is an important topic"
        )
        
        assert "I vote 'Yes' on the topic: AI Ethics" in result
        assert "Reasoning: This is an important topic" in result
    
    def test_voting_tool_no_vote(self):
        """Test voting no."""
        tool = VotingTool()
        result = tool._run(
            topic="Space Tourism",
            vote="no",
            reasoning="Not a priority right now"
        )
        
        assert "I vote 'No' on the topic: Space Tourism" in result
        assert "Reasoning: Not a priority right now" in result
    
    def test_voting_tool_abstain(self):
        """Test abstaining from vote."""
        tool = VotingTool()
        result = tool._run(
            topic="Quantum Computing",
            vote="abstain"
        )
        
        assert "I vote 'Abstain' on the topic: Quantum Computing" in result
        assert "Reasoning:" not in result  # No reasoning provided
    
    def test_voting_tool_invalid_vote(self):
        """Test invalid vote value."""
        tool = VotingTool()
        result = tool._run(
            topic="Test Topic",
            vote="maybe"  # Invalid vote
        )
        
        assert "Error: Vote must be 'yes', 'no', or 'abstain'" in result
    
    def test_voting_tool_case_insensitive(self):
        """Test that votes are case insensitive."""
        tool = VotingTool()
        
        # Test uppercase
        result = tool._run(topic="Test", vote="YES")
        assert "I vote 'Yes'" in result
        
        # Test mixed case
        result = tool._run(topic="Test", vote="No")
        assert "I vote 'No'" in result


class TestSummaryTool:
    """Test SummaryTool functionality."""
    
    def test_summary_tool_creation(self):
        """Test creating a summary tool."""
        tool = SummaryTool()
        assert tool.name == "summarize"
        assert "generate a summary" in tool.description
        assert tool.return_direct is False
    
    def test_summary_tool_concise_style(self):
        """Test concise summary generation."""
        tool = SummaryTool()
        content = " ".join(["word"] * 200)  # 200 words
        
        result = tool._run(
            content=content,
            max_length=50,
            style="concise"
        )
        
        # Should be truncated
        assert result.endswith("...")
        # Should be shorter than original
        assert len(result.split()) < 100
    
    def test_summary_tool_detailed_style(self):
        """Test detailed summary generation."""
        tool = SummaryTool()
        content = " ".join(["word"] * 200)
        
        result = tool._run(
            content=content,
            max_length=150,
            style="detailed"
        )
        
        # Should preserve more content
        assert len(result.split()) <= 151  # max_length + possible "..."
    
    def test_summary_tool_bullet_style(self):
        """Test bullet point summary generation."""
        tool = SummaryTool()
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        result = tool._run(
            content=content,
            style="bullet"
        )
        
        assert "Summary:" in result
        assert "• First sentence" in result
        assert "• Second sentence" in result
        assert "• Third sentence" in result
        assert "• Fourth sentence" in result
    
    def test_summary_tool_default_parameters(self):
        """Test summary with default parameters."""
        tool = SummaryTool()
        content = "This is a test content for summarization."
        
        result = tool._run(content=content)
        
        # Should use default max_length (150) and style (concise)
        assert len(result) > 0
        assert "This is a test content" in result


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_count_votes(self):
        """Test vote counting function."""
        votes = [
            {"vote": "yes"},
            {"vote": "yes"},
            {"vote": "no"},
            {"vote": "abstain"},
            {"vote": "YES"},  # Case variation
            {"vote": "invalid"},  # Should be ignored
        ]
        
        counts = count_votes(votes)
        
        assert counts["yes"] == 3  # Including uppercase
        assert counts["no"] == 1
        assert counts["abstain"] == 1
    
    def test_count_votes_empty(self):
        """Test counting empty vote list."""
        counts = count_votes([])
        
        assert counts["yes"] == 0
        assert counts["no"] == 0
        assert counts["abstain"] == 0
    
    def test_format_agenda_simple(self):
        """Test formatting agenda without votes."""
        topics = ["AI Ethics", "Climate Change", "Education Reform"]
        
        agenda = format_agenda(topics)
        
        assert "Discussion Agenda:" in agenda
        assert "1. AI Ethics" in agenda
        assert "2. Climate Change" in agenda
        assert "3. Education Reform" in agenda
    
    def test_format_agenda_with_votes(self):
        """Test formatting agenda with vote information."""
        topics = ["Topic A", "Topic B"]
        votes = {
            "Topic A": "5 yes, 2 no",
            "Topic B": "3 yes, 4 no"
        }
        
        agenda = format_agenda(topics, votes)
        
        assert "1. Topic A (Votes: 5 yes, 2 no)" in agenda
        assert "2. Topic B (Votes: 3 yes, 4 no)" in agenda
    
    def test_create_discussion_tools(self):
        """Test creating standard discussion tools."""
        tools = create_discussion_tools()
        
        assert len(tools) == 3
        
        # Check tool types
        tool_types = {type(tool) for tool in tools}
        assert ProposalTool in tool_types
        assert VotingTool in tool_types
        assert SummaryTool in tool_types
        
        # Check tool names
        tool_names = {tool.name for tool in tools}
        assert "propose_topics" in tool_names
        assert "vote" in tool_names
        assert "summarize" in tool_names