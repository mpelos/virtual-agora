"""Tests for specialized agents in Virtual Agora v1.3."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel

from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.topic_report_agent import TopicReportAgent
from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent
from virtual_agora.config.models import Config, Provider


class FakeLLM(BaseChatModel):
    """Fake LLM for testing."""
    
    response: str = "Test response"
    calls: list = []
    
    def __init__(self, response: str = "Test response", **kwargs):
        super().__init__(**kwargs)
        self.response = response
        self.calls = []
    
    def _generate(self, messages, **kwargs):
        self.calls.append(messages)
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage
        generation = ChatGeneration(message=AIMessage(content=self.response))
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self):
        return "fake"
    
    async def _agenerate(self, messages, **kwargs):
        return self._generate(messages, **kwargs)


class TestSummarizerAgent:
    """Tests for SummarizerAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        llm = FakeLLM()
        agent = SummarizerAgent(
            agent_id="summarizer",
            llm=llm,
            compression_ratio=0.3,
            max_summary_tokens=500
        )
        
        assert agent.agent_id == "summarizer"
        assert agent.compression_ratio == 0.3
        assert agent.max_summary_tokens == 500
        assert "specialized text compression tool" in agent.system_prompt
    
    def test_summarize_round(self):
        """Test round summarization."""
        summary_text = "Key points: Agreement on topic A, debate on topic B."
        llm = FakeLLM(response=summary_text)
        agent = SummarizerAgent(agent_id="summarizer", llm=llm, enable_error_handling=False)
        
        messages = [
            {"content": "Agent 1 thinks we should focus on security."},
            {"content": "Agent 2 agrees but wants to consider performance."},
            {"content": "Agent 3 suggests a balanced approach."}
        ]
        
        result = agent.summarize_round(messages, "Security vs Performance", 1)
        
        assert result == summary_text
        assert len(llm.calls) == 1
        # Check that the prompt includes the topic and round number
        prompt = str(llm.calls[0])
        assert "Security vs Performance" in prompt
        assert "Round: 1" in prompt
    
    def test_generate_progressive_summary(self):
        """Test progressive summary generation."""
        prog_summary = "Overall discussion showed consensus on key issues."
        llm = FakeLLM(response=prog_summary)
        agent = SummarizerAgent(agent_id="summarizer", llm=llm, enable_error_handling=False)
        
        summaries = [
            "Round 1: Initial proposals discussed.",
            "Round 2: Debate on implementation details.",
            "Round 3: Consensus reached on approach."
        ]
        
        result = agent.generate_progressive_summary(summaries, "Implementation Strategy")
        
        assert result == prog_summary
        assert len(llm.calls) == 1
        # Check that all summaries are included
        prompt = str(llm.calls[0])
        for summary in summaries:
            assert summary in prompt
    
    def test_extract_key_insights(self):
        """Test key insight extraction."""
        insights_text = "- Security must be the top priority\n- Performance can be optimized later\n- User experience is critical"
        llm = FakeLLM(response=insights_text)
        agent = SummarizerAgent(agent_id="summarizer", llm=llm, enable_error_handling=False)
        
        messages = [
            {"content": "We need to prioritize security above all else."},
            {"content": "Performance optimizations can come in phase 2."},
            {"content": "Don't forget about user experience!"}
        ]
        
        insights = agent.extract_key_insights(messages, "System Design")
        
        assert len(insights) == 3
        assert "Security must be the top priority" in insights
        assert "Performance can be optimized later" in insights
        assert "User experience is critical" in insights


class TestTopicReportAgent:
    """Tests for TopicReportAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        llm = FakeLLM()
        agent = TopicReportAgent(agent_id="topic_report", llm=llm, enable_error_handling=False)
        
        assert agent.agent_id == "topic_report"
        assert "specialized synthesis tool" in agent.system_prompt
        assert "comprehensive, standalone report" in agent.system_prompt
    
    def test_synthesize_topic(self):
        """Test topic synthesis."""
        report = """## Topic: Security Architecture

### Overview
The discussion focused on designing a robust security architecture.

### Key Themes
1. Defense in depth strategy
2. Zero trust principles
3. Continuous monitoring

### Consensus Points
- Security must be built-in, not bolted-on
- Regular audits are essential

### Areas of Debate
- Level of encryption needed
- Performance trade-offs

### Key Insights
- Modern threats require adaptive security
- Automation is crucial for response time

### Next Steps
- Implement security framework
- Establish monitoring protocols"""
        
        llm = FakeLLM(response=report)
        agent = TopicReportAgent(agent_id="topic_report", llm=llm, enable_error_handling=False)
        
        round_summaries = [
            "Round 1: Discussed threat landscape and security principles.",
            "Round 2: Debated encryption standards and implementation.",
            "Round 3: Agreed on monitoring and response strategies."
        ]
        
        final_considerations = [
            "We should not compromise on encryption standards.",
            "Performance impact needs careful measurement."
        ]
        
        result = agent.synthesize_topic(
            round_summaries,
            final_considerations,
            "Security Architecture",
            "System Design Best Practices"
        )
        
        assert "Security Architecture" in result
        assert "Defense in depth" in result
        assert len(llm.calls) == 1
    
    def test_handle_minority_considerations(self):
        """Test minority view synthesis."""
        minority_synthesis = """The dissenting agents raised important concerns about performance impact and implementation complexity that warrant further consideration."""
        
        llm = FakeLLM(response=minority_synthesis)
        agent = TopicReportAgent(agent_id="topic_report", llm=llm, enable_error_handling=False)
        
        dissenting_views = [
            {"agent": "gpt-4-1", "content": "The performance impact could be severe."},
            {"agent": "claude-1", "content": "Implementation complexity is being underestimated."}
        ]
        
        result = agent.handle_minority_considerations(
            ["gpt-4-1", "claude-1"],
            dissenting_views,
            "Security Implementation",
            "Majority agrees to proceed with full encryption"
        )
        
        assert "performance impact" in result
        assert "implementation complexity" in result


class TestEcclesiaReportAgent:
    """Tests for EcclesiaReportAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        llm = FakeLLM()
        agent = EcclesiaReportAgent(agent_id="ecclesia_report", llm=llm, enable_error_handling=False)
        
        assert agent.agent_id == "ecclesia_report"
        assert "The Writer" in agent.system_prompt
        assert "comprehensive, multi-section final report" in agent.system_prompt
    
    def test_generate_report_structure(self):
        """Test report structure generation."""
        structure_json = '["Executive Summary", "Key Themes", "Consensus Points", "Open Questions", "Recommendations", "Conclusion"]'
        llm = FakeLLM(response=structure_json)
        agent = EcclesiaReportAgent(agent_id="ecclesia_report", llm=llm, enable_error_handling=False)
        
        topic_reports = {
            "Security Architecture": "Report on security design...",
            "Performance Optimization": "Report on performance...",
            "User Experience": "Report on UX considerations..."
        }
        
        sections = agent.generate_report_structure(topic_reports, "System Design")
        
        assert len(sections) == 6
        assert "Executive Summary" in sections
        assert "Recommendations" in sections
    
    def test_write_section(self):
        """Test section writing."""
        section_content = """# Executive Summary

This session explored three critical aspects of system design: security architecture, performance optimization, and user experience. The discussion revealed strong consensus on security-first principles while highlighting tensions between performance and user experience requirements.

Key outcomes include:
- Agreement on zero-trust security model
- Performance baseline targets established
- User experience principles defined

The session provides a solid foundation for the implementation phase."""
        
        llm = FakeLLM(response=section_content)
        agent = EcclesiaReportAgent(agent_id="ecclesia_report", llm=llm, enable_error_handling=False)
        
        topic_reports = {
            "Security Architecture": "Detailed security discussion...",
            "Performance Optimization": "Performance analysis...",
            "User Experience": "UX considerations..."
        }
        
        result = agent.write_section(
            "Executive Summary",
            topic_reports,
            "System Design",
            {}
        )
        
        assert "Executive Summary" in result
        assert "security-first principles" in result
        assert "performance" in result


class TestAgentIntegration:
    """Integration tests for specialized agents."""
    
    @patch('virtual_agora.providers.factory.ProviderFactory.create_provider')
    def test_agent_factory_creates_all_agents(self, mock_create_provider):
        """Test that agent factory creates all specialized agents."""
        from virtual_agora.agents.agent_factory import AgentFactory
        from virtual_agora.config.models import (
            Config, ModeratorConfig, SummarizerConfig,
            TopicReportConfig, EcclesiaReportConfig, AgentConfig
        )
        
        # Mock LLM creation
        mock_llm = FakeLLM()
        mock_create_provider.return_value = mock_llm
        
        # Create test configuration
        config = Config(
            moderator=ModeratorConfig(provider=Provider.GOOGLE, model="gemini-2.5-pro"),
            summarizer=SummarizerConfig(provider=Provider.OPENAI, model="gpt-4o"),
            topic_report=TopicReportConfig(provider=Provider.ANTHROPIC, model="claude-3-opus"),
            ecclesia_report=EcclesiaReportConfig(provider=Provider.GOOGLE, model="gemini-2.5-pro"),
            agents=[
                AgentConfig(provider=Provider.OPENAI, model="gpt-4o", count=2)
            ]
        )
        
        factory = AgentFactory(config)
        specialized = factory.create_specialized_agents()
        
        assert 'moderator' in specialized
        assert 'summarizer' in specialized
        assert 'topic_report' in specialized
        assert 'ecclesia_report' in specialized
        
        assert isinstance(specialized['summarizer'], SummarizerAgent)
        assert isinstance(specialized['topic_report'], TopicReportAgent)
        assert isinstance(specialized['ecclesia_report'], EcclesiaReportAgent)
        
        # Verify correct number of provider calls (4 specialized agents)
        assert mock_create_provider.call_count >= 4


if __name__ == "__main__":
    pytest.main([__file__])