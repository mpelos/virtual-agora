# Phase 2: Specialized Agent Architecture Guide

## Purpose
This document provides comprehensive instructions for decomposing the monolithic ModeratorAgent (2,861 lines) into five specialized agents, each with a single, well-defined responsibility. This phase transforms the architecture from mode-based switching to tool-based invocation.

## Prerequisites
- Completed Phase 1 (Configuration & State updates)
- Understanding of the current ModeratorAgent implementation
- Familiarity with LangChain agent patterns
- Access to v1.3 specification prompts

## Current Architecture Analysis

### ModeratorAgent Method Mapping

The current ModeratorAgent contains methods that must be distributed among specialized agents:

#### Methods to Keep in Refactored ModeratorAgent
**Purpose**: Process facilitation and relevance enforcement only

| Method | Lines | Purpose | Keep/Move |
|--------|-------|---------|-----------|
| `parse_agenda_json` | 351-422 | Parse voting results to JSON | Keep |
| `synthesize_agenda` | 593-659 | Synthesize votes into agenda | Keep |
| `evaluate_message_relevance` | 1569-1690 | Check if message is on-topic | Keep |
| `track_relevance_violation` | 1690-1733 | Track off-topic warnings | Keep |
| `issue_relevance_warning` | 1733-1785 | Warn agents about relevance | Keep |
| `mute_agent` | 1785-1824 | Temporarily mute off-topic agents | Keep |
| `request_topic_proposals` | 422-461 | Ask agents for topics | Keep |
| `collect_proposals` | 461-516 | Gather topic proposals | Keep |

#### Methods to Move to SummarizerAgent
**Purpose**: Round compression and context management

| Method | Lines | New Location |
|--------|-------|--------------|
| `generate_round_summary` | 1125-1204 | Core method |
| `generate_progressive_summary` | 1482-1553 | Progressive compression |
| `extract_key_insights` | 1349-1482 | Insight extraction |

#### Methods to Move to TopicReportAgent
**Purpose**: Synthesize concluded agenda items

| Method | Lines | New Location |
|--------|-------|--------------|
| `generate_topic_summary` | 1204-1291 | Core method |
| `_generate_topic_summary_map_reduce` | 1291-1349 | Implementation detail |
| `handle_minority_considerations` | 2304-2370 | Include dissent |

#### Methods to Move to EcclesiaReportAgent
**Purpose**: Generate final multi-part report

| Method | Lines | New Location |
|--------|-------|--------------|
| `generate_report_structure` | 659-719 | Define sections |
| `generate_section_content` | 719-769 | Write sections |
| `define_report_structure` | 2549-2580 | Async version |
| `generate_report_section` | 2580-2624 | Async version |

#### Methods to Remove (Mode-Related)
| Method | Lines | Reason |
|--------|-------|--------|
| `set_mode` | 151-172 | No more modes |
| `get_current_mode` | 172-180 | No more modes |
| `_get_mode_specific_prompt` | 180-240 | Specialized prompts per agent |

## Specialized Agent Implementations

### Step 1: Create Base Structure for Specialized Agents

Each specialized agent should follow this structure:

```python
# src/virtual_agora/agents/summarizer.py
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)

class SummarizerAgent(LLMAgent):
    """Specialized agent for text compression and summarization.
    
    This agent is responsible for creating concise, agent-agnostic
    summaries of discussion rounds to manage context effectively.
    """
    
    PROMPT_VERSION = "1.3"
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        compression_ratio: float = 0.3,  # Target 30% of original
        max_summary_tokens: int = 500,
        **kwargs
    ):
        """Initialize the Summarizer agent.
        
        Args:
            agent_id: Unique identifier
            llm: Language model instance
            compression_ratio: Target compression ratio
            max_summary_tokens: Maximum tokens per summary
        """
        # Use specialized prompt from v1.3 spec
        system_prompt = self._get_summarizer_prompt()
        
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            system_prompt=system_prompt,
            **kwargs
        )
        
        self.compression_ratio = compression_ratio
        self.max_summary_tokens = max_summary_tokens
    
    def _get_summarizer_prompt(self) -> str:
        """Get the specialized summarizer prompt from v1.3 spec."""
        return """You are a specialized text compression tool for Virtual Agora. 
Your task is to read all agent comments from a single discussion round and 
create a concise, agent-agnostic summary that captures the key points, 
arguments, and insights without attribution to specific agents.

Focus on:
1. Main arguments presented
2. Points of agreement and disagreement  
3. New insights or perspectives introduced
4. Questions raised or areas requiring further exploration

Your summary will be used as context for future rounds, so ensure it 
preserves essential information while being substantially more concise 
than the original. Write in third person, avoid agent names, and 
maintain objectivity."""
```

### Step 2: Implement Core Methods for Each Agent

#### SummarizerAgent Core Methods

```python
def summarize_round(
    self, 
    messages: List[Dict[str, Any]], 
    topic: str,
    round_number: int
) -> str:
    """Create a compressed summary of a discussion round.
    
    Args:
        messages: List of agent messages from the round
        topic: Current discussion topic
        round_number: Current round number
        
    Returns:
        Compressed summary text
    """
    # Extract message content
    content_list = [msg['content'] for msg in messages]
    combined_text = "\n\n".join(content_list)
    
    # Calculate target length
    original_tokens = self._estimate_tokens(combined_text)
    target_tokens = int(original_tokens * self.compression_ratio)
    target_tokens = min(target_tokens, self.max_summary_tokens)
    
    # Create summarization prompt
    prompt = f"""Topic: {topic}
Round: {round_number}

Agent Comments:
{combined_text}

Create a summary of approximately {target_tokens} tokens that captures 
the essential points while maintaining objectivity."""
    
    # Generate summary
    summary = self.generate_response(prompt)
    
    # Log compression metrics
    summary_tokens = self._estimate_tokens(summary)
    logger.info(
        f"Round {round_number} compressed: "
        f"{original_tokens} -> {summary_tokens} tokens "
        f"({summary_tokens/original_tokens:.1%} of original)"
    )
    
    return summary
```

#### TopicReportAgent Core Methods

```python
def synthesize_topic(
    self,
    round_summaries: List[str],
    final_considerations: List[str],
    topic: str,
    discussion_theme: str
) -> str:
    """Create comprehensive report for concluded topic.
    
    Args:
        round_summaries: All round summaries for this topic
        final_considerations: Final thoughts from agents
        topic: The concluded topic
        discussion_theme: Overall session theme
        
    Returns:
        Comprehensive topic report
    """
    # Combine all summaries
    all_summaries = "\n\n".join([
        f"Round {i+1} Summary:\n{summary}" 
        for i, summary in enumerate(round_summaries)
    ])
    
    # Include final considerations if any
    considerations_text = ""
    if final_considerations:
        considerations_text = "\n\nFinal Considerations:\n" + \
            "\n".join(final_considerations)
    
    # Generate comprehensive report
    prompt = f"""Discussion Theme: {discussion_theme}
Concluded Topic: {topic}

Round Summaries:
{all_summaries}
{considerations_text}

Create a comprehensive report following this structure:
1. Topic overview and key questions addressed
2. Major themes and arguments that emerged
3. Points of consensus among participants
4. Areas of disagreement or ongoing debate
5. Key insights and novel perspectives
6. Implications and potential next steps

Write as an objective analyst, not a participant."""
    
    return self.generate_response(prompt)
```

#### EcclesiaReportAgent Core Methods

```python
def generate_report_structure(
    self, 
    topic_reports: Dict[str, str],
    discussion_theme: str
) -> List[str]:
    """Define the structure for the final report.
    
    Args:
        topic_reports: All topic reports keyed by topic
        discussion_theme: Overall session theme
        
    Returns:
        List of section titles for the report
    """
    # Prepare context
    topics_list = list(topic_reports.keys())
    topics_summary = "\n".join([
        f"- {topic}" for topic in topics_list
    ])
    
    prompt = f"""Discussion Theme: {discussion_theme}

Topics Discussed:
{topics_summary}

Based on these topics, define a logical report structure.
Output ONLY a JSON list of section titles.

Example: ["Executive Summary", "Key Themes", "Consensus Points", 
"Open Questions", "Recommendations", "Conclusion"]"""
    
    # Generate structure with JSON validation
    response = self.generate_json_response(
        prompt, 
        schema={"type": "array", "items": {"type": "string"}}
    )
    
    return response

def write_section(
    self,
    section_title: str,
    topic_reports: Dict[str, str],
    discussion_theme: str,
    previous_sections: Dict[str, str]
) -> str:
    """Write content for a specific report section.
    
    Args:
        section_title: Title of section to write
        topic_reports: All topic reports
        discussion_theme: Overall theme
        previous_sections: Already written sections
        
    Returns:
        Section content
    """
    # Context from previous sections
    prev_context = ""
    if previous_sections:
        prev_context = "Previously written sections:\n" + \
            "\n".join([
                f"{title}:\n{content[:200]}..." 
                for title, content in previous_sections.items()
            ])
    
    prompt = f"""Theme: {discussion_theme}
Section to Write: {section_title}

{prev_context}

Using all topic reports, write the content for "{section_title}".
Focus on synthesis across topics, not just summarization."""
    
    return self.generate_response(prompt)
```

### Step 3: Refactor ModeratorAgent

Remove all methods that have been moved to specialized agents:

```python
# src/virtual_agora/agents/moderator.py (refactored)
class ModeratorAgent(LLMAgent):
    """Specialized Moderator for process facilitation.
    
    Responsibilities:
    - Agenda synthesis from votes
    - Relevance enforcement
    - Process announcements
    
    This agent no longer handles summarization or report generation.
    """
    
    PROMPT_VERSION = "1.3"
    
    def __init__(
        self,
        agent_id: str,
        llm: BaseChatModel,
        relevance_threshold: float = 0.7,
        warning_threshold: int = 2,
        mute_duration_minutes: int = 5,
        **kwargs
    ):
        # Remove mode parameter - no longer needed
        system_prompt = self._get_facilitation_prompt()
        
        super().__init__(
            agent_id=agent_id,
            llm=llm,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Only relevance-related attributes remain
        self.relevance_threshold = relevance_threshold
        self.warning_threshold = warning_threshold
        self.mute_duration_minutes = mute_duration_minutes
        self.warnings = {}  # agent_id -> warning count
        self.muted_agents = {}  # agent_id -> unmute_time
    
    def _get_facilitation_prompt(self) -> str:
        """Get the process facilitation prompt from v1.3 spec."""
        return """You are a specialized reasoning tool for Virtual Agora's 
process facilitation. You are NOT a discussion participant and have 
NO opinions on topics. You are invoked by graph nodes to perform 
specific analytical tasks:

1. **Proposal Compilation**: Read all agent proposals and create a 
   single, deduplicated list of unique agenda items.
2. **Vote Synthesis**: Analyze natural language votes from agents 
   and produce a rank-ordered agenda. Output MUST be valid JSON: 
   {"proposed_agenda": ["Item A", "Item B", "Item C"]}.
3. Break ties using objective criteria (clarity, scope, relevance).

You must be precise, analytical, and strictly adhere to required 
output formats. Focus solely on process logic, never content opinions."""
```

### Step 4: Update Agent Factory

Modify the agent factory to create specialized agents:

```python
# src/virtual_agora/agents/agent_factory.py
from virtual_agora.agents.summarizer import SummarizerAgent
from virtual_agora.agents.topic_report_agent import TopicReportAgent
from virtual_agora.agents.ecclesia_report_agent import EcclesiaReportAgent

def create_specialized_agents(config: Config) -> Dict[str, LLMAgent]:
    """Create all specialized agents from configuration.
    
    Returns:
        Dictionary mapping agent type to agent instance
    """
    agents = {}
    
    # Create moderator (process facilitation only)
    moderator_llm = create_provider(
        provider=config.moderator.provider.value,
        model=config.moderator.model
    )
    agents['moderator'] = ModeratorAgent(
        agent_id='moderator',
        llm=moderator_llm
    )
    
    # Create summarizer
    summarizer_llm = create_provider(
        provider=config.summarizer.provider.value,
        model=config.summarizer.model
    )
    agents['summarizer'] = SummarizerAgent(
        agent_id='summarizer',
        llm=summarizer_llm
    )
    
    # Create topic report agent
    topic_report_llm = create_provider(
        provider=config.topic_report.provider.value,
        model=config.topic_report.model
    )
    agents['topic_report'] = TopicReportAgent(
        agent_id='topic_report',
        llm=topic_report_llm
    )
    
    # Create ecclesia report agent
    ecclesia_report_llm = create_provider(
        provider=config.ecclesia_report.provider.value,
        model=config.ecclesia_report.model
    )
    agents['ecclesia_report'] = EcclesiaReportAgent(
        agent_id='ecclesia_report',
        llm=ecclesia_report_llm
    )
    
    logger.info(f"Created {len(agents)} specialized agents")
    return agents
```

## Implementation Strategy

### Development Workflow

1. **Create Agent Files First**
   ```bash
   touch src/virtual_agora/agents/summarizer.py
   touch src/virtual_agora/agents/topic_report_agent.py
   touch src/virtual_agora/agents/ecclesia_report_agent.py
   ```

2. **Implement in Order**
   - SummarizerAgent (simplest, well-defined task)
   - TopicReportAgent (builds on summaries)
   - EcclesiaReportAgent (depends on topic reports)
   - Refactor ModeratorAgent last

3. **Test Each Agent Independently**
   - Unit tests for each agent
   - Integration tests with mock LLMs
   - Prompt validation tests

### Migration Safety Checklist

- [ ] All methods are accounted for (moved or kept)
- [ ] No duplicate functionality between agents
- [ ] Each agent has a single, clear responsibility
- [ ] Prompts match v1.3 specification exactly
- [ ] State interactions are well-defined
- [ ] Error handling is preserved
- [ ] Logging provides good observability

### Testing Requirements

1. **Unit Tests per Agent**
   ```python
   # tests/agents/test_summarizer.py
   def test_summarizer_compression():
       """Test summary achieves target compression."""
       summarizer = SummarizerAgent(
           agent_id='test',
           llm=FakeLLM(),
           compression_ratio=0.3
       )
       messages = [
           {'content': 'Long message ' * 100},
           {'content': 'Another long message ' * 100}
       ]
       summary = summarizer.summarize_round(messages, 'Test Topic', 1)
       assert len(summary) < len(str(messages)) * 0.4
   ```

2. **Integration Tests**
   ```python
   def test_specialized_agents_coordination():
       """Test agents work together correctly."""
       # Create all agents
       agents = create_specialized_agents(test_config)
       
       # Simulate round -> summary -> topic report flow
       round_summary = agents['summarizer'].summarize_round(...)
       topic_report = agents['topic_report'].synthesize_topic(...)
       
       assert round_summary in topic_report
   ```

## Common Pitfalls and Solutions

### Pitfall 1: Prompt Drift
**Problem**: Modified prompts don't match v1.3 spec
**Solution**: Copy prompts verbatim from spec, test outputs

### Pitfall 2: Lost Functionality
**Problem**: Some edge case handling gets lost in refactor
**Solution**: Comprehensive method mapping, thorough testing

### Pitfall 3: State Coupling
**Problem**: Agents directly modify shared state
**Solution**: Agents return values, nodes update state

### Pitfall 4: Performance Regression
**Problem**: Multiple agents slower than single agent
**Solution**: Parallel invocation where possible

## Validation Checklist

- [ ] ModeratorAgent reduced to <500 lines
- [ ] Each specialized agent <500 lines
- [ ] All v1.3 prompts implemented exactly
- [ ] No mode switching logic remains
- [ ] Agent factory creates all agents
- [ ] Unit tests pass for each agent
- [ ] Integration tests pass
- [ ] Performance benchmarks acceptable

## References

- Current ModeratorAgent: `src/virtual_agora/agents/moderator.py`
- v1.3 Agent Prompts: Section 5 of `docs/project_spec_2.md`
- LLMAgent Base Class: `src/virtual_agora/agents/llm_agent.py`
- Agent Factory: `src/virtual_agora/agents/agent_factory.py`

## Next Phase

Once all specialized agents are implemented and tested, proceed to Phase 3: Graph Flow Transformation.

---

**Document Version**: 1.0
**Phase**: 2 of 5
**Status**: Implementation Guide