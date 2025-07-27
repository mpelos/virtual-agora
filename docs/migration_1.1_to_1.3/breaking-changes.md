# Breaking Changes: v1.1 to v1.3

## Overview
This document catalogs all breaking changes introduced in Virtual Agora v1.3. Developers must review and address each change to ensure successful migration from v1.1.

## Configuration Changes

### Required New Fields

#### Root Configuration
```yaml
# v1.1 (old) - WILL NOT WORK IN v1.3
moderator:
  provider: Google
  model: gemini-2.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2

# v1.3 (new) - REQUIRED STRUCTURE
moderator:
  provider: Google
  model: gemini-2.5-pro

# NEW REQUIRED FIELDS
summarizer:
  provider: OpenAI
  model: gpt-4o

topic_report:
  provider: Anthropic
  model: claude-3-opus-20240229

ecclesia_report:
  provider: Google
  model: gemini-2.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2
```

**Impact**: Applications will fail to start without these fields.

**Migration Path**: 
1. Use provided migration utility: `migrate_config_v1_to_v3()`
2. Or manually add required sections with appropriate models

### Deprecated Configuration Options

1. **Moderator Temperature/Max Tokens**
   - v1.1: Could be set on moderator
   - v1.3: Each specialized agent has its own settings
   
2. **Global Agent Settings**
   - v1.1: Single temperature for all agents
   - v1.3: Per-agent-type configuration

## API Changes

### ModeratorAgent Class

#### Constructor Changes
```python
# v1.1 - BREAKING CHANGE
moderator = ModeratorAgent(
    agent_id="moderator",
    llm=llm,
    mode="facilitation",  # REMOVED
    system_prompt=None,  # Now uses fixed v1.3 prompt
)

# v1.3 - NEW SIGNATURE
moderator = ModeratorAgent(
    agent_id="moderator",
    llm=llm,
    # No mode parameter
    # Fixed system prompt from spec
)
```

#### Removed Methods
The following methods have been removed from ModeratorAgent:

| Method | Replacement |
|--------|-------------|
| `set_mode()` | N/A - No modes |
| `get_current_mode()` | N/A - No modes |
| `generate_round_summary()` | `SummarizerAgent.summarize_round()` |
| `generate_topic_summary()` | `TopicReportAgent.synthesize_topic()` |
| `generate_report_structure()` | `EcclesiaReportAgent.generate_report_structure()` |
| `generate_section_content()` | `EcclesiaReportAgent.write_section()` |

#### Changed Method Signatures
```python
# v1.1
moderator.generate_response(prompt, mode="synthesis")

# v1.3
moderator.generate_response(prompt)  # No mode parameter
```

### New Required Classes

Applications must now instantiate these new classes:

```python
# v1.3 Required agents
from virtual_agora.agents import (
    SummarizerAgent,      # NEW
    TopicReportAgent,     # NEW
    EcclesiaReportAgent,  # NEW
)
```

### State Schema Changes

#### New Required Fields
```python
# VirtualAgoraState additions
specialized_agents: Dict[str, str]  # NEW: agent_type -> agent_id
round_summaries: List[RoundSummary]  # NEW: Structured summaries
periodic_stop_counter: int  # NEW: For 5-round stops
user_stop_history: List[UserStop]  # NEW: Stop tracking
```

#### Modified Fields
```python
# HITLState changes
approval_type: Optional[str]  # EXPANDED SET:
    # NEW: 'periodic_stop'
    # NEW: 'topic_conclusion_override'
    # NEW: 'agent_poll_override'
    # NEW: 'final_report_approval'
```

### Graph Node Changes

#### Node Function Signatures
```python
# v1.1 - Single agent parameter
def synthesis_node(state, moderator):
    return moderator.synthesize_agenda(state["votes"])

# v1.3 - Specialized agent parameters
def round_summarization_node(state, summarizer):
    return summarizer.summarize_round(...)

def topic_report_node(state, topic_report_agent):
    return topic_report_agent.synthesize_topic(...)
```

#### New Required Nodes
The following nodes must be implemented:
- `periodic_user_stop_node` - 5-round checkpoints
- `final_considerations_node` - Dissent handling
- `agent_poll_node` - Session end voting

## Behavior Changes

### Discussion Flow

1. **Periodic User Control**
   - v1.1: User could only intervene at topic end
   - v1.3: Automatic prompts every 5 rounds

2. **Topic Conclusion**
   - v1.1: Simple majority vote
   - v1.3: Majority + 1 required, user override possible

3. **Final Considerations**
   - v1.1: Always from dissenters only
   - v1.3: Context-dependent (all agents if user-forced)

### Report Generation

1. **Timing**
   - v1.1: Triggered by moderator mode switch
   - v1.3: Explicit nodes with specialized agents

2. **Structure**
   - v1.1: Single moderator decision
   - v1.3: EcclesiaReportAgent with JSON structure

### HITL Interactions

1. **Frequency**
   - v1.1: 3-4 interaction points
   - v1.3: 6-8 interaction points

2. **User Control**
   - v1.1: Limited override capability
   - v1.3: Enhanced override at multiple points

## Output Format Changes

### Log Files
```python
# v1.1 Log format
[2024-01-01 12:00:00] MODERATOR (synthesis): Generating agenda...

# v1.3 Log format  
[2024-01-01 12:00:00] SUMMARIZER: Compressing round 3...
[2024-01-01 12:00:01] TOPIC_REPORT: Synthesizing AI Safety...
```

### Report File Names
```python
# v1.1
"topic_summary_AI_Safety.md"
"final_report.md"

# v1.3
"agenda_summary_AI_Safety.md"  # 'agenda_' prefix
"final_report_01_Executive_Summary.md"  # Numbered sections
"final_report_02_Key_Themes.md"
```

## Import Changes

### Module Reorganization
```python
# v1.1
from virtual_agora.agents import ModeratorAgent

# v1.3 - Additional imports required
from virtual_agora.agents import (
    ModeratorAgent,      # Still exists but reduced
    SummarizerAgent,     # NEW
    TopicReportAgent,    # NEW
    EcclesiaReportAgent, # NEW
)
```

### Removed Imports
```python
# v1.1 - No longer valid
from virtual_agora.agents.moderator import ModeratorMode  # REMOVED
```

## Error Messages

### Configuration Errors
```
# v1.1
"ModeratorConfig missing required field: model"

# v1.3 - New error messages
"Config missing required field: summarizer"
"Config missing required field: topic_report"
"Config missing required field: ecclesia_report"
```

### Runtime Errors
```
# v1.1
"ModeratorAgent mode 'writer' not recognized"

# v1.3
"ModeratorAgent constructor got unexpected keyword argument 'mode'"
```

## Migration Tools

### Configuration Migration
```python
from virtual_agora.config.migration import migrate_config_v1_to_v3

# Automatic migration
old_config = load_yaml("config.yml")
new_config = migrate_config_v1_to_v3(old_config)
save_yaml("config_v3.yml", new_config)
```

### State Migration
```python
from virtual_agora.state.migration import migrate_state_v1_to_v3

# For resuming sessions
old_state = load_checkpoint("checkpoint_v1.pkl")
new_state = migrate_state_v1_to_v3(old_state)
```

### Compatibility Mode
```python
# Environment variable for gradual migration
export VIRTUAL_AGORA_COMPAT_MODE=v1

# Enables:
# - Warnings instead of errors for missing fields
# - Automatic config migration on load
# - Legacy method adapters
```

## Deprecation Timeline

### Immediate Breaking Changes (v1.3.0)
- ModeratorAgent `mode` parameter
- Mode-specific methods in ModeratorAgent
- Old configuration structure

### Deprecated with Warnings (removed in v1.4.0)
- Legacy state fields
- Old import paths
- v1.1 log format

### Future Removals (v1.5.0)
- Compatibility mode
- Migration utilities
- Legacy file name patterns

## Quick Migration Checklist

- [ ] Update configuration file with new agent sections
- [ ] Remove `mode` parameter from ModeratorAgent creation
- [ ] Create instances of new specialized agents
- [ ] Update graph nodes to use specialized agents
- [ ] Implement new HITL gates (periodic stops)
- [ ] Update imports for new agent classes
- [ ] Test with migration utilities
- [ ] Update logging/monitoring for new agents
- [ ] Verify report file name changes
- [ ] Review and update any custom modifications

## Support Resources

- Migration Scripts: `scripts/migrate_v1_to_v3.py`
- Example Configs: `examples/config_v3.yml`
- Test Suite: `tests/migration/`
- Support: GitHub Issues with tag `migration-v1.3`

---

**Document Version**: 1.0
**Last Updated**: [Current Date]
**Status**: Final