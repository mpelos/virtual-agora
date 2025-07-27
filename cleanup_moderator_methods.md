# ModeratorAgent Methods to Remove in v1.3 Cleanup

## Methods to Remove (Now in Specialized Agents)

### Synthesis Methods (→ SummarizerAgent)
- `generate_round_summary()` - Lines 1032-1110
- `generate_topic_summary()` - Lines 1111-1197  
- `_generate_topic_summary_map_reduce()` - Lines 1198-1255
- `extract_key_insights()` - Lines 1256-1388
- `generate_progressive_summary()` - Lines 1389-1459

### Agenda Synthesis Methods (→ Specialized handling)
- `request_topic_proposals()` - Lines 329-367 (uses synthesis mode)
- `collect_proposals()` - Lines 368-422 (uses synthesis mode)
- `request_votes()` - Lines 451-499 (uses synthesis mode)
- `synthesize_agenda()` - Lines 500-565 (uses synthesis mode)
- `synthesize_agenda_modifications()` - Lines 2581-2642 (async, uses synthesis mode)

### Report Writing Methods (→ EcclesiaReportAgent)
- `generate_report_structure()` - Lines 566-625
- `generate_section_content()` - Lines 626-675
- `define_report_structure()` - Lines 2456-2486 (async)
- `generate_report_section()` - Lines 2487-2530 (async)

### Methods with Synthesis Logic
- `tally_poll_results()` - Lines 2071-2210 (uses synthesis mode)
- `incorporate_minority_views()` - Lines 2401-2455 (uses synthesis mode)

### Methods that Need Mode Switching Removed
- `evaluate_message_relevance()` - Lines 1476-1596 (remove mode switch, keep functionality)

## Methods to Keep (Pure Facilitation)

### Core Facilitation
- `announce_topic()`
- `manage_turn_order()`
- `get_next_speaker()`
- `signal_round_completion()`

### Participation Tracking
- `track_participation()`
- `handle_agent_timeout()`
- `get_participation_summary()`

### Relevance Enforcement
- `set_topic_context()`
- `evaluate_message_relevance()` (after removing mode switch)
- `track_relevance_violation()`
- `issue_relevance_warning()`
- `mute_agent()`
- `check_agent_mute_status()`
- `process_message_for_relevance()`
- `get_relevance_enforcement_summary()`

### Polling (Keep but simplify)
- `initiate_conclusion_poll()`
- `cast_vote()`
- `check_poll_status()`
- `get_active_polls()`
- `handle_minority_considerations()`

### Utility Methods
- `generate_json_response()`
- `_validate_json_schema()`
- `parse_agenda_json()`
- `update_conversation_context()`
- `get_conversation_context()`
- `clear_conversation_context()`
- `validate_neutrality()`

## Summary
- **Total Methods to Remove**: ~16 methods
- **Total Methods to Keep**: ~24 methods
- **Lines to Remove**: ~1,500+ lines (rough estimate)
- **Expected Final Size**: ~500-800 lines (from current 2,768)