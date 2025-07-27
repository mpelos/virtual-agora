# Virtual Agora v1.1 to v1.3 Migration Overview

## Executive Summary

This document provides a comprehensive guide for migrating Virtual Agora from version 1.1 (monolithic moderator architecture) to version 1.3 (node-centric architecture with specialized agents). The migration transforms a 2,861-line ModeratorAgent into five specialized agents, each focused on a single responsibility.

## Key Architectural Changes

### From v1.1 (Current)
- Single ModeratorAgent with internal modes (facilitation, synthesis, writer)
- Mode switching within one large class
- Graph nodes calling the same agent with different modes
- Complex state management within the agent

### To v1.3 (Target)
- Five specialized agents as tools:
  1. **ModeratorAgent**: Process facilitation only
  2. **SummarizerAgent**: Round compression
  3. **TopicReportAgent**: Agenda item synthesis
  4. **EcclesiaReportAgent**: Final report generation
  5. **DiscussingAgents**: Debate participants (unchanged)
- Graph nodes orchestrate by calling specific agents
- Clean separation of concerns
- Simplified state management

## Migration Phases

### Phase 1: Configuration & State Foundation
**Timeline**: 1-2 days
**Risk**: Low
**Dependencies**: None

Update configuration models and state schema to support the new architecture while maintaining backward compatibility.

### Phase 2: Specialized Agent Architecture
**Timeline**: 3-4 days
**Risk**: Medium
**Dependencies**: Phase 1

Extract functionality from the monolithic ModeratorAgent into specialized agent classes with focused responsibilities.

### Phase 3: Graph Flow Transformation
**Timeline**: 2-3 days
**Risk**: High
**Dependencies**: Phase 2

Rebuild the LangGraph flow to match the v1.3 specification with proper node-to-agent mappings.

### Phase 4: Enhanced HITL & UI Integration
**Timeline**: 2 days
**Risk**: Medium
**Dependencies**: Phase 3

Implement new HITL gates including periodic 5-round stops and enhanced user controls.

### Phase 5: Testing & Validation
**Timeline**: 2-3 days
**Risk**: Low
**Dependencies**: Phase 4

Comprehensive testing strategy ensuring functional parity and improved performance.

## Breaking Changes

### Configuration File Structure
The `config.yml` file now requires four specialized agent configurations:

```yaml
# v1.1 (old)
moderator:
  provider: Google
  model: gemini-2.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2

# v1.3 (new)
moderator:
  provider: Google
  model: gemini-2.5-pro

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

### API Changes
- `ModeratorAgent` no longer accepts `mode` parameter
- New agent classes must be instantiated separately
- Graph nodes now require all specialized agents during initialization

### State Schema Updates
- Enhanced HITL state for new approval types
- Additional fields for round summaries and topic reports
- New tracking for periodic user stops

## Success Metrics

1. **Code Quality**
   - ModeratorAgent reduced from 2,861 to ~500 lines
   - Each specialized agent under 500 lines
   - Clear single responsibility for each agent

2. **Performance**
   - No degradation in response times
   - Reduced memory footprint per agent
   - Better parallel processing capability

3. **Maintainability**
   - 80% reduction in complexity per component
   - Easier to test individual agents
   - Clear separation of concerns

4. **User Experience**
   - Enhanced control through new HITL gates
   - Better progress visibility
   - More granular session control

## Risk Mitigation

### Technical Risks
- **Data Loss**: Implement state migration tools
- **Performance Regression**: Benchmark before/after
- **Integration Issues**: Comprehensive integration testing

### Process Risks
- **Timeline Slip**: Buffer time between phases
- **Resource Availability**: Assign dedicated developers per phase
- **Scope Creep**: Strict adherence to v1.3 specification

## Rollback Strategy

Each phase includes rollback procedures:
1. Git branch protection for each phase
2. Database migration scripts with rollback
3. Configuration version detection
4. Feature flags for gradual rollout

## Developer Resources

- Original specification: `docs/project_spec.md` (v1.1)
- Target specification: `docs/project_spec_2.md` (v1.3)
- Phase-specific guides in `docs/migration_1.1_to_1.3/`
- Test scenarios in `tests/migration/`

## Communication Plan

1. **Pre-Migration**: Announce timeline and breaking changes
2. **During Migration**: Daily status updates
3. **Post-Migration**: User guide for new features
4. **Support**: Dedicated migration support channel

## Next Steps

1. Review and approve this migration plan
2. Assign developers to each phase
3. Set up migration branches and CI/CD
4. Begin Phase 1 implementation

---

**Document Version**: 1.0
**Last Updated**: [Current Date]
**Status**: DRAFT