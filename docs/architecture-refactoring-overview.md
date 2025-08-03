# Virtual Agora Architecture Refactoring Overview

## Executive Summary

The Virtual Agora codebase suffers from architectural brittleness that makes simple changes complex and risky. This document outlines a structured refactoring plan to improve maintainability, reduce coupling, and enable easier feature development.

## Current Architecture Problems

### 1. Monolithic Node Design
- **Issue**: `nodes_v13.py` (1,185 lines) contains all flow logic
- **Impact**: High coupling, difficult testing, mixed concerns
- **Evidence**: Single file handles 15+ different node types

### 2. Scattered Round Management
- **Issue**: Round numbering logic exists in 4+ different files
- **Impact**: Inconsistent state, difficult to reason about timing
- **Evidence**: `current_round = state.get("current_round", 0) + 1` appears in multiple places

### 3. Tight Component Coupling
- **Issue**: Simple changes require modifications across 6+ files
- **Impact**: High change amplification, increased risk of bugs
- **Evidence**: Moving user participation from end-of-round to start-of-round requires touching graph, nodes, context, messages, state, and HITL systems

### 4. Mixed Abstraction Levels
- **Issue**: High-level flow control mixed with low-level implementation
- **Impact**: Difficult to understand system behavior, hard to modify
- **Evidence**: `discussion_round_node` handles flow control, message formatting, context building, and agent invocation

## Refactoring Goals

### Primary Objectives
1. **Reduce Coupling**: Make components independently modifiable
2. **Improve Cohesion**: Group related functionality together
3. **Centralize State**: Single source of truth for round/message state
4. **Separate Concerns**: Clear boundaries between flow, domain, and infrastructure

### Success Criteria
- **Simple changes stay simple**: Moving user participation should require 1-2 file changes
- **Clear ownership**: Each component has a single, well-defined responsibility
- **Testable components**: Each abstraction can be tested in isolation
- **Predictable behavior**: System state transitions are explicit and verifiable

## High-Level Refactoring Strategy

### Phase 1: Extract Core Abstractions (Week 1)
- Create dedicated round management
- Extract message coordination logic
- Establish clear state boundaries

### Phase 2: Restructure Flow Control (Week 2)
- Separate orchestration from implementation
- Create pluggable node architecture
- Centralize flow decision logic

### Phase 3: Consolidate Node Logic (Week 3)
- Break down monolithic `nodes_v13.py`
- Create focused, single-purpose nodes
- Implement clean interfaces between nodes

### Phase 4: Integration & Validation (Week 4)
- Integrate new architecture
- Comprehensive testing
- Performance validation
- Feature parity verification

## Target Architecture

### Core Abstractions
```
RoundManager: Centralized round state and transitions
FlowOrchestrator: High-level discussion flow coordination
NodeRegistry: Pluggable node architecture
MessageCoordinator: Message assembly and routing
ContextService: Context building and management
```

### Component Boundaries
```
Flow Layer:      Orchestration and routing logic
Domain Layer:    Business rules and state management  
Service Layer:   Message, context, and agent services
Infrastructure:  LangGraph integration, HITL, persistence
```

### Benefits After Refactoring
- **Easier feature development**: Clear extension points
- **Improved testability**: Isolated, mockable components
- **Better maintainability**: Single responsibility, loose coupling
- **Reduced risk**: Changes isolated to relevant components

## Implementation Approach

### Risk Mitigation
- **Incremental refactoring**: No big-bang changes
- **Feature preservation**: Maintain existing functionality throughout
- **Comprehensive testing**: Test-driven refactoring approach
- **Rollback capability**: Each phase can be rolled back independently

### Developer Guidelines
- **Test first**: Write tests before refactoring
- **Small steps**: Each commit should be atomic and reversible
- **Interface stability**: Maintain public interfaces during transition
- **Documentation**: Update docs with each phase

## Next Steps

Refer to the detailed implementation plan in `architecture-refactoring-steps.md` for specific tasks, acceptance criteria, and implementation guidance.

---

**Status**: Planning Phase  
**Last Updated**: August 2025  
**Review Date**: After each phase completion