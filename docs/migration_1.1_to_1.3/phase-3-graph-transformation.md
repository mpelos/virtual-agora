# Phase 3: Graph Flow Transformation Guide

## Purpose
This document provides detailed instructions for transforming the LangGraph flow from the v1.1 agent-centric model to the v1.3 node-centric architecture. The new design treats specialized agents as tools invoked by graph nodes, creating a cleaner separation between process orchestration and reasoning tasks.

## Prerequisites
- Completed Phase 1 (Configuration & State)
- Completed Phase 2 (Specialized Agents)
- Understanding of LangGraph concepts
- Access to v1.3 Mermaid flow diagram

## Current vs Target Architecture

### v1.1 Flow (Current)
- Nodes call ModeratorAgent with different modes
- Single agent handles multiple responsibilities
- Complex state management within agent
- Linear flow with basic conditionals

### v1.3 Flow (Target)
- Nodes orchestrate specialized agent invocations
- Each agent has single responsibility
- State managed by graph, not agents
- Complex conditionals with enhanced HITL

## Node Mapping Specification

### Phase 0: Initialization Nodes

```python
# src/virtual_agora/flow/nodes.py

class InitializationNodes:
    """Nodes for Phase 0: System initialization."""
    
    @staticmethod
    def config_and_keys_node(state: VirtualAgoraState) -> Dict[str, Any]:
        """Load configuration and API keys.
        
        This node:
        1. Loads .env file for API keys
        2. Parses config.yml
        3. Validates configuration
        4. Initializes logging
        """
        # No agent invocation - pure system setup
        return {"config_loaded": True}
    
    @staticmethod
    def agent_instantiation_node(
        state: VirtualAgoraState,
        specialized_agents: Dict[str, LLMAgent],
        discussing_agents: List[LLMAgent]
    ) -> Dict[str, Any]:
        """Create all agent instances.
        
        This node:
        1. Creates 5 specialized agents from config
        2. Creates N discussing agents from config
        3. Stores agent references in state
        """
        # Store agent IDs for later invocation
        return {
            "specialized_agents": {
                "moderator": specialized_agents["moderator"].agent_id,
                "summarizer": specialized_agents["summarizer"].agent_id,
                "topic_report": specialized_agents["topic_report"].agent_id,
                "ecclesia_report": specialized_agents["ecclesia_report"].agent_id,
            },
            "agents": {
                agent.agent_id: {
                    "id": agent.agent_id,
                    "model": agent.model_name,
                    "provider": agent.provider,
                    "role": "participant"
                }
                for agent in discussing_agents
            }
        }
    
    @staticmethod
    def get_theme_node(state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL node to get discussion theme.
        
        This is a special node that pauses execution for user input.
        """
        # Set HITL state for theme collection
        return {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "theme_input",
                "prompt_message": "Please enter the topic you would like the agents to discuss:",
                "options": None
            }
        }
```

### Phase 1: Agenda Setting Nodes

```python
class AgendaNodes:
    """Nodes for Phase 1: Democratic agenda setting."""
    
    @staticmethod
    def agenda_proposal_node(
        state: VirtualAgoraState,
        discussing_agents: List[LLMAgent]
    ) -> Dict[str, Any]:
        """Request topic proposals from discussing agents.
        
        This node:
        1. Prompts each discussing agent for 3-5 topics
        2. Collects all proposals
        3. Updates state with raw proposals
        """
        theme = state["main_topic"]
        proposals = []
        
        for agent in discussing_agents:
            prompt = f"""Based on the theme '{theme}', propose 3-5 specific 
            sub-topics for discussion. Be concise and specific."""
            
            response = agent.generate_response(prompt)
            proposals.append({
                "agent_id": agent.agent_id,
                "proposals": response
            })
        
        return {"proposed_topics": proposals}
    
    @staticmethod
    def collate_proposals_node(
        state: VirtualAgoraState,
        moderator: ModeratorAgent
    ) -> Dict[str, Any]:
        """Moderator deduplicates and compiles proposals.
        
        This node invokes the ModeratorAgent as a tool to:
        1. Read all proposals
        2. Remove duplicates
        3. Create unified list
        """
        proposals = state["proposed_topics"]
        
        # Invoke moderator as a tool
        unified_list = moderator.collect_proposals(proposals)
        
        return {"collated_topics": unified_list}
    
    @staticmethod
    def agenda_voting_node(
        state: VirtualAgoraState,
        discussing_agents: List[LLMAgent]
    ) -> Dict[str, Any]:
        """Collect votes on agenda ordering.
        
        Similar to proposal collection but for votes.
        """
        topics = state["collated_topics"]
        votes = []
        
        for agent in discussing_agents:
            prompt = f"""Vote on your preferred discussion order for these topics:
            {topics}
            
            Express your preferences in natural language."""
            
            response = agent.generate_response(prompt)
            votes.append({
                "agent_id": agent.agent_id,
                "vote": response
            })
        
        return {"agenda_votes": votes}
    
    @staticmethod
    def synthesize_agenda_node(
        state: VirtualAgoraState,
        moderator: ModeratorAgent
    ) -> Dict[str, Any]:
        """Moderator synthesizes votes into final agenda.
        
        This node invokes ModeratorAgent to:
        1. Analyze all votes
        2. Break ties
        3. Produce JSON agenda
        """
        votes = state["agenda_votes"]
        
        # Invoke moderator for synthesis
        agenda_json = moderator.synthesize_agenda(votes)
        
        return {"proposed_agenda": agenda_json["proposed_agenda"]}
    
    @staticmethod
    def agenda_approval_node(state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL node for user agenda approval.
        
        Pauses for user to approve or edit agenda.
        """
        agenda = state["proposed_agenda"]
        
        return {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "agenda_approval",
                "prompt_message": f"Proposed agenda: {agenda}\nApprove or edit?",
                "options": ["approve", "edit"]
            }
        }
```

### Phase 2: Discussion Loop Nodes

```python
class DiscussionNodes:
    """Nodes for Phase 2: Topic discussion."""
    
    @staticmethod
    def announce_item_node(
        state: VirtualAgoraState,
        moderator: ModeratorAgent
    ) -> Dict[str, Any]:
        """Announce current agenda item.
        
        Simple announcement, no complex logic.
        """
        current_topic = state["active_topic"]
        round_num = state["current_round"]
        
        announcement = moderator.announce_topic(current_topic, round_num)
        
        return {"last_announcement": announcement}
    
    @staticmethod
    def discussion_round_node(
        state: VirtualAgoraState,
        discussing_agents: List[LLMAgent],
        moderator: ModeratorAgent
    ) -> Dict[str, Any]:
        """Execute one round of discussion.
        
        This node:
        1. Manages turn rotation
        2. Collects agent responses
        3. Enforces relevance
        """
        # Get context for agents
        theme = state["main_topic"]
        topic = state["active_topic"]
        round_summaries = state.get("round_summaries", [])
        current_round_messages = []
        
        # Rotate speaking order
        speaking_order = state["speaking_order"]
        if state["current_round"] > 1:
            # Rotate: [A,B,C] -> [B,C,A]
            speaking_order = speaking_order[1:] + speaking_order[:1]
        
        # Collect responses
        for agent_id in speaking_order:
            agent = next(a for a in discussing_agents if a.agent_id == agent_id)
            
            # Build context
            context = f"""
            Theme: {theme}
            Current Topic: {topic}
            
            Previous Round Summaries:
            {chr(10).join(round_summaries)}
            
            Current Round Comments:
            {chr(10).join([f"{m['speaker']}: {m['content']}" for m in current_round_messages])}
            """
            
            # Get response
            response = agent.generate_response(
                f"{context}\n\nProvide your thoughts on '{topic}'."
            )
            
            # Check relevance
            relevance_check = moderator.evaluate_message_relevance(
                response, topic
            )
            
            if relevance_check["is_relevant"]:
                current_round_messages.append({
                    "speaker": agent_id,
                    "content": response
                })
            else:
                # Handle warning/muting
                warning = moderator.issue_relevance_warning(agent_id)
                current_round_messages.append({
                    "speaker": "moderator",
                    "content": warning
                })
        
        return {
            "messages": current_round_messages,
            "speaking_order": speaking_order
        }
    
    @staticmethod
    def round_summarization_node(
        state: VirtualAgoraState,
        summarizer: SummarizerAgent
    ) -> Dict[str, Any]:
        """Invoke Summarizer to compress round.
        
        This node specifically invokes the SummarizerAgent.
        """
        messages = state["messages"][-state["agents_count"]:]  # Last round
        topic = state["active_topic"]
        round_num = state["current_round"]
        
        # Invoke summarizer as a tool
        summary = summarizer.summarize_round(
            messages=messages,
            topic=topic,
            round_number=round_num
        )
        
        return {
            "round_summaries": [summary],  # Appended via reducer
            "last_round_summary": summary
        }
    
    @staticmethod
    def end_topic_poll_node(
        state: VirtualAgoraState,
        discussing_agents: List[LLMAgent]
    ) -> Dict[str, Any]:
        """Poll agents on topic conclusion.
        
        Only triggered after round 3+.
        """
        topic = state["active_topic"]
        votes = []
        
        for agent in discussing_agents:
            prompt = f"""Should we conclude the discussion on '{topic}'? 
            Please respond with 'Yes' or 'No' and a short justification."""
            
            response = agent.generate_response(prompt)
            
            # Parse vote
            vote = "yes" if "yes" in response.lower()[:10] else "no"
            votes.append({
                "agent_id": agent.agent_id,
                "vote": vote,
                "justification": response
            })
        
        return {"topic_conclusion_votes": votes}
    
    @staticmethod
    def periodic_user_stop_node(state: VirtualAgoraState) -> Dict[str, Any]:
        """HITL node for 5-round periodic stops.
        
        New in v1.3 - gives user periodic control.
        """
        return {
            "hitl_state": {
                "awaiting_approval": True,
                "approval_type": "periodic_stop",
                "prompt_message": "Do you wish to end the current agenda item discussion?",
                "options": ["yes", "no"]
            },
            "periodic_stop_counter": 0  # Reset counter
        }
```

### Phase 3: Topic Conclusion Nodes

```python
class ConclusionNodes:
    """Nodes for Phase 3: Topic conclusion and reporting."""
    
    @staticmethod
    def final_considerations_node(
        state: VirtualAgoraState,
        discussing_agents: List[LLMAgent]
    ) -> Dict[str, Any]:
        """Collect final thoughts from agents.
        
        Logic varies based on how conclusion was triggered:
        - Vote-based: Only dissenting agents
        - User-forced: All agents
        """
        topic = state["active_topic"]
        final_thoughts = []
        
        # Determine which agents to prompt
        if state.get("user_forced_conclusion"):
            # User forced - all agents
            agents_to_prompt = discussing_agents
        else:
            # Vote based - only dissenters
            votes = state["topic_conclusion_votes"]
            dissenter_ids = [
                v["agent_id"] for v in votes if v["vote"] == "no"
            ]
            agents_to_prompt = [
                a for a in discussing_agents if a.agent_id in dissenter_ids
            ]
        
        # Collect final thoughts
        for agent in agents_to_prompt:
            prompt = f"""The discussion on '{topic}' is concluding. 
            Please provide your final considerations on this topic."""
            
            response = agent.generate_response(prompt)
            final_thoughts.append({
                "agent_id": agent.agent_id,
                "consideration": response
            })
        
        return {"final_considerations": final_thoughts}
    
    @staticmethod
    def topic_report_generation_node(
        state: VirtualAgoraState,
        topic_report_agent: TopicReportAgent
    ) -> Dict[str, Any]:
        """Invoke Topic Report Agent for synthesis.
        
        This node specifically invokes the TopicReportAgent.
        """
        topic = state["active_topic"]
        theme = state["main_topic"]
        round_summaries = [
            s for s in state["round_summaries"] 
            if s.get("topic") == topic
        ]
        final_considerations = state.get("final_considerations", [])
        
        # Invoke topic report agent
        report = topic_report_agent.synthesize_topic(
            round_summaries=[s["content"] for s in round_summaries],
            final_considerations=[
                f["consideration"] for f in final_considerations
            ],
            topic=topic,
            discussion_theme=theme
        )
        
        return {
            "topic_summaries": {topic: report},
            "last_topic_report": report
        }
    
    @staticmethod
    def file_output_node(state: VirtualAgoraState) -> Dict[str, Any]:
        """Save topic report to file.
        
        Pure I/O operation, no agent invocation.
        """
        topic = state["active_topic"]
        report = state["last_topic_report"]
        
        # Generate filename
        safe_topic = topic.replace(" ", "_").replace("/", "_")
        filename = f"agenda_summary_{safe_topic}.md"
        
        # Save file
        with open(f"reports/{filename}", "w") as f:
            f.write(f"# Topic Report: {topic}\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(report)
        
        return {"topic_report_saved": filename}
```

### Conditional Edge Logic

```python
# src/virtual_agora/flow/edges.py

class EnhancedConditions:
    """Enhanced conditional logic for v1.3 flow."""
    
    @staticmethod
    def check_round_threshold(state: VirtualAgoraState) -> str:
        """Determine if polling should start (round >= 3)."""
        if state["current_round"] >= 3:
            return "start_polling"
        return "continue_discussion"
    
    @staticmethod
    def check_periodic_stop(state: VirtualAgoraState) -> str:
        """Check if it's time for 5-round user stop."""
        if state["current_round"] % 5 == 0:
            return "periodic_stop"
        return "check_votes"
    
    @staticmethod
    def evaluate_conclusion_vote(state: VirtualAgoraState) -> str:
        """Evaluate both agent votes and user override."""
        # Check user override first
        if state.get("user_forced_conclusion"):
            return "conclude_topic"
        
        # Check agent votes
        votes = state.get("topic_conclusion_votes", [])
        yes_votes = sum(1 for v in votes if v["vote"] == "yes")
        total_votes = len(votes)
        
        # Majority + 1 required
        if yes_votes > (total_votes / 2):
            return "conclude_topic"
        
        return "continue_discussion"
    
    @staticmethod
    def check_agenda_remaining(state: VirtualAgoraState) -> str:
        """Check if more agenda items remain."""
        current_index = state["agenda"]["current_topic_index"]
        total_topics = len(state["agenda"]["topics"])
        
        if current_index >= total_topics - 1:
            return "no_items_remaining"
        return "items_remaining"
    
    @staticmethod
    def evaluate_session_continuation(state: VirtualAgoraState) -> str:
        """Evaluate agent and user decisions on continuation."""
        # Agent vote to end session
        if state.get("agents_vote_end_session"):
            return "end_session"
        
        # User decision
        if not state.get("user_approves_continuation"):
            return "end_session"
        
        # Check if agenda is empty
        if not state.get("remaining_topics"):
            return "end_session"
        
        return "continue_session"
```

### Graph Construction

```python
# src/virtual_agora/flow/graph.py

class VirtualAgoraFlow:
    """Main flow orchestrator for v1.3."""
    
    def build_graph(self) -> StateGraph:
        """Construct the complete v1.3 flow graph."""
        
        # Create graph
        graph = StateGraph(VirtualAgoraState)
        
        # Phase 0: Initialization
        graph.add_node("config_and_keys", self.nodes.config_and_keys_node)
        graph.add_node("agent_instantiation", self.nodes.agent_instantiation_node)
        graph.add_node("get_theme", self.nodes.get_theme_node)
        
        # Phase 1: Agenda Setting
        graph.add_node("agenda_proposal", self.nodes.agenda_proposal_node)
        graph.add_node("collate_proposals", self.nodes.collate_proposals_node)
        graph.add_node("agenda_voting", self.nodes.agenda_voting_node)
        graph.add_node("synthesize_agenda", self.nodes.synthesize_agenda_node)
        graph.add_node("agenda_approval", self.nodes.agenda_approval_node)
        
        # Phase 2: Discussion Loop
        graph.add_node("announce_item", self.nodes.announce_item_node)
        graph.add_node("discussion_round", self.nodes.discussion_round_node)
        graph.add_node("round_summarization", self.nodes.round_summarization_node)
        graph.add_node("end_topic_poll", self.nodes.end_topic_poll_node)
        graph.add_node("periodic_user_stop", self.nodes.periodic_user_stop_node)
        
        # Phase 3: Topic Conclusion
        graph.add_node("final_considerations", self.nodes.final_considerations_node)
        graph.add_node("topic_report_generation", self.nodes.topic_report_generation_node)
        graph.add_node("file_output", self.nodes.file_output_node)
        
        # Phase 4: Continuation
        graph.add_node("agent_poll", self.nodes.agent_poll_node)
        graph.add_node("user_approval", self.nodes.user_approval_node)
        graph.add_node("agenda_modification", self.nodes.agenda_modification_node)
        
        # Phase 5: Final Report
        graph.add_node("final_report", self.nodes.final_report_node)
        graph.add_node("multi_file_output", self.nodes.multi_file_output_node)
        
        # Add edges (matching v1.3 Mermaid diagram)
        graph.add_edge(START, "config_and_keys")
        graph.add_edge("config_and_keys", "agent_instantiation")
        graph.add_edge("agent_instantiation", "get_theme")
        graph.add_edge("get_theme", "agenda_proposal")
        
        # Conditional edges
        graph.add_conditional_edges(
            "round_summarization",
            self.conditions.check_round_threshold,
            {
                "start_polling": "end_topic_poll",
                "continue_discussion": "discussion_round"
            }
        )
        
        graph.add_conditional_edges(
            "end_topic_poll",
            self.conditions.check_periodic_stop,
            {
                "periodic_stop": "periodic_user_stop",
                "check_votes": "evaluate_votes"
            }
        )
        
        # ... (complete edge definitions)
        
        return graph.compile()
```

## Implementation Instructions

### Development Workflow

1. **Update Node Classes**
   - Create new node classes for v1.3 functionality
   - Ensure each node has single responsibility
   - Nodes invoke agents, don't contain logic

2. **Update Edge Conditions**
   - Implement new conditional logic
   - Add periodic stop checks
   - Handle user overrides

3. **Wire the Graph**
   - Follow v1.3 Mermaid diagram exactly
   - Test each path through the graph
   - Validate state updates

### Agent Invocation Patterns

```python
# Pattern 1: Simple agent invocation
def some_node(state, agent):
    result = agent.do_task(state["input"])
    return {"output": result}

# Pattern 2: Multiple agent coordination
def complex_node(state, agent1, agent2):
    step1 = agent1.first_task(state["data"])
    step2 = agent2.second_task(step1)
    return {"final_result": step2}

# Pattern 3: Conditional agent invocation
def conditional_node(state, agents):
    if state["condition"]:
        result = agents["type_a"].process()
    else:
        result = agents["type_b"].process()
    return {"result": result}
```

### Testing Strategy

1. **Node Unit Tests**
   ```python
   def test_summarization_node():
       """Test summarizer invocation."""
       state = create_test_state()
       summarizer = Mock(spec=SummarizerAgent)
       
       result = round_summarization_node(state, summarizer)
       
       summarizer.summarize_round.assert_called_once()
       assert "round_summaries" in result
   ```

2. **Path Integration Tests**
   ```python
   def test_complete_topic_flow():
       """Test full topic discussion to report."""
       graph = build_test_graph()
       
       # Run through complete topic
       result = graph.invoke({
           "main_topic": "AI Safety",
           "active_topic": "Alignment",
           "current_round": 1
       })
       
       assert "topic_report_saved" in result
   ```

## Common Challenges and Solutions

### Challenge 1: State Size Growth
**Problem**: State grows with each round
**Solution**: Implement state pruning in nodes

### Challenge 2: Agent Coordination
**Problem**: Agents need shared context
**Solution**: Pass context explicitly, don't share state

### Challenge 3: HITL Complexity
**Problem**: Multiple HITL gates complicate flow
**Solution**: Centralized HITL handling pattern

### Challenge 4: Error Propagation
**Problem**: Agent errors break flow
**Solution**: Try-except in nodes with graceful degradation

## Validation Checklist

- [ ] All v1.3 Mermaid nodes implemented
- [ ] Specialized agents invoked correctly
- [ ] Enhanced conditionals working
- [ ] Periodic stops triggering at round % 5
- [ ] Dual polling system functional
- [ ] State updates match specification
- [ ] All paths through graph tested
- [ ] Performance acceptable

## References

- v1.3 Flow Diagram: Section 8 of `docs/project_spec_2.md`
- Current Graph: `src/virtual_agora/flow/graph.py`
- Node Implementation: `src/virtual_agora/flow/nodes.py`
- Edge Conditions: `src/virtual_agora/flow/edges.py`

## Next Phase

Once graph transformation is complete and tested, proceed to Phase 4: HITL & UI Enhancement.

---

**Document Version**: 1.0
**Phase**: 3 of 5
**Status**: Implementation Guide