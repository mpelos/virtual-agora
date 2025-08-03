"""Response templates and patterns for deterministic LLMs.

This module contains all the response templates and context matching patterns
used by deterministic LLMs to generate realistic, consistent responses.
"""

from typing import Dict, List, Any
import re

# Moderator response templates
MODERATOR_RESPONSES = {
    "agenda_proposal": """Based on the topic "{last_user_message}", I propose the following discussion agenda:

1. **Core Concepts and Definitions** - Establish foundational understanding
2. **Current State Analysis** - Examine the present situation
3. **Key Challenges and Opportunities** - Identify critical issues
4. **Potential Solutions and Approaches** - Explore viable options
5. **Implementation Considerations** - Discuss practical aspects

This agenda will allow us to have a structured and comprehensive discussion. Each topic should provide valuable insights for our participants.""",
    "round_summary": """## Round {call_count} Summary

In this discussion round, our participants covered several important points:

**Key Themes Discussed:**
- Foundational concepts and their implications
- Practical challenges in implementation
- Different perspectives on potential solutions

**Notable Insights:**
- The complexity of the topic requires careful consideration
- Multiple approaches may be needed for effective solutions
- Stakeholder perspectives vary significantly

**Areas for Further Exploration:**
- Implementation timelines and resource requirements
- Risk mitigation strategies
- Success metrics and evaluation criteria

The discussion has been productive and we're making good progress toward comprehensive understanding.""",
    "voting_decision": """Based on the discussion so far, I recommend we **continue** with this topic.

**Rationale:**
- The participants have raised valuable points that merit further exploration
- We have not yet fully exhausted the depth of this subject
- Additional rounds would benefit from the insights already shared

**Suggested Focus for Next Round:**
- Building on the foundational concepts already established
- Diving deeper into implementation specifics
- Addressing any remaining questions or concerns

This topic continues to offer significant value for our discussion.""",
    "default": """As the moderator, I'm here to facilitate our discussion and ensure we make progress on the topic at hand. Let me help guide our conversation toward productive outcomes.""",
}

# Discussion agent response templates
AGENT_RESPONSES = {
    "discussion_contribution": """Thank you for the opportunity to contribute to this discussion.

I believe the topic of "{last_user_message}" is particularly important because it touches on fundamental aspects that affect how we approach similar challenges. 

**My perspective on this matter:**

From my analysis, there are several key considerations we should examine:

1. **Practical Implementation** - We need to consider how theoretical concepts translate into real-world applications
2. **Stakeholder Impact** - Different groups will be affected in various ways by any decisions we make
3. **Resource Requirements** - Both human and material resources must be factored into our planning

**Specific insights I'd like to contribute:**

The most effective approaches I've observed tend to balance innovation with proven methodologies. While it's tempting to pursue entirely novel solutions, there's significant value in building upon established foundations.

Additionally, I think we should consider the scalability of any proposed solutions. What works in limited scenarios may face challenges when applied more broadly.

I look forward to hearing other perspectives on these points.""",
    "user_participation": """I appreciate your input on this matter. Your perspective adds valuable context to our discussion.

Building on what you've shared, I'd like to add that this connects to broader patterns we often see in similar situations. The interconnected nature of these issues means that solutions must be comprehensive rather than targeting isolated aspects.

**Key connections I see:**
- Your point about implementation aligns with my earlier observations about practical considerations
- The resource implications you mention are particularly relevant to sustainability
- The timeline aspects deserve careful attention in our planning

**Additional considerations:**
I think we should also explore how this might affect different stakeholder groups and consider their varying perspectives and needs.

Thank you for enriching our conversation with your insights.""",
    "round_conclusion": """To summarize my position on this topic:

**Core Arguments:**
1. The complexity of this subject requires a multi-faceted approach
2. Practical implementation must be balanced with theoretical soundness
3. Stakeholder considerations are paramount for successful outcomes

**Key Insights Gained:**
Through this discussion, I've come to appreciate the nuanced nature of the challenges we face. The various perspectives shared have highlighted both opportunities and potential pitfalls.

**Looking Forward:**
I believe we've established a solid foundation for further exploration. The next logical steps would involve deeper examination of the implementation strategies we've touched upon.

**Final Thought:**
The collaborative nature of this discussion has been valuable - the diverse viewpoints have enriched my understanding significantly.""",
    "default": """I appreciate the opportunity to participate in this discussion. Based on the conversation so far, I think there are several important aspects we should consider as we move forward.""",
}

# Summarizer response templates
SUMMARIZER_RESPONSES = {
    "round_summary": """## Discussion Round {call_count} Summary

**Topic Focus:** {active_topic}

**Participants:** {agent_count} discussion agents contributed perspectives

**Key Discussion Points:**

1. **Foundational Concepts**
   - Participants established core definitions and frameworks
   - Multiple perspectives were shared on fundamental principles
   - Common ground was identified for further exploration

2. **Implementation Considerations**
   - Practical challenges and opportunities were examined
   - Resource requirements and constraints were discussed
   - Timeline and sequencing factors were considered

3. **Stakeholder Perspectives**
   - Various viewpoints were represented in the discussion
   - Different impact scenarios were explored
   - Consensus areas and disagreements were identified

**Emerging Themes:**
- The complexity of the topic requires comprehensive approaches
- Balance between innovation and proven methods is crucial
- Stakeholder engagement is essential for successful outcomes

**Quality of Discussion:**
The conversation maintained focus while allowing for diverse perspectives. Participants built effectively on each other's contributions.

**Readiness for Next Phase:**
Based on the depth and quality of contributions, this topic appears ready for {"continued exploration" if call_count < 5 else "conclusion and summary"}.""",
    "topic_conclusion": """## Topic Conclusion Summary: {active_topic}

**Discussion Overview:**
This topic was explored through {call_count} rounds of discussion, with comprehensive coverage of key aspects and considerations.

**Major Insights Achieved:**

1. **Conceptual Clarity**
   - Core definitions and frameworks were established
   - Different approaches and methodologies were examined
   - Theoretical foundations were connected to practical applications

2. **Implementation Pathway**
   - Practical steps and considerations were identified
   - Resource requirements and constraints were mapped
   - Risk factors and mitigation strategies were discussed

3. **Stakeholder Considerations**
   - Multiple perspectives were represented and analyzed
   - Impact scenarios across different groups were explored
   - Consensus areas and outstanding concerns were documented

**Key Recommendations:**
Based on the discussion, the following recommendations emerged:
- Pursue a balanced approach combining innovation with proven methods
- Ensure comprehensive stakeholder engagement throughout implementation
- Maintain flexibility to adapt based on feedback and changing circumstances

**Discussion Quality Assessment:**
The conversation was thorough and well-structured, with participants effectively building on each other's contributions to reach meaningful insights.

**Status:** Ready for final report compilation.""",
    "default": """This summary captures the key points and insights from our recent discussion round. The participants have made valuable contributions toward understanding the topic at hand.""",
}

# Report writer response templates
REPORT_WRITER_RESPONSES = {
    "final_report": """# Virtual Agora Discussion Report

## Executive Summary

This report presents the outcomes of a structured multi-agent discussion on the topic: **{main_topic}**

The discussion was conducted through {total_rounds} rounds of conversation involving {agent_count} discussion agents, with comprehensive moderation and summarization to ensure productive dialogue.

## Discussion Overview

**Session Information:**
- Topic: {main_topic}
- Total Rounds: {total_rounds}
- Participants: {agent_count} AI agents plus human moderator
- Duration: {discussion_duration}
- Date: {session_date}

## Key Insights and Findings

### 1. Foundational Understanding
The discussion established clear foundational concepts and frameworks necessary for addressing the topic. Participants demonstrated comprehensive understanding of core principles and their implications.

### 2. Practical Implementation
Significant attention was given to practical implementation considerations, including:
- Resource requirements and constraints
- Timeline considerations and sequencing
- Risk factors and mitigation strategies
- Success metrics and evaluation criteria

### 3. Stakeholder Perspectives
The conversation thoroughly explored various stakeholder viewpoints:
- Different impact scenarios were examined
- Diverse perspectives were represented and analyzed
- Areas of consensus and disagreement were identified
- Engagement strategies were discussed

## Major Recommendations

Based on the comprehensive discussion, the following key recommendations emerged:

1. **Balanced Approach**: Pursue solutions that combine innovative thinking with proven methodologies
2. **Stakeholder Engagement**: Ensure comprehensive involvement of all affected parties throughout the process
3. **Adaptive Implementation**: Maintain flexibility to adjust approaches based on feedback and changing circumstances
4. **Continuous Evaluation**: Implement robust monitoring and evaluation mechanisms to track progress

## Discussion Quality Assessment

The multi-agent discussion demonstrated:
- **Depth**: Complex topics were explored thoroughly across multiple dimensions
- **Breadth**: Various perspectives and approaches were considered comprehensively  
- **Coherence**: Participants effectively built upon each other's contributions
- **Practicality**: Theoretical insights were consistently connected to real-world applications

## Conclusion

This discussion successfully achieved its objectives of exploring {main_topic} from multiple perspectives and developing actionable insights. The collaborative approach yielded comprehensive understanding and practical recommendations for moving forward.

The structured format facilitated productive dialogue while ensuring all key aspects received appropriate attention. The resulting insights provide a solid foundation for future decision-making and implementation efforts.

---
*Report generated by Virtual Agora Multi-Agent Discussion System*
*Session ID: {session_id}*""",
    "topic_report": """# Topic Report: {active_topic}

## Summary
This report covers the discussion of {active_topic} conducted over {rounds_for_topic} rounds.

## Key Points Discussed
- Foundational concepts and definitions were established
- Implementation challenges and opportunities were explored
- Various stakeholder perspectives were considered
- Practical solutions and approaches were evaluated

## Main Insights
The discussion revealed the complexity and multi-faceted nature of this topic, with participants offering diverse but complementary perspectives that contributed to a comprehensive understanding.

## Recommendations
Based on the discussion, a balanced approach combining theoretical rigor with practical considerations appears most promising for addressing the challenges identified.

## Next Steps
The insights from this topic discussion provide valuable input for overall session conclusions and recommendations.""",
    "default": """This report summarizes the key outcomes and insights from our discussion session. The collaborative approach has yielded valuable perspectives on the topics explored.""",
}

# Context patterns for matching responses to situations
CONTEXT_PATTERNS = {
    "moderator": {
        "agenda_proposal": {
            "keywords": ["agenda", "propose", "topic", "discuss", "structure"],
            "message_patterns": [
                r".*agenda.*",
                r".*structure.*discussion.*",
                r".*propose.*topic.*",
            ],
            "system_prompt_patterns": [r".*moderator.*", r".*facilitate.*"],
            "call_count_range": (1, 3),
        },
        "round_summary": {
            "keywords": ["summary", "summarize", "round", "progress", "recap"],
            "message_patterns": [r".*summar.*", r".*recap.*", r".*progress.*"],
            "call_count_range": (2, 10),
        },
        "voting_decision": {
            "keywords": ["vote", "decision", "continue", "proceed", "next"],
            "message_patterns": [r".*vote.*", r".*decision.*", r".*continue.*"],
            "call_count_range": (3, 15),
        },
    },
    "agent": {
        "discussion_contribution": {
            "keywords": ["discuss", "perspective", "view", "opinion", "analysis"],
            "message_patterns": [r".*discuss.*", r".*perspective.*", r".*view.*"],
            "system_prompt_patterns": [r".*participant.*", r".*agent.*"],
            "call_count_range": (1, 20),
        },
        "user_participation": {
            "keywords": ["user", "human", "input", "feedback", "response"],
            "message_patterns": [r".*user.*", r".*human.*", r".*input.*"],
            "call_count_range": (1, 20),
        },
        "round_conclusion": {
            "keywords": ["conclude", "summary", "position", "final", "end"],
            "message_patterns": [r".*conclud.*", r".*summar.*", r".*final.*"],
            "call_count_range": (3, 20),
        },
    },
    "summarizer": {
        "round_summary": {
            "keywords": ["summarize", "summary", "round", "discussion"],
            "message_patterns": [r".*summar.*", r".*round.*", r".*discussion.*"],
            "system_prompt_patterns": [r".*summariz.*", r".*summary.*"],
            "call_count_range": (1, 15),
        },
        "topic_conclusion": {
            "keywords": ["conclude", "conclusion", "topic", "final", "complete"],
            "message_patterns": [r".*conclud.*", r".*final.*", r".*complete.*"],
            "call_count_range": (3, 15),
        },
    },
    "report_writer": {
        "final_report": {
            "keywords": ["report", "final", "complete", "document", "summary"],
            "message_patterns": [r".*report.*", r".*final.*", r".*document.*"],
            "system_prompt_patterns": [r".*report.*", r".*document.*"],
            "call_count_range": (1, 5),
        },
        "topic_report": {
            "keywords": ["topic", "report", "document", "summary"],
            "message_patterns": [r".*topic.*report.*", r".*document.*topic.*"],
            "call_count_range": (1, 10),
        },
    },
}

# Interrupt trigger patterns
INTERRUPT_PATTERNS = {
    "agenda_approval": {
        "triggers": ["agenda", "proposal", "approve"],
        "type": "agenda_approval",
        "data": {
            "proposed_agenda": [
                "Core Concepts and Definitions",
                "Current State Analysis",
                "Key Challenges and Opportunities",
                "Potential Solutions and Approaches",
                "Implementation Considerations",
            ]
        },
    },
    "user_turn_participation": {
        "triggers": ["user", "participation", "turn", "input"],
        "type": "user_turn_participation",
        "data": {
            "current_round": 3,
            "current_topic": "Discussion Topic",
            "previous_summary": "Previous discussion summary",
        },
    },
    "topic_continuation": {
        "triggers": ["continue", "next", "topic", "proceed"],
        "type": "topic_continuation",
        "data": {
            "completed_topic": "Current Topic",
            "remaining_topics": ["Topic 1", "Topic 2"],
            "agent_recommendation": "continue",
        },
    },
    "periodic_stop": {
        "triggers": ["checkpoint", "stop", "check", "status"],
        "type": "periodic_stop",
        "data": {
            "current_round": 3,
            "current_topic": "Discussion Topic",
            "checkpoint_interval": 3,
        },
    },
}


def get_response_templates(role: str) -> Dict[str, str]:
    """Get response templates for a specific role.

    Args:
        role: The role name (moderator, agent, summarizer, report_writer)

    Returns:
        Dictionary of response templates for the role
    """
    role_templates = {
        "moderator": MODERATOR_RESPONSES,
        "agent": AGENT_RESPONSES,
        "summarizer": SUMMARIZER_RESPONSES,
        "report_writer": REPORT_WRITER_RESPONSES,
    }
    return role_templates.get(
        role, {"default": "I am a {role} agent responding to your message."}
    )


def get_context_patterns(role: str) -> Dict[str, Dict[str, Any]]:
    """Get context matching patterns for a specific role.

    Args:
        role: The role name

    Returns:
        Dictionary of context patterns for the role
    """
    return CONTEXT_PATTERNS.get(role, {})


def get_interrupt_patterns() -> Dict[str, Dict[str, Any]]:
    """Get interrupt trigger patterns.

    Returns:
        Dictionary of interrupt patterns
    """
    return INTERRUPT_PATTERNS


def match_context_to_pattern(
    context: Dict[str, Any], role: str, pattern_name: str
) -> bool:
    """Check if context matches a specific pattern for a role.

    Args:
        context: Extracted context information
        role: The role name
        pattern_name: The pattern name to check

    Returns:
        True if context matches the pattern
    """
    patterns = get_context_patterns(role)
    if pattern_name not in patterns:
        return False

    pattern = patterns[pattern_name]

    # Check call count range
    call_count_range = pattern.get("call_count_range")
    if call_count_range:
        min_calls, max_calls = call_count_range
        if not (min_calls <= context.get("call_count", 0) <= max_calls):
            return False

    # Check keywords in last message
    keywords = pattern.get("keywords", [])
    last_message = context.get("last_user_message", "").lower()
    if keywords and any(keyword in last_message for keyword in keywords):
        return True

    # Check message patterns
    message_patterns = pattern.get("message_patterns", [])
    for pattern_regex in message_patterns:
        if re.search(pattern_regex, last_message, re.IGNORECASE):
            return True

    # Check system prompt patterns
    system_prompt_patterns = pattern.get("system_prompt_patterns", [])
    system_prompt = context.get("system_prompt", "").lower()
    for pattern_regex in system_prompt_patterns:
        if re.search(pattern_regex, system_prompt, re.IGNORECASE):
            return True

    return False


def should_trigger_interrupt(context: Dict[str, Any], trigger_name: str) -> bool:
    """Check if context should trigger a specific interrupt.

    Args:
        context: Extracted context information
        trigger_name: Name of the interrupt trigger

    Returns:
        True if interrupt should be triggered
    """
    interrupt_patterns = get_interrupt_patterns()
    if trigger_name not in interrupt_patterns:
        return False

    pattern = interrupt_patterns[trigger_name]
    triggers = pattern.get("triggers", [])
    last_message = context.get("last_user_message", "").lower()

    return any(trigger in last_message for trigger in triggers)
