"""Fake LLM implementations for integration testing.

This module provides various fake LLM implementations that can simulate
realistic agent responses without making external API calls.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Union, Iterator
from unittest.mock import Mock

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult, ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.runnables import RunnableConfig
from pydantic import Field, PrivateAttr


class FakeLLMBase(BaseChatModel):
    """Base class for fake LLM implementations."""

    model_name: str = Field(default="fake-model", alias="model")
    temperature: float = Field(default=0.0)

    # Use private attributes for mutable state
    _call_count: int = PrivateAttr(default=0)
    _last_messages: List[BaseMessage] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "fake"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat result."""
        # Update private attributes
        self._call_count += 1
        self._last_messages = messages

        response_text = self._get_response(messages, **kwargs)
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _get_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """Override this method in subclasses to provide specific responses."""
        return "Default fake response"

    @property
    def call_count(self) -> int:
        """Get the call count."""
        return self._call_count

    @property
    def last_messages(self) -> List[BaseMessage]:
        """Get the last messages."""
        return self._last_messages


class PredictableFakeLLM(FakeLLMBase):
    """Fake LLM that returns predefined responses based on message patterns."""

    response_patterns: Dict[str, str] = Field(default_factory=dict)
    default_response: str = Field(
        default="I don't have a specific response for this input."
    )

    def _get_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """Return response based on pattern matching."""
        # Get the last human message content
        last_message = ""
        for msg in reversed(messages):
            if isinstance(msg, (HumanMessage, SystemMessage)):
                last_message = msg.content.lower()
                break

        # Match against patterns
        for pattern, response in self.response_patterns.items():
            if re.search(pattern.lower(), last_message):
                return response

        return self.default_response


class ModeratorFakeLLM(PredictableFakeLLM):
    """Fake LLM specifically designed for moderator responses."""

    def __init__(self, **kwargs):
        # Initialize with moderator-specific patterns
        moderator_patterns = {
            r"propose.*topics?": "Please propose 3-5 sub-topics for discussion based on the main topic. \nConsider different angles, implications, and areas of focus that would benefit \nfrom diverse perspectives.",
            r"vote.*agenda": "Please review the following proposed topics and vote on your \npreferred order of discussion. Rank them from most important to least important \nand provide brief justification for your preferences.",
            r"synthesize.*agenda": '{"proposed_agenda": ["Technical Implementation", "Legal and Regulatory Considerations", "Social Impact and Adoption", "Economic Implications"]}',
            r"summarize.*round": "**Round Summary:**\nThe agents discussed technical implementation details, focusing on scalability \nand security concerns. Key points included infrastructure requirements, \npotential challenges, and proposed solutions.",
            r"conclude.*topic": "Should we conclude the discussion on this topic? \nPlease respond with 'Yes' or 'No' and provide a brief justification \nfor your decision.",
            r"final.*considerations": "As someone who voted to continue the discussion, please share \nyour final considerations on this topic before we move forward.",
            r"topic.*summary": "# Topic Summary: Technical Implementation\n\n## Overview\nThe discussion covered various aspects of technical implementation, including \narchitecture decisions, scalability considerations, and security measures.\n\n## Key Points Discussed\n- Infrastructure requirements and scalability\n- Security protocols and best practices  \n- Integration challenges and solutions\n- Performance optimization strategies\n\n## Conclusions\nThe agents reached consensus on core technical requirements while acknowledging \nareas that need further investigation.",
            r"report.*structure": '["Executive Summary", "Technical Analysis", "Legal and Regulatory Framework", "Social Impact Assessment", "Economic Implications", "Recommendations and Next Steps"]',
            r"section.*content": "# Executive Summary\n\nThis report synthesizes the comprehensive discussion on the proposed topic, \nexamining technical, legal, social, and economic dimensions through the \nperspectives of multiple AI agents representing diverse viewpoints.\n\n## Key Findings\n- Technical feasibility confirmed with specific implementation requirements\n- Legal framework requires careful consideration of regulatory compliance\n- Social impact shows both opportunities and challenges for adoption\n- Economic implications suggest significant potential with measured risks",
        }

        # Call parent with patterns
        super().__init__(response_patterns=moderator_patterns, **kwargs)


class AgentFakeLLM(PredictableFakeLLM):
    """Fake LLM for discussion agents with configurable personality."""

    agent_personality: str = Field(default="balanced")
    agent_id: str = Field(default="agent")

    def __init__(self, **kwargs):
        # Extract personality and agent_id if provided
        personality = kwargs.pop("agent_personality", "balanced")
        agent_id = kwargs.pop("agent_id", "agent")

        # Generate patterns based on personality
        agent_patterns = self._create_agent_patterns(personality, agent_id)

        # Initialize with patterns
        super().__init__(
            response_patterns=agent_patterns,
            agent_personality=personality,
            agent_id=agent_id,
            **kwargs,
        )

    def _create_agent_patterns(self, personality: str, agent_id: str) -> Dict[str, str]:
        """Create response patterns based on personality."""
        return {
            r"propose.*topics?": self._generate_topic_proposal(personality),
            r"vote.*agenda": self._generate_agenda_vote(personality),
            r"discuss.*topic": self._generate_discussion_response(
                personality, agent_id
            ),
            r"conclude.*topic": self._generate_conclusion_vote(),
            r"final.*considerations": self._generate_final_considerations(personality),
            r"modify.*agenda": self._generate_agenda_modification(),
        }

    def _generate_topic_proposal(self, personality: str) -> str:
        proposals = {
            "optimistic": [
                "Future opportunities and potential",
                "Innovation drivers and catalysts",
                "Positive societal impact",
                "Success stories and best practices",
            ],
            "skeptical": [
                "Potential risks and challenges",
                "Implementation barriers",
                "Unintended consequences",
                "Resource constraints and limitations",
            ],
            "technical": [
                "Technical architecture and design",
                "Performance and scalability",
                "Security and reliability",
                "Integration and compatibility",
            ],
            "balanced": [
                "Core requirements and objectives",
                "Implementation approach",
                "Risk assessment and mitigation",
                "Success metrics and evaluation",
            ],
        }

        personality_proposals = proposals.get(personality, proposals["balanced"])
        return f"I propose the following topics: {', '.join(personality_proposals[:3])}"

    def _generate_agenda_vote(self, personality: str) -> str:
        vote_styles = {
            "optimistic": "I prioritize topics that focus on positive outcomes and opportunities.",
            "skeptical": "I believe we should address potential risks and challenges first.",
            "technical": "I vote for technical topics to be prioritized for thorough analysis.",
            "balanced": "I support a balanced approach covering all key aspects systematically.",
        }

        style = vote_styles.get(personality, vote_styles["balanced"])
        return f"{style} My ranking prioritizes practical implementation concerns."

    def _generate_discussion_response(self, personality: str, agent_id: str) -> str:
        responses = {
            "optimistic": "This presents exciting opportunities for innovation and positive change. "
            "I see significant potential for beneficial outcomes if implemented thoughtfully.",
            "skeptical": "While there may be benefits, we must carefully consider the potential risks "
            "and challenges. I'm concerned about unintended consequences.",
            "technical": "From a technical perspective, we need to address architecture, scalability, "
            "and security requirements. The implementation details are crucial.",
            "balanced": "This requires careful analysis of both opportunities and challenges. "
            "A measured approach considering all factors would be most prudent.",
        }

        base_response = responses.get(personality, responses["balanced"])
        return f"{base_response} ({agent_id})"

    def _generate_conclusion_vote(self) -> str:
        # Vary votes to create realistic scenarios
        votes = ["Yes", "No"]
        vote = random.choice(votes)

        justifications = {
            "Yes": [
                "We've covered the key points comprehensively.",
                "The main issues have been adequately discussed.",
                "I believe we have sufficient information to proceed.",
            ],
            "No": [
                "There are still important aspects to explore.",
                "We need more detailed discussion on implementation.",
                "Some critical questions remain unanswered.",
            ],
        }

        justification = random.choice(justifications[vote])
        return f"{vote}. {justification}"

    def _generate_final_considerations(self, personality: str) -> str:
        return (
            f"As a final consideration, I want to emphasize the importance of {personality} "
            f"perspective in this discussion. We should ensure all viewpoints are properly addressed."
        )

    def _generate_agenda_modification(self) -> str:
        return (
            "Based on our discussion, I suggest we consider adding a topic on "
            "implementation timeline and resource allocation."
        )


class RandomFakeLLM(FakeLLMBase):
    """Fake LLM that returns random responses from a pool."""

    response_pool: List[str] = Field(
        default_factory=lambda: [
            "This is an interesting point to consider.",
            "I have a different perspective on this matter.",
            "Let me elaborate on the technical aspects.",
            "From a practical standpoint, we should examine...",
            "The implications of this approach include...",
        ]
    )

    def _get_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """Return a random response from the pool."""
        return random.choice(self.response_pool)


class ErrorFakeLLM(FakeLLMBase):
    """Fake LLM that simulates various error conditions."""

    error_rate: float = Field(default=0.3)
    error_types: List[str] = Field(
        default_factory=lambda: ["timeout", "api_error", "invalid_response"]
    )

    def _get_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """Sometimes return errors, sometimes normal responses."""
        if random.random() < self.error_rate:
            error_type = random.choice(self.error_types)

            if error_type == "timeout":
                raise TimeoutError("Simulated LLM timeout")
            elif error_type == "api_error":
                raise Exception("Simulated API error")
            elif error_type == "invalid_response":
                return "INVALID_JSON_RESPONSE_THAT_CANNOT_BE_PARSED"

        return "Normal response after potential error simulation."


class ScenarioFakeLLM(FakeLLMBase):
    """Fake LLM that implements specific test scenarios."""

    scenario: str = Field(default="default")
    _scenario_state: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def _get_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """Return response based on the configured scenario."""
        if self.scenario == "quick_consensus":
            return self._quick_consensus_response(messages)
        elif self.scenario == "extended_debate":
            return self._extended_debate_response(messages)
        elif self.scenario == "minority_dissent":
            return self._minority_dissent_response(messages)
        else:
            return "Default scenario response."

    def _quick_consensus_response(self, messages: List[BaseMessage]) -> str:
        """Scenario where agents quickly reach consensus."""
        if "conclude" in messages[-1].content.lower():
            return "Yes. We've reached clear consensus on this topic."
        return "I agree with the previous points and support this direction."

    def _extended_debate_response(self, messages: List[BaseMessage]) -> str:
        """Scenario where discussion continues for many rounds."""
        if "conclude" in messages[-1].content.lower():
            if self.call_count < 5:
                return "No. There are still important aspects to discuss in detail."
            else:
                return "Yes. After thorough discussion, I'm ready to conclude."
        return f"I'd like to examine this from another angle (round {self.call_count})."

    def _minority_dissent_response(self, messages: List[BaseMessage]) -> str:
        """Scenario where one agent consistently dissents."""
        if "conclude" in messages[-1].content.lower():
            # Create minority dissent by having this agent vote No occasionally
            if self.call_count % 3 == 0:
                return "No. I believe we need more comprehensive analysis."
            else:
                return "Yes. I support concluding this topic."
        return "I have concerns about this approach that merit discussion."


def create_fake_llm_pool(
    num_agents: int = 3, personalities: List[str] = None, scenario: str = "default"
) -> Dict[str, FakeLLMBase]:
    """Create a pool of fake LLMs for testing.

    Args:
        num_agents: Number of discussion agents to create
        personalities: List of personality types for agents
        scenario: Overall scenario type for the test

    Returns:
        Dictionary mapping agent IDs to fake LLM instances
    """
    if personalities is None:
        personalities = ["balanced", "optimistic", "skeptical", "technical"]

    llm_pool = {}

    # Create moderator LLM
    llm_pool["moderator"] = ModeratorFakeLLM()

    # Create agent LLMs
    for i in range(num_agents):
        agent_id = f"agent_{i+1}"
        personality = personalities[i % len(personalities)]

        if scenario == "error_prone":
            llm_pool[agent_id] = ErrorFakeLLM(error_rate=0.2)
        else:
            llm_pool[agent_id] = AgentFakeLLM(
                agent_personality=personality, agent_id=agent_id
            )

    return llm_pool
