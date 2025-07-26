"""Tool Integration Module for Virtual Agora.

This module provides base tool definitions for Virtual Agora operations,
implementing tools that agents can use during discussions.
"""

from typing import List, Dict, Any, Optional, Type
from datetime import datetime
import json
import uuid

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


# Pydantic models for tool inputs
class ProposalInput(BaseModel):
    """Input schema for proposal tool."""

    topics: List[str] = Field(
        description="List of 3-5 discussion topics to propose",
        min_length=3,
        max_length=5,
    )
    rationale: Optional[str] = Field(
        description="Brief rationale for the proposed topics", default=None
    )


class VotingInput(BaseModel):
    """Input schema for voting tool."""

    topic: str = Field(description="The topic being voted on")
    vote: str = Field(
        description="Vote value: 'yes', 'no', or 'abstain'",
        pattern="^(yes|no|abstain)$",
    )
    reasoning: Optional[str] = Field(description="Reasoning for the vote", default=None)


class SummaryInput(BaseModel):
    """Input schema for summary tool."""

    content: str = Field(description="Content to summarize")
    max_length: Optional[int] = Field(
        description="Maximum length of summary in words", default=150, ge=50, le=500
    )
    style: Optional[str] = Field(
        description="Summary style: 'concise', 'detailed', or 'bullet'",
        default="concise",
        pattern="^(concise|detailed|bullet)$",
    )


class ProposalTool(BaseTool):
    """Tool for proposing discussion topics."""

    name: str = "propose_topics"
    description: str = (
        "Use this tool to propose 3-5 discussion topics for the Virtual Agora session. "
        "Topics should be relevant to the main discussion theme."
    )
    args_schema: Type[BaseModel] = ProposalInput
    return_direct: bool = False

    def _run(
        self,
        topics: List[str],
        rationale: Optional[str] = None,
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Execute the proposal tool.

        Args:
            topics: List of proposed topics
            rationale: Optional rationale for proposals
            run_manager: LangChain run manager
            config: Runtime configuration

        Returns:
            Formatted proposal response
        """
        try:
            # Validate topic count
            if len(topics) < 3 or len(topics) > 5:
                return "Error: Please propose between 3 and 5 topics."

            # Create proposal structure
            proposal = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "topics": topics,
                "rationale": rationale,
            }

            # Format response
            response = "I propose the following discussion topics:\n"
            for i, topic in enumerate(topics, 1):
                response += f"{i}. {topic}\n"

            if rationale:
                response += f"\nRationale: {rationale}"

            logger.info(f"Created proposal with {len(topics)} topics")

            return response

        except Exception as e:
            logger.error(f"Error in proposal tool: {e}")
            return f"Error creating proposal: {str(e)}"

    async def _arun(
        self,
        topics: List[str],
        rationale: Optional[str] = None,
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Async version of proposal tool execution."""
        # For now, just call sync version
        return self._run(topics, rationale, run_manager, config)


class VotingTool(BaseTool):
    """Tool for voting on topics or decisions."""

    name: str = "vote"
    description: str = (
        "Use this tool to vote on a topic or decision. "
        "You can vote 'yes', 'no', or 'abstain' with optional reasoning."
    )
    args_schema: Type[BaseModel] = VotingInput
    return_direct: bool = False

    def _run(
        self,
        topic: str,
        vote: str,
        reasoning: Optional[str] = None,
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Execute the voting tool.

        Args:
            topic: Topic being voted on
            vote: Vote value (yes/no/abstain)
            reasoning: Optional reasoning
            run_manager: LangChain run manager
            config: Runtime configuration

        Returns:
            Formatted vote response
        """
        try:
            # Normalize vote
            vote = vote.lower().strip()
            if vote not in ["yes", "no", "abstain"]:
                return "Error: Vote must be 'yes', 'no', or 'abstain'."

            # Create vote record
            vote_record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "vote": vote,
                "reasoning": reasoning,
            }

            # Format response
            vote_text = {"yes": "Yes", "no": "No", "abstain": "Abstain"}[vote]

            response = f"I vote '{vote_text}' on the topic: {topic}"

            if reasoning:
                response += f"\n\nReasoning: {reasoning}"

            logger.info(f"Recorded vote '{vote}' on topic '{topic}'")

            return response

        except Exception as e:
            logger.error(f"Error in voting tool: {e}")
            return f"Error recording vote: {str(e)}"

    async def _arun(
        self,
        topic: str,
        vote: str,
        reasoning: Optional[str] = None,
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Async version of voting tool execution."""
        return self._run(topic, vote, reasoning, run_manager, config)


class SummaryTool(BaseTool):
    """Tool for generating summaries of discussions."""

    name: str = "summarize"
    description: str = (
        "Use this tool to generate a summary of discussion content. "
        "You can specify the style (concise, detailed, bullet) and maximum length."
    )
    args_schema: Type[BaseModel] = SummaryInput
    return_direct: bool = False

    def _run(
        self,
        content: str,
        max_length: Optional[int] = 150,
        style: Optional[str] = "concise",
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Execute the summary tool.

        Args:
            content: Content to summarize
            max_length: Maximum summary length in words
            style: Summary style
            run_manager: LangChain run manager
            config: Runtime configuration

        Returns:
            Generated summary
        """
        try:
            # Simple implementation - in real use, this would call an LLM
            # or use more sophisticated summarization

            words = content.split()

            if style == "bullet":
                # Extract key sentences for bullet points
                sentences = content.split(".")
                key_sentences = sentences[: min(5, len(sentences))]
                summary = "Summary:\n"
                for sentence in key_sentences:
                    if sentence.strip():
                        summary += f"• {sentence.strip()}\n"
            elif style == "detailed":
                # Keep more content for detailed summary
                word_limit = min(max_length, len(words))
                summary = " ".join(words[:word_limit])
                if len(words) > word_limit:
                    summary += "..."
            else:  # concise
                # Create concise summary
                word_limit = min(max_length // 2, len(words))
                summary = " ".join(words[:word_limit])
                if len(words) > word_limit:
                    summary += "..."

            logger.info(f"Generated {style} summary with {len(summary.split())} words")

            return summary

        except Exception as e:
            logger.error(f"Error in summary tool: {e}")
            return f"Error generating summary: {str(e)}"

    async def _arun(
        self,
        content: str,
        max_length: Optional[int] = 150,
        style: Optional[str] = "concise",
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Async version of summary tool execution."""
        return self._run(content, max_length, style, run_manager, config)


def create_discussion_tools() -> List[BaseTool]:
    """Create the standard set of discussion tools.

    Returns:
        List of tool instances for Virtual Agora discussions
    """
    return [
        ProposalTool(),
        VotingTool(),
        SummaryTool(),
    ]


# Alternative: Use @tool decorator for simpler tools
@tool
def count_votes(votes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count votes from a list of vote records.

    Args:
        votes: List of vote dictionaries with 'vote' field

    Returns:
        Dictionary with vote counts
    """
    counts = {"yes": 0, "no": 0, "abstain": 0}

    for vote_record in votes:
        vote = vote_record.get("vote", "").lower()
        if vote in counts:
            counts[vote] += 1

    return counts


@tool
def format_agenda(topics: List[str], votes: Optional[Dict[str, int]] = None) -> str:
    """Format a list of topics into an agenda.

    Args:
        topics: List of discussion topics
        votes: Optional vote counts per topic

    Returns:
        Formatted agenda string
    """
    agenda = "Discussion Agenda:\n\n"

    for i, topic in enumerate(topics, 1):
        agenda += f"{i}. {topic}"

        if votes and topic in votes:
            vote_info = votes[topic]
            agenda += f" (Votes: {vote_info})"

        agenda += "\n"

    return agenda
