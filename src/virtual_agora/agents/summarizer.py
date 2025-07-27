"""Summarizer Agent implementation for Virtual Agora v1.3.

This module provides a specialized agent for text compression and summarization,
responsible for creating concise, agent-agnostic summaries of discussion rounds.
"""

from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

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
        **kwargs,
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
            agent_id=agent_id, llm=llm, system_prompt=system_prompt, **kwargs
        )

        self.compression_ratio = compression_ratio
        self.max_summary_tokens = max_summary_tokens

    def _get_summarizer_prompt(self) -> str:
        """Get the specialized summarizer prompt from v1.3 spec."""
        return """You are a specialized text compression tool for Virtual Agora. Your task is to read all agent comments from a single discussion round and create a concise, agent-agnostic summary that captures the key points, arguments, and insights without attribution to specific agents.

Focus on:
1. Main arguments presented
2. Points of agreement and disagreement
3. New insights or perspectives introduced
4. Questions raised or areas requiring further exploration

Your summary will be used as context for future rounds, so ensure it preserves essential information while being substantially more concise than the original. Write in third person, avoid agent names, and maintain objectivity."""

    def summarize_round(
        self, messages: List[Dict[str, Any]], topic: str, round_number: int
    ) -> str:
        """Create a compressed summary of a discussion round.

        Args:
            messages: List of agent messages from the round
            topic: Current discussion topic
            round_number: Current round number

        Returns:
            Compressed summary text
        """
        # Extract message content - filter out empty/whitespace-only content
        content_list = [
            msg.get("content", "")
            for msg in messages
            if msg.get("content") and msg.get("content").strip()
        ]

        # Handle empty messages case
        if not content_list:
            return "No discussion content to summarize in this round."

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

Create a summary of approximately {target_tokens} tokens that captures the essential points while maintaining objectivity. Focus on key arguments, agreements, disagreements, and new insights."""

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

    def generate_progressive_summary(
        self, summaries: List[str], topic: str, max_tokens: int = 1000
    ) -> str:
        """Generate a progressive summary of multiple round summaries.

        This method is migrated from ModeratorAgent's generate_progressive_summary.

        Args:
            summaries: List of round summaries to compress further
            topic: Current discussion topic
            max_tokens: Maximum tokens for the progressive summary

        Returns:
            Compressed progressive summary
        """
        # Combine all summaries
        combined_summaries = "\n\n".join(
            [f"Round {i+1} Summary:\n{summary}" for i, summary in enumerate(summaries)]
        )

        prompt = f"""Topic: {topic}

Previous Round Summaries:
{combined_summaries}

Create a unified progressive summary that captures the evolution of the discussion across all rounds. Maximum {max_tokens} tokens. Focus on:
1. Major themes that have emerged
2. How the discussion has evolved
3. Key points of consensus and disagreement
4. Important questions or areas for further exploration"""

        progressive_summary = self.generate_response(prompt)

        logger.info(f"Generated progressive summary for {len(summaries)} rounds")

        return progressive_summary

    def extract_key_insights(
        self,
        messages: List[Dict[str, Any]],
        topic: str,
        previous_insights: Optional[List[str]] = None,
    ) -> List[str]:
        """Extract key insights from discussion messages.

        This method is migrated from ModeratorAgent's extract_key_insights.

        Args:
            messages: List of agent messages
            topic: Current discussion topic
            previous_insights: Previous insights to build upon

        Returns:
            List of key insights
        """
        # Extract message content - filter out empty/whitespace-only content
        content_list = [
            msg.get("content", "")
            for msg in messages
            if msg.get("content") and msg.get("content").strip()
        ]
        combined_text = "\n\n".join(content_list)

        previous_context = ""
        if previous_insights:
            previous_context = f"\nPrevious Insights:\n" + "\n".join(
                [f"- {insight}" for insight in previous_insights]
            )

        prompt = f"""Topic: {topic}

Discussion Content:
{combined_text}
{previous_context}

Extract 3-5 key insights from this discussion. Focus on:
1. Novel perspectives or arguments
2. Important revelations or conclusions
3. Critical questions that have emerged
4. Significant shifts in understanding

Format each insight as a concise, standalone statement."""

        response = self.generate_response(prompt)

        # Parse insights from response
        insights = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (
                line.startswith("-") or line.startswith("•") or line[0].isdigit()
            ):
                # Remove bullet points or numbers
                insight = line.lstrip("-•0123456789. ").strip()
                if insight:
                    insights.append(insight)

        logger.info(f"Extracted {len(insights)} key insights from discussion")

        return insights

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters
        # This is a simplified approach; in production, use tiktoken or similar
        return len(text) // 4
