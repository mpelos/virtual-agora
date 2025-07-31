"""Summarizer Agent implementation for Virtual Agora v1.3.

This module provides a specialized agent for text compression and summarization,
responsible for creating concise, agent-agnostic summaries of discussion rounds.
"""

from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger


# Define message utility function to avoid circular imports
def get_message_attribute(msg, attr_name, default=None):
    """Get attribute from message in a standardized way.
    Args:
        msg: Message object (BaseMessage or dict)
        attr_name: Name of attribute to get
        default: Default value if attribute not found
    Returns:
        Attribute value or default
    """
    if hasattr(msg, "content"):  # LangChain BaseMessage
        return getattr(msg, "additional_kwargs", {}).get(attr_name, default)
    else:  # Virtual Agora dict format
        return msg.get(attr_name, default)


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
        return (
            "**Identity:** You are the Summarizer, a specialized text compression and synthesis tool for Virtual Agora.\n"
            "**Core Directive:** You have two main tasks:\n\n"
            "**Task 1 - Round Summarization:** Read the raw text of a discussion round and produce a concise, neutral, and agent-agnostic summary. Your output is critical for maintaining context in future rounds.\n\n"
            "**Task 2 - Topic Conclusion Summary:** After a topic discussion concludes, create a single paragraph summary that captures: 1. The key resolution or consensus reached. 2. Main points of agreement among agents. 3. Any outstanding questions or areas of disagreement. 4. The practical implications or next steps identified. This summary will be provided to agents when discussing future topics so they understand what has been previously resolved.\n\n"
            "**Key Synthesis Areas:**\n"
            "- **Main Arguments:** What were the primary claims or arguments presented?\n"
            "- **Points of Consensus:** Where did the participants seem to agree?\n"
            "- **Points of Contention:** What were the key disagreements or debates?\n"
            "- **New Information & Insights:** Were any new perspectives, evidence, or questions introduced that shifted the conversation?\n\n"
            "**Strict Constraints:**\n"
            "- **Agent-Agnostic:** NEVER attribute points to specific agents (e.g., 'Agent A said...'). Summarize the ideas themselves.\n"
            "- **Neutrality:** Do not inject your own opinions, analysis, or interpretations. Your job is to reflect the content of the discussion, not to comment on it.\n"
            "- **Third-Person Perspective:** Write the summary in a detached, third-person narrative style (e.g., 'The discussion covered...', 'One viewpoint suggested...').\n"
            "- **Conciseness:** Preserve the essential information while being substantially more concise than the original text.\n\n"
            "Write in third person, avoid agent names, and maintain objectivity in both tasks."
        )

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
        # Handle both BaseMessage objects and dict formats
        content_list = []
        for msg in messages:
            try:
                # Get content using appropriate method based on message type
                if hasattr(msg, "content"):  # BaseMessage object
                    content = msg.content if msg.content is not None else ""
                    speaker_id = get_message_attribute(msg, "speaker_id", "Unknown")
                else:  # Dict format
                    content = msg.get("content", "") if msg else ""
                    speaker_id = msg.get("speaker_id", "Unknown") if msg else "Unknown"

                # Only include non-empty content
                if content and str(content).strip():
                    content_list.append(f"- {speaker_id}: {content}")

            except Exception as e:
                logger.warning(f"Failed to extract content from message: {e}")
                # Continue processing other messages
                continue

        # Handle empty messages case
        if not content_list:
            return "No new discussion content was provided in this round."

        combined_text = "\n".join(content_list)

        # Calculate target length
        original_tokens = self._estimate_tokens(combined_text)
        target_tokens = int(original_tokens * self.compression_ratio)
        target_tokens = min(target_tokens, self.max_summary_tokens)

        # Create summarization prompt
        prompt = (
            f"**Task:** Summarize Discussion Round.\n"
            f"**Topic:** {topic}\n"
            f"**Round:** {round_number}\n\n"
            f"**Raw Discussion Transcript:**\n---\n{combined_text}\n---\n\n"
            f"**Instructions:**\n"
            f"1.  Create a concise, neutral, and agent-agnostic summary of the key points from the transcript.\n"
            f"2.  Synthesize the main arguments, points of consensus, disagreements, and any new insights.\n"
            f"3.  Write in the third person. Do not attribute comments to specific speakers.\n"
            f"4.  The target length is approximately {target_tokens} tokens.\n"
            f"5.  Your response should contain **only the summary text** and nothing else."
        )

        # Generate summary
        summary = self.generate_response(prompt)

        # Log compression metrics
        summary_tokens = self._estimate_tokens(summary)
        if original_tokens > 0:
            logger.info(
                f"Round {round_number} compressed: "
                f"{original_tokens} -> {summary_tokens} tokens "
                f"({summary_tokens/original_tokens:.1%} of original)"
            )
        else:
            logger.info(
                f"Round {round_number} summary generated with {summary_tokens} tokens."
            )

        return summary

    def summarize_topic_conclusion(
        self, round_summaries: List[str], final_considerations: List[str], topic: str
    ) -> str:
        """Create a one-paragraph summary of a concluded topic discussion.

        This method creates a concise conclusion summary that captures the key
        resolution, consensus points, outstanding questions, and practical
        implications from a completed topic discussion.

        Args:
            round_summaries: List of round summaries from the topic discussion
            final_considerations: List of final thoughts from agents
            topic: The topic that was discussed

        Returns:
            One-paragraph summary of the topic conclusion
        """
        # Combine round summaries
        combined_summaries = "\n\n".join(
            [f"Round {i+1}: {summary}" for i, summary in enumerate(round_summaries)]
        )

        # Combine final considerations
        combined_considerations = "\n".join(
            [f"- {consideration}" for consideration in final_considerations]
        )

        # Create topic conclusion prompt
        prompt = (
            f"**Task:** Create Topic Conclusion Summary.\n"
            f"**Topic:** {topic}\n\n"
            f"**Round Summaries:**\n---\n{combined_summaries}\n---\n\n"
            f"**Final Considerations:**\n---\n{combined_considerations}\n---\n\n"
            f"**Instructions:**\n"
            f"1. Create a single paragraph summary of this topic's conclusion.\n"
            f"2. Capture: key resolution/consensus, main agreement points, outstanding questions, practical implications.\n"
            f"3. This summary will help agents understand what was resolved when discussing future topics.\n"
            f"4. Write in third person, be objective, and avoid agent attributions.\n"
            f"5. Keep it concise but comprehensive (aim for 3-5 sentences).\n"
            f"6. Your response should contain **only the summary paragraph** and nothing else."
        )

        # Generate topic conclusion summary
        topic_summary = self.generate_response(prompt)

        # Log the generation
        summary_tokens = self._estimate_tokens(topic_summary)
        logger.info(
            f"Generated topic conclusion summary for '{topic}': {summary_tokens} tokens"
        )

        return topic_summary

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
            [
                f"**Round {i+1} Summary:**\n{summary}"
                for i, summary in enumerate(summaries)
            ]
        )

        prompt = (
            f"**Task:** Create a Unified Progressive Summary.\n"
            f"**Topic:** {topic}\n\n"
            f"**Input:** A series of summaries from consecutive discussion rounds.\n---\n{combined_summaries}\n---\n\n"
            f"**Instructions:**\n"
            f"1.  Synthesize the provided round summaries into a single, coherent narrative.\n"
            f"2.  Trace the evolution of the discussion, highlighting how arguments developed, shifted, or were resolved.\n"
            f"3.  Identify the major themes, key points of consensus/disagreement, and the most critical unanswered questions.\n"
            f"4.  The final summary should not exceed {max_tokens} tokens.\n"
            f"5.  Your response should contain **only the unified summary text** and nothing else."
        )

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
        # Handle both BaseMessage objects and dict formats
        content_list = []
        for msg in messages:
            try:
                # Get content using appropriate method based on message type
                if hasattr(msg, "content"):  # BaseMessage object
                    content = msg.content if msg.content is not None else ""
                else:  # Dict format
                    content = msg.get("content", "") if msg else ""

                # Only include non-empty content
                if content and str(content).strip():
                    content_list.append(content)

            except Exception as e:
                logger.warning(
                    f"Failed to extract content from message in key insights: {e}"
                )
                # Continue processing other messages
                continue
        combined_text = "\n\n".join(content_list)

        previous_context = ""
        if previous_insights:
            previous_context = (
                "\n**Previously Identified Insights (for context):**\n"
                + "\n".join([f"- {insight}" for insight in previous_insights])
            )

        prompt = (
            f"**Task:** Extract Key Insights.\n"
            f"**Topic:** {topic}\n\n"
            f"**Raw Discussion Transcript:**\n---\n{combined_text}\n---\n{previous_context}\n\n"
            f"**Instructions:**\n"
            f"1.  Analyze the discussion and identify the 3-5 most significant and novel insights.\n"
            f"2.  Focus on revelatory conclusions, critical unanswered questions, or major shifts in the group's understanding. Do not simply summarize the discussion.\n"
            f"3.  Each insight must be a concise, standalone statement.\n"
            f"4.  Format your response as a numbered list. Your response should contain **only the numbered list** and nothing else."
        )

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
