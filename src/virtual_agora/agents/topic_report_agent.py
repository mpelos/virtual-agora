"""Topic Report Agent implementation for Virtual Agora v1.3.

This module provides a specialized agent for synthesizing concluded agenda items,
creating comprehensive reports from round summaries and final considerations.
"""

from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


class TopicReportAgent(LLMAgent):
    """Specialized agent for topic report synthesis.

    This agent is invoked after an agenda item is concluded to synthesize
    all round summaries and final considerations into a comprehensive report.
    """

    PROMPT_VERSION = "1.3"

    def __init__(self, agent_id: str, llm: BaseChatModel, **kwargs):
        """Initialize the Topic Report agent.

        Args:
            agent_id: Unique identifier
            llm: Language model instance
        """
        # Use specialized prompt from v1.3 spec
        system_prompt = self._get_topic_report_prompt()

        super().__init__(
            agent_id=agent_id, llm=llm, system_prompt=system_prompt, **kwargs
        )

    def _get_topic_report_prompt(self) -> str:
        """Get the specialized topic report prompt from v1.3 spec."""
        return """You are a specialized synthesis tool for Virtual Agora's topic reporting. Your task is to analyze ALL compacted round summaries and final considerations for a concluded agenda item and create a comprehensive, standalone report. Structure your report to include:
1. Topic overview and key questions addressed
2. Major themes and arguments that emerged
3. Points of consensus among participants
4. Areas of disagreement or ongoing debate
5. Key insights and novel perspectives
6. Implications and potential next steps

Your report should be thorough enough that someone who didn't participate in the discussion can understand the full scope of the conversation. Write as an objective analyst, not a participant."""

    def synthesize_topic(
        self,
        round_summaries: List[str],
        final_considerations: List[str],
        topic: str,
        discussion_theme: str,
    ) -> str:
        """Create comprehensive report for concluded topic.

        Args:
            round_summaries: All round summaries for this topic
            final_considerations: Final thoughts from agents
            topic: The concluded topic
            discussion_theme: Overall session theme

        Returns:
            Comprehensive topic report
        """
        # Combine all summaries
        all_summaries = "\n\n".join(
            [
                f"Round {i+1} Summary:\n{summary}"
                for i, summary in enumerate(round_summaries)
            ]
        )

        # Include final considerations if any
        considerations_text = ""
        if final_considerations:
            considerations_text = "\n\nFinal Considerations:\n" + "\n".join(
                [f"- {consideration}" for consideration in final_considerations]
            )

        # Generate comprehensive report
        prompt = f"""Discussion Theme: {discussion_theme}
Concluded Topic: {topic}

Round Summaries:
{all_summaries}
{considerations_text}

Create a comprehensive report following this structure:
1. Topic overview and key questions addressed
2. Major themes and arguments that emerged
3. Points of consensus among participants
4. Areas of disagreement or ongoing debate
5. Key insights and novel perspectives
6. Implications and potential next steps

Write as an objective analyst, not a participant. Ensure the report is thorough and self-contained."""

        report = self.generate_response(prompt)

        logger.info(
            f"Generated topic report for '{topic}' with {len(round_summaries)} rounds"
        )

        return report

    def generate_topic_summary(
        self,
        messages: List[Dict[str, Any]],
        topic: str,
        theme: str,
        include_minority_views: bool = True,
    ) -> str:
        """Generate a comprehensive topic summary (alternative method).

        This method is migrated from ModeratorAgent's generate_topic_summary.

        Args:
            messages: All messages for the topic
            topic: The topic being summarized
            theme: Overall discussion theme
            include_minority_views: Whether to include dissenting views

        Returns:
            Topic summary
        """
        # Extract content from messages
        content_list = []
        for msg in messages:
            if msg.get("content"):
                agent_name = msg.get("name", "Unknown")
                content = msg.get("content")
                content_list.append(f"{agent_name}: {content}")

        combined_discussion = "\n\n".join(content_list)

        minority_instruction = ""
        if include_minority_views:
            minority_instruction = "\nPay special attention to minority viewpoints and dissenting opinions."

        prompt = f"""Discussion Theme: {theme}
Topic: {topic}

Full Discussion:
{combined_discussion}

Create a comprehensive summary of this topic discussion. Include:
1. Main arguments and perspectives presented
2. Points of agreement and consensus
3. Areas of disagreement or debate
4. Key insights and discoveries
5. Unresolved questions
6. Implications for future discussion
{minority_instruction}

Write objectively as an analyst reviewing the discussion."""

        summary = self.generate_response(prompt)

        logger.info(f"Generated comprehensive topic summary for '{topic}'")

        return summary

    def handle_minority_considerations(
        self,
        dissenting_agents: List[str],
        dissenting_views: List[Dict[str, str]],
        topic: str,
        majority_conclusion: str,
    ) -> str:
        """Synthesize minority viewpoints into the topic report.

        This method is migrated from ModeratorAgent's handle_minority_considerations.

        Args:
            dissenting_agents: List of agent IDs who dissented
            dissenting_views: Their final considerations
            topic: The topic being concluded
            majority_conclusion: The majority's conclusion

        Returns:
            Synthesis of minority views
        """
        # Format dissenting views
        formatted_views = []
        for view in dissenting_views:
            agent = view.get("agent", "Unknown")
            content = view.get("content", "")
            formatted_views.append(f"{agent}:\n{content}")

        combined_views = "\n\n".join(formatted_views)

        prompt = f"""Topic: {topic}

Majority Conclusion:
{majority_conclusion}

Dissenting Views from {len(dissenting_agents)} agents:
{combined_views}

Synthesize these minority viewpoints into a balanced addendum that:
1. Fairly represents the dissenting perspectives
2. Explains why these agents disagreed with concluding the topic
3. Highlights important considerations raised by the minority
4. Suggests how these views might inform future discussion

Write objectively without taking sides."""

        minority_synthesis = self.generate_response(prompt)

        logger.info(
            f"Synthesized {len(dissenting_agents)} minority views for topic '{topic}'"
        )

        return minority_synthesis

    def generate_topic_summary_map_reduce(
        self, round_summaries: List[str], topic: str, chunk_size: int = 3
    ) -> str:
        """Generate topic summary using map-reduce approach for long discussions.

        This method is migrated from ModeratorAgent's _generate_topic_summary_map_reduce.

        Args:
            round_summaries: List of round summaries
            topic: The topic being summarized
            chunk_size: Number of summaries to process at once

        Returns:
            Combined topic summary
        """
        if len(round_summaries) <= chunk_size:
            # Small enough to process directly
            return self.synthesize_topic(round_summaries, [], topic, "")

        # Map phase: Process chunks
        chunk_summaries = []
        for i in range(0, len(round_summaries), chunk_size):
            chunk = round_summaries[i : i + chunk_size]

            chunk_text = "\n\n".join(
                [
                    f"Round {i+j+1} Summary:\n{summary}"
                    for j, summary in enumerate(chunk)
                ]
            )

            prompt = f"""Topic: {topic}

Summaries to combine:
{chunk_text}

Create a unified summary of these {len(chunk)} rounds, preserving key points and insights."""

            chunk_summary = self.generate_response(prompt)
            chunk_summaries.append(chunk_summary)

        # Reduce phase: Combine chunk summaries
        logger.info(
            f"Map-reduce: Processing {len(chunk_summaries)} chunks for topic '{topic}'"
        )

        # Recursively reduce if needed
        if len(chunk_summaries) > chunk_size:
            return self.generate_topic_summary_map_reduce(
                chunk_summaries, topic, chunk_size
            )

        # Final reduction
        combined_chunks = "\n\n".join(
            [
                f"Chunk {i+1} Summary:\n{summary}"
                for i, summary in enumerate(chunk_summaries)
            ]
        )

        prompt = f"""Topic: {topic}

Combined chunk summaries:
{combined_chunks}

Create a final, comprehensive summary that integrates all the key points from the entire discussion."""

        final_summary = self.generate_response(prompt)

        return final_summary
