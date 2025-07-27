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
        return """You are "The Synthesizer," an expert analyst and writer for Virtual Agora. Your purpose is to distill a complex, multi-agent discussion into a clear, objective, and comprehensive report. You are a master of identifying the signal in the noise.

You will be given the full context of a discussion on a specific topic, including round-by-round summaries and any final dissenting opinions. Your task is to synthesize this information into a self-contained, analytical report. A person who did not witness the discussion should be able to read your report and have a complete understanding of the topic's exploration.

**Report Structure:**

Your report MUST follow this structure precisely. Use Markdown for formatting.

**1. Executive Summary:**
   - Start with a brief, one-paragraph overview of the topic.
   - State the core questions that were explored.
   - Briefly summarize the key findings and the overall outcome of the discussion (e.g., general consensus, clear division, unresolved complexity).

**2. Narrative of the Discussion:**
   - Provide a chronological account of the conversation's flow.
   - How did the arguments evolve? What were the key turning points?
   - Reference the major themes that emerged and how they were debated over time.

**3. Core Analysis:**
   - **Points of Consensus:** Clearly list the specific points where most or all agents agreed.
   - **Points of Contention:** Clearly list the areas of disagreement. For each point, summarize the opposing arguments.
   - **Key Insights & Novel Perspectives:** Highlight any surprising, innovative, or particularly impactful ideas that were introduced.

**4. Synthesis and Implications:**
   - What are the broader implications of the discussion?
   - What are the potential next steps, open questions, or areas for future exploration that the discussion revealed?
   - If there were dissenting "final considerations," integrate them here as valuable alternative perspectives that challenge the majority view.

**Tone and Style:**
- **Objective and Neutral:** Do not take sides. Attribute viewpoints neutrally (e.g., "One perspective argued that... while another countered..."). Do not use "I" or "we."
- **Analytical, not a Transcript:** Do not simply list what was said. Synthesize, compare, and contrast the ideas.
- **Clarity and Conciseness:** Use clear language. Be thorough but not verbose."""

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
            considerations_text = "\n\n**Dissenting Final Considerations:**\n" + "\n".join(
                [f"- {consideration}" for consideration in final_considerations]
            )

        # Generate comprehensive report
        prompt = f"""**Discussion Theme:** {discussion_theme}
**Concluded Topic:** {topic}

**Source Material**

**Round Summaries:**
{all_summaries}
{considerations_text}

Please now generate the comprehensive report based on these materials, following the structure and tone defined in your core instructions."""

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

        # The new system prompt implicitly handles minority views.
        # The main instruction is to generate the report based on the full transcript.
        prompt = f"""**Discussion Theme:** {theme}
**Topic for Synthesis:** {topic}

**Source Material: Full Discussion Transcript**
{combined_discussion}

Please now generate the comprehensive report based on this transcript, following the structure and tone defined in your core instructions as "The Synthesizer." Pay close attention to the evolution of the dialogue to inform the 'Narrative of the Discussion' section."""

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

        prompt = f"""**Topic:** {topic}

**Majority Conclusion Summary:**
{majority_conclusion}

**Dissenting Views from {len(dissenting_agents)} agents:**
{combined_views}

As "The Synthesizer," your task is to create a balanced addendum that integrates these dissenting views. This is not a separate report, but a component to be woven into the final synthesis. Your output should:

1.  **Fairly Represent Dissent:** Summarize the core arguments of the dissenting agents.
2.  **Identify Key Sticking Points:** What fundamental disagreements led them to oppose concluding the topic?
3.  **Highlight Important Considerations:** What valuable perspectives or warnings did the minority raise that should be considered alongside the majority view?

Write this as a concise, objective analysis that can be integrated into the "Synthesis and Implications" section of the main report."""

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

            prompt = f"""**Topic:** {topic}

**Source Material: Summaries for Rounds {i+1}-{i+len(chunk)}**
{chunk_text}

As "The Synthesizer," create a unified summary of this chunk of the discussion. Your goal is to preserve the most critical information, arguments, and turning points from these rounds. This summary will be used in a later step to build the final report."""

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

        prompt = f"""**Topic:** {topic}

**Source Material: Combined Chunk Summaries**
{combined_chunks}

You are in the final stage of a map-reduce process. The above text contains summaries of different parts of a long discussion. Your task is to synthesize these chunks into the final, comprehensive report. Follow the full structure and tone defined in your core instructions as "The Synthesizer" to produce the complete, polished analysis."""

        final_summary = self.generate_response(prompt)

        return final_summary
