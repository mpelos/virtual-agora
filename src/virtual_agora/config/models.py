"""Configuration schema definitions for Virtual Agora.

This module defines Pydantic models for validating and parsing
the YAML configuration file.
"""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Provider(str, Enum):
    """Supported LLM providers."""

    GOOGLE = "Google"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GROK = "Grok"

    @classmethod
    def _missing_(cls, value: object) -> Optional["Provider"]:
        """Handle case-insensitive provider names."""
        if isinstance(value, str):
            # Try case-insensitive match
            value_lower = value.lower()
            for member in cls:
                if member.value.lower() == value_lower:
                    return member
        return None


class ModeratorConfig(BaseModel):
    """Configuration for the Moderator agent.

    The moderator is responsible for process facilitation,
    agenda synthesis, and relevance enforcement.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Provider = Field(
        ..., description="LLM provider for the moderator (e.g., Google, OpenAI)"
    )
    model: str = Field(
        ...,
        description="Model name/ID for the moderator (e.g., gemini-2.5-pro)",
        min_length=1,
    )


class SummarizerConfig(BaseModel):
    """Configuration for the Summarizer agent.

    The summarizer is responsible for compressing round discussions
    into compacted context for future rounds.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Provider = Field(
        ..., description="LLM provider for the summarizer (e.g., OpenAI, Google)"
    )
    model: str = Field(
        ...,
        description="Model name/ID for the summarizer (e.g., gpt-4o)",
        min_length=1,
    )
    # Optional fields for future extensibility
    temperature: Optional[float] = Field(
        default=0.3,  # Lower temperature for consistent summarization
        ge=0.0,
        le=1.0,
        description="Temperature for summarization",
    )
    max_tokens: Optional[int] = Field(
        default=500, gt=0, description="Maximum tokens for summaries"
    )


class ReportWriterConfig(BaseModel):
    """Configuration for the Report Writer agent.

    The report writer agent handles all long-form report generation
    through an iterative process, including both topic reports and
    comprehensive session reports.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Provider = Field(
        ...,
        description="LLM provider for the report writer agent (e.g., Anthropic, Google)",
    )
    model: str = Field(
        ...,
        description="Model name/ID for the report writer agent (e.g., claude-3-opus-20240229)",
        min_length=1,
    )
    # Optional fields for future extensibility
    temperature: Optional[float] = Field(
        default=0.6,  # Balanced temperature for comprehensive iterative synthesis
        ge=0.0,
        le=1.0,
        description="Temperature for report generation",
    )
    max_tokens: Optional[int] = Field(
        default=2000, gt=0, description="Maximum tokens per section iteration"
    )


class AgentConfig(BaseModel):
    """Configuration for discussing agents (debate participants).

    Each agent configuration can create one or more discussing agents
    of the same type (provider and model). These are the primary
    participants who propose agenda items, debate, and vote.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Provider = Field(..., description="LLM provider for the agent(s)")
    model: str = Field(
        ...,
        description="Model name/ID for the agent(s)",
        min_length=1,
    )
    count: int = Field(
        default=1,
        description="Number of agents to create with this configuration",
        ge=1,
        le=10,  # Reasonable limit to prevent accidental resource exhaustion
    )


class Config(BaseModel):
    """Root configuration model for Virtual Agora v1.3.

    This model represents the complete configuration loaded from
    the YAML file with support for specialized agents.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    moderator: ModeratorConfig = Field(
        ..., description="Configuration for the moderator agent"
    )
    summarizer: SummarizerConfig = Field(
        ..., description="Configuration for the summarizer agent"
    )
    report_writer: ReportWriterConfig = Field(
        ..., description="Configuration for the report writer agent"
    )
    agents: list[AgentConfig] = Field(
        ...,
        description="List of discussing agent configurations",
        min_length=1,  # At least one agent required
    )

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, v: list[AgentConfig]) -> list[AgentConfig]:
        """Validate the agent list."""
        # Count total number of agents
        total_agents = sum(agent.count for agent in v)

        if total_agents > 20:
            raise ValueError(
                f"Too many agents configured ({total_agents}). "
                "Maximum 20 agents allowed for performance reasons."
            )

        if total_agents < 2:
            raise ValueError(
                f"At least 2 agents required for a discussion, "
                f"but only {total_agents} configured."
            )

        return v

    def get_total_agent_count(self) -> int:
        """Get the total number of discussion agents."""
        return sum(agent.count for agent in self.agents)

    def get_agent_names(self) -> list[str]:
        """Generate list of agent names based on configuration."""
        names = []
        for agent_config in self.agents:
            base_name = f"{agent_config.model}"
            if agent_config.count == 1:
                names.append(f"{base_name}-1")
            else:
                for i in range(1, agent_config.count + 1):
                    names.append(f"{base_name}-{i}")
        return names
