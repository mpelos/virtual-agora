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
    
    The moderator is responsible for facilitating the discussion,
    synthesizing votes, and generating summaries.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    provider: Provider = Field(
        ...,
        description="LLM provider for the moderator (e.g., Google, OpenAI)"
    )
    model: str = Field(
        ...,
        description="Model name/ID for the moderator (e.g., gemini-1.5-pro)",
        min_length=1,
    )
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:
        """Validate model name based on provider."""
        provider = info.data.get("provider")
        
        if provider == Provider.GOOGLE:
            valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
            if not any(v.startswith(prefix) for prefix in valid_models):
                raise ValueError(
                    f"Invalid Google model: {v}. "
                    f"Expected one of: {', '.join(valid_models)}"
                )
        elif provider == Provider.OPENAI:
            valid_models = ["gpt-4", "gpt-3.5"]
            if not any(v.startswith(prefix) for prefix in valid_models):
                raise ValueError(
                    f"Invalid OpenAI model: {v}. "
                    f"Expected model starting with: {', '.join(valid_models)}"
                )
        elif provider == Provider.ANTHROPIC:
            valid_models = ["claude-3", "claude-2"]
            if not any(v.startswith(prefix) for prefix in valid_models):
                raise ValueError(
                    f"Invalid Anthropic model: {v}. "
                    f"Expected model starting with: {', '.join(valid_models)}"
                )
        # For Grok, we don't validate since model names are not yet known
        
        return v


class AgentConfig(BaseModel):
    """Configuration for a discussion agent or group of agents.
    
    Each agent configuration can create one or more agents of the
    same type (provider and model).
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    provider: Provider = Field(
        ...,
        description="LLM provider for the agent(s)"
    )
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
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info) -> str:
        """Validate model name based on provider."""
        # Use the same validation logic as ModeratorConfig
        provider = info.data.get("provider")
        
        if provider == Provider.GOOGLE:
            valid_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
            if not any(v.startswith(prefix) for prefix in valid_models):
                raise ValueError(
                    f"Invalid Google model: {v}. "
                    f"Expected one of: {', '.join(valid_models)}"
                )
        elif provider == Provider.OPENAI:
            valid_models = ["gpt-4", "gpt-3.5"]
            if not any(v.startswith(prefix) for prefix in valid_models):
                raise ValueError(
                    f"Invalid OpenAI model: {v}. "
                    f"Expected model starting with: {', '.join(valid_models)}"
                )
        elif provider == Provider.ANTHROPIC:
            valid_models = ["claude-3", "claude-2"]
            if not any(v.startswith(prefix) for prefix in valid_models):
                raise ValueError(
                    f"Invalid Anthropic model: {v}. "
                    f"Expected model starting with: {', '.join(valid_models)}"
                )
        
        return v


class Config(BaseModel):
    """Root configuration model for Virtual Agora.
    
    This model represents the complete configuration loaded from
    the YAML file.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    moderator: ModeratorConfig = Field(
        ...,
        description="Configuration for the moderator agent"
    )
    agents: list[AgentConfig] = Field(
        ...,
        description="List of agent configurations",
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