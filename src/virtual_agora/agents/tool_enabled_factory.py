"""Factory for creating tool-enabled agents in Virtual Agora.

This module provides factory functions for creating agents with
tool capabilities based on their role and the discussion phase.
"""

from typing import List, Optional, Dict, Any, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from virtual_agora.agents.llm_agent import LLMAgent
from virtual_agora.tools import (
    ProposalTool,
    VotingTool,
    SummaryTool,
    create_discussion_tools,
)
from virtual_agora.utils.logging import get_logger


logger = get_logger(__name__)


def create_tool_enabled_moderator(
    agent_id: str,
    llm: BaseChatModel,
    system_prompt: Optional[str] = None,
    enable_error_handling: bool = True,
    max_retries: int = 3,
    fallback_llm: Optional[BaseChatModel] = None
) -> LLMAgent:
    """Create a moderator agent with appropriate tools.
    
    The moderator gets tools for:
    - Summarizing discussions
    - Formatting agendas
    - Managing votes
    
    Args:
        agent_id: Unique identifier for the agent
        llm: LangChain chat model instance
        system_prompt: Optional custom system prompt
        enable_error_handling: Whether to enable error handling
        max_retries: Maximum retry attempts
        fallback_llm: Optional fallback LLM
        
    Returns:
        Tool-enabled moderator agent
    """
    # Create moderator-specific tools
    tools = [
        SummaryTool(),  # For phase summaries
        VotingTool(),   # For managing votes
    ]
    
    # Use the create_with_tools factory method
    agent = LLMAgent.create_with_tools(
        agent_id=agent_id,
        llm=llm,
        tools=tools,
        role="moderator",
        system_prompt=system_prompt,
        enable_error_handling=enable_error_handling,
        max_retries=max_retries,
        fallback_llm=fallback_llm
    )
    
    logger.info(f"Created tool-enabled moderator agent '{agent_id}' with {len(tools)} tools")
    return agent


def create_tool_enabled_participant(
    agent_id: str,
    llm: BaseChatModel,
    phase: int,
    system_prompt: Optional[str] = None,
    enable_error_handling: bool = True,
    max_retries: int = 3,
    fallback_llm: Optional[BaseChatModel] = None,
    custom_tools: Optional[Sequence[BaseTool]] = None
) -> LLMAgent:
    """Create a participant agent with phase-appropriate tools.
    
    Tools are selected based on the discussion phase:
    - Phase 1 (Agenda Setting): ProposalTool
    - Phase 2 (Discussion): SummaryTool
    - Phase 3 (Consensus): VotingTool
    
    Args:
        agent_id: Unique identifier for the agent
        llm: LangChain chat model instance
        phase: Current discussion phase
        system_prompt: Optional custom system prompt
        enable_error_handling: Whether to enable error handling
        max_retries: Maximum retry attempts
        fallback_llm: Optional fallback LLM
        custom_tools: Optional custom tools to use instead of defaults
        
    Returns:
        Tool-enabled participant agent
    """
    if custom_tools is not None:
        tools = list(custom_tools)
    else:
        # Select tools based on phase
        tools = []
        
        if phase == 1:  # Agenda Setting
            tools.append(ProposalTool())
        elif phase == 2:  # Discussion
            tools.append(SummaryTool())
        elif phase == 3:  # Consensus
            tools.append(VotingTool())
        
        # Participants can always use summary tool
        if phase != 2 and not any(isinstance(t, SummaryTool) for t in tools):
            tools.append(SummaryTool())
    
    # Use the create_with_tools factory method
    agent = LLMAgent.create_with_tools(
        agent_id=agent_id,
        llm=llm,
        tools=tools,
        role="participant",
        system_prompt=system_prompt,
        enable_error_handling=enable_error_handling,
        max_retries=max_retries,
        fallback_llm=fallback_llm
    )
    
    logger.info(
        f"Created tool-enabled participant agent '{agent_id}' "
        f"for phase {phase} with {len(tools)} tools"
    )
    return agent


def create_discussion_agents_with_tools(
    agent_configs: List[Dict[str, Any]],
    phase: int = 1,
    enable_error_handling: bool = True,
    max_retries: int = 3
) -> Dict[str, LLMAgent]:
    """Create multiple discussion agents with appropriate tools.
    
    Args:
        agent_configs: List of agent configurations, each containing:
            - id: Agent ID
            - llm: LangChain chat model
            - role: 'moderator' or 'participant'
            - system_prompt: Optional custom prompt
            - tools: Optional custom tools
        phase: Current discussion phase
        enable_error_handling: Whether to enable error handling
        max_retries: Maximum retry attempts
        
    Returns:
        Dictionary mapping agent IDs to LLMAgent instances
    """
    agents = {}
    
    for config in agent_configs:
        agent_id = config["id"]
        llm = config["llm"]
        role = config.get("role", "participant")
        system_prompt = config.get("system_prompt")
        custom_tools = config.get("tools")
        fallback_llm = config.get("fallback_llm")
        
        if role == "moderator":
            agent = create_tool_enabled_moderator(
                agent_id=agent_id,
                llm=llm,
                system_prompt=system_prompt,
                enable_error_handling=enable_error_handling,
                max_retries=max_retries,
                fallback_llm=fallback_llm
            )
        else:
            agent = create_tool_enabled_participant(
                agent_id=agent_id,
                llm=llm,
                phase=phase,
                system_prompt=system_prompt,
                enable_error_handling=enable_error_handling,
                max_retries=max_retries,
                fallback_llm=fallback_llm,
                custom_tools=custom_tools
            )
        
        agents[agent_id] = agent
    
    logger.info(f"Created {len(agents)} tool-enabled agents")
    return agents


def add_tools_to_existing_agent(
    agent: LLMAgent,
    tools: Sequence[BaseTool],
    replace: bool = False
) -> None:
    """Add tools to an existing agent.
    
    Args:
        agent: Existing LLMAgent instance
        tools: Tools to add
        replace: If True, replace existing tools; if False, append
    """
    if replace:
        agent.bind_tools(tools)
    else:
        # Combine with existing tools
        existing_tools = agent.tools or []
        all_tools = list(existing_tools) + list(tools)
        
        # Remove duplicates based on tool name
        unique_tools = {}
        for tool in all_tools:
            unique_tools[tool.name] = tool
        
        agent.bind_tools(list(unique_tools.values()))
    
    logger.info(
        f"{'Replaced' if replace else 'Added'} {len(tools)} tools "
        f"to agent '{agent.agent_id}'. Total tools: {len(agent.tools)}"
    )


def create_specialized_agent(
    agent_id: str,
    llm: BaseChatModel,
    specialization: str,
    system_prompt: Optional[str] = None,
    enable_error_handling: bool = True,
    max_retries: int = 3,
    fallback_llm: Optional[BaseChatModel] = None
) -> LLMAgent:
    """Create an agent with specialized tool sets.
    
    Specializations:
    - 'analyst': Gets all analysis tools
    - 'facilitator': Gets proposal and voting tools
    - 'summarizer': Gets summary tools
    - 'generalist': Gets all available tools
    
    Args:
        agent_id: Unique identifier for the agent
        llm: LangChain chat model instance
        specialization: Type of specialization
        system_prompt: Optional custom system prompt
        enable_error_handling: Whether to enable error handling
        max_retries: Maximum retry attempts
        fallback_llm: Optional fallback LLM
        
    Returns:
        Specialized tool-enabled agent
    """
    # Define tool sets for each specialization
    tool_sets = {
        "analyst": [SummaryTool()],
        "facilitator": [ProposalTool(), VotingTool()],
        "summarizer": [SummaryTool()],
        "generalist": create_discussion_tools(),
    }
    
    tools = tool_sets.get(specialization, [])
    
    # Adjust system prompt based on specialization
    if not system_prompt:
        prompts = {
            "analyst": (
                "You are an analytical participant focused on understanding "
                "and summarizing key points in the discussion."
            ),
            "facilitator": (
                "You are a facilitative participant who helps guide the "
                "discussion by proposing topics and managing decisions."
            ),
            "summarizer": (
                "You are a participant focused on capturing and summarizing "
                "the essence of the discussion."
            ),
            "generalist": (
                "You are a versatile participant who can propose topics, "
                "vote on decisions, and summarize discussions."
            ),
        }
        system_prompt = prompts.get(specialization)
    
    agent = LLMAgent.create_with_tools(
        agent_id=agent_id,
        llm=llm,
        tools=tools,
        role="participant",
        system_prompt=system_prompt,
        enable_error_handling=enable_error_handling,
        max_retries=max_retries,
        fallback_llm=fallback_llm
    )
    
    logger.info(
        f"Created specialized '{specialization}' agent '{agent_id}' "
        f"with {len(tools)} tools"
    )
    return agent