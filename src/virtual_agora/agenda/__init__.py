"""Agenda Management System for Virtual Agora.

This module provides the democratic agenda setting and modification system
that allows agents to propose, vote on, and dynamically update discussion
topics throughout the session.

The system implements the following key components:
- Topic proposal collection from all agents
- Democratic voting orchestration
- Agenda synthesis and ranking
- Dynamic agenda modifications
- Comprehensive analytics and reporting
- Edge case handling and fault tolerance
"""

from virtual_agora.agenda.manager import AgendaManager
from virtual_agora.agenda.models import (
    Proposal,
    Vote,
    VoteType,
    AgendaItem,
    AgendaState,
    ProposalCollection,
    VoteCollection,
    AgendaModification,
    AgendaAnalytics,
    TopicTransition,
)
from virtual_agora.agenda.voting import VotingOrchestrator
from virtual_agora.agenda.synthesis import AgendaSynthesizer
from virtual_agora.agenda.analytics import AgendaAnalyticsCollector

__all__ = [
    "AgendaManager",
    "Proposal",
    "Vote",
    "VoteType",
    "AgendaItem",
    "AgendaState",
    "ProposalCollection",
    "VoteCollection",
    "AgendaModification",
    "AgendaAnalytics",
    "TopicTransition",
    "VotingOrchestrator",
    "AgendaSynthesizer",
    "AgendaAnalyticsCollector",
]
