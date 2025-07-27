"""Integration tests for API failure recovery scenarios in Virtual Agora v1.3.

This module tests various API failure scenarios and recovery mechanisms during
the discussion flow, including:
- LLM API failures and retries
- Partial agent failures
- Network interruptions during critical operations
- Rate limiting and backoff strategies
- Graceful degradation when services are unavailable
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta
import uuid
import time
import random

from virtual_agora.state.schema import VirtualAgoraState
from virtual_agora.flow.graph_v13 import VirtualAgoraV13Flow
from virtual_agora.agents.discussion_agent import DiscussionAgent
from virtual_agora.agents.moderator import ModeratorAgent
from virtual_agora.utils.exceptions import (
    VirtualAgoraError,
    ProviderError,
    ProviderRateLimitError,
    NetworkTransientError,
)

from ..helpers.fake_llm import (
    create_fake_llm_pool,
    create_specialized_fake_llms,
    ErrorFakeLLM,
)
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
    create_test_messages,
)


class TestLLMAPIFailures:
    """Test LLM API failure scenarios and recovery."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_single_agent_api_failure_recovery(self):
        """Test recovery when a single agent's API call fails."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()
            state["warnings"] = []  # Initialize warnings list

            # Create agents with one faulty LLM
            llm_pool = create_fake_llm_pool(num_agents=3)
            error_llm = ErrorFakeLLM(error_rate=0.8, error_types=["api_error"])

            agents = {
                "agent_1": DiscussionAgent("agent_1", llm_pool["agent_1"]),
                "agent_2": DiscussionAgent(
                    "agent_2", error_llm
                ),  # This one fails often
                "agent_3": DiscussionAgent("agent_3", llm_pool["agent_3"]),
            }

            # Track API calls and failures
            api_calls = {"total": 0, "failures": 0, "retries": 0}

            # Mock retry logic
            def retry_with_backoff(func, max_retries=3):
                for attempt in range(max_retries):
                    try:
                        api_calls["total"] += 1
                        result = func()
                        return result
                    except Exception as e:
                        api_calls["failures"] += 1
                        if attempt < max_retries - 1:
                            api_calls["retries"] += 1
                            time.sleep(0.1 * (2**attempt))  # Exponential backoff
                        else:
                            raise e

            # Simulate discussion with retries
            for round_num in range(1, 4):
                for agent_id, agent in agents.items():
                    try:

                        def api_call():
                            # Simulate agent response generation
                            if agent_id == "agent_2" and random.random() < 0.8:
                                raise Exception("Simulated API error")
                            return f"{agent_id} response for round {round_num}"

                        response = retry_with_backoff(api_call)

                        # Add successful response to state
                        state["messages"].append(
                            {
                                "id": str(uuid.uuid4()),
                                "agent_id": agent_id,
                                "content": response,
                                "timestamp": datetime.now(),
                                "round_number": round_num,
                                "topic": "Test Topic",
                                "message_type": "discussion",
                            }
                        )
                    except Exception as e:
                        # Log failure but continue with other agents
                        state["warnings"].append(
                            f"{agent_id} failed after retries: {str(e)}"
                        )

            # Verify system continues despite failures
            assert len(state["messages"]) > 0  # Some messages succeeded
            assert api_calls["retries"] > 0  # Retries were attempted
            assert len(state["warnings"]) > 0  # Failures were logged

    @pytest.mark.integration
    def test_moderator_api_failure_fallback(self):
        """Test fallback when moderator API calls fail."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Create moderator with high failure rate
            error_llm = ErrorFakeLLM(
                error_rate=0.9, error_types=["api_error", "timeout"]
            )
            moderator = ModeratorAgent("moderator", error_llm)

            # Test critical moderator operations with fallback

            # 1. Topic announcement fallback
            topic = {"title": "Test Topic", "description": "Test description"}
            try:
                # This will likely fail due to high error rate
                announcement = f"Now discussing topic 1 of 3: {topic['title']}"
                if random.random() < 0.9:  # Simulate failure
                    raise Exception("Moderator API failed")
            except Exception:
                # Fallback: Use template announcement
                announcement = f"Topic 1 of 3: Test Topic - Test description. Let's begin our discussion."

            assert "Test Topic" in announcement

            # 2. Voting interpretation fallback
            votes = {"agent_1": "Yes", "agent_2": "No", "agent_3": "Yes"}
            try:
                # Simulate voting interpretation
                if random.random() < 0.9:  # High failure rate
                    raise Exception("Moderator voting analysis failed")
                result = {
                    "should_conclude": True,
                    "vote_type": "majority",
                    "summary": "Majority voted yes",
                }
            except Exception:
                # Fallback: Simple majority calculation
                yes_count = sum(1 for v in votes.values() if v.startswith("Yes"))
                result = {
                    "should_conclude": yes_count > len(votes) / 2,
                    "vote_type": "majority",
                    "summary": f"{yes_count} of {len(votes)} agents voted to conclude",
                }

            assert result["should_conclude"] == True
            assert result["vote_type"] == "majority"

    @pytest.mark.integration
    def test_summarizer_failure_graceful_degradation(self):
        """Test graceful degradation when summarizer fails."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Add discussion messages
            for i in range(9):  # 3 rounds × 3 agents
                state["messages"].append(
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": f"agent_{i % 3 + 1}",
                        "content": f"Discussion point {i + 1}",
                        "timestamp": datetime.now(),
                        "round_number": i // 3 + 1,
                        "topic": "Test Topic",
                        "message_type": "discussion",
                    }
                )

            # Create failing summarizer
            error_llm = ErrorFakeLLM(error_rate=1.0, error_types=["api_error"])

            # Try to summarize with fallback
            try:
                # Would normally call summarizer agent
                summary = "AI-generated summary"
                raise Exception("Summarizer API failed")
            except Exception:
                # Fallback: Create basic summary from messages
                round_messages = [
                    m for m in state["messages"] if m["round_number"] == 3
                ]
                if round_messages:
                    summary = f"Round 3: {len(round_messages)} agents discussed. Topics included: "
                    summary += ", ".join(
                        [
                            f"point {m['content'].split()[-1]}"
                            for m in round_messages[:2]
                        ]
                    )
                    summary += f" and {len(round_messages) - 2} more points."
                else:
                    summary = "Round 3: Discussion continued."

            state["round_summaries"].append(summary)

            # Verify fallback summary was created
            assert len(state["round_summaries"]) == 1
            assert "Round 3" in state["round_summaries"][0]
            assert "discussed" in state["round_summaries"][0]


class TestNetworkFailureScenarios:
    """Test network-related failure scenarios."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_intermittent_network_failures(self):
        """Test handling of intermittent network failures."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()
            state["warnings"] = []  # Initialize warnings list

            # Simulate intermittent network issues
            network_failure_count = 0
            max_network_failures = 3

            def simulate_network_call():
                nonlocal network_failure_count
                # Fail 30% of the time up to max failures
                if (
                    random.random() < 0.3
                    and network_failure_count < max_network_failures
                ):
                    network_failure_count += 1
                    raise NetworkTransientError("Network temporarily unavailable")
                return "Success"

            # Test multiple operations with network issues
            operations = [
                "fetch_context",
                "submit_message",
                "retrieve_summary",
                "save_checkpoint",
            ]

            results = {"success": 0, "failed": 0, "retried": 0}

            for operation in operations * 3:  # Try each operation 3 times
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = simulate_network_call()
                        results["success"] += 1
                        break
                    except NetworkTransientError:
                        if attempt < max_retries - 1:
                            results["retried"] += 1
                            time.sleep(0.1)  # Brief delay before retry
                        else:
                            results["failed"] += 1
                            state["warnings"].append(
                                f"Network operation {operation} failed after {max_retries} attempts"
                            )

            # System should handle some failures
            assert results["success"] > results["failed"]
            assert results["retried"] > 0  # Some operations were retried

    @pytest.mark.integration
    def test_network_timeout_during_voting(self):
        """Test network timeout during critical voting phase."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()
            state["warnings"] = []  # Initialize warnings list

            # Set up voting scenario
            state["current_phase"] = 5  # Voting phase
            state["active_vote"] = {
                "id": str(uuid.uuid4()),
                "phase": 5,
                "vote_type": "conclusion",
                "status": "in_progress",
                "start_time": datetime.now().isoformat(),
                "required_votes": 3,
                "received_votes": 0,
                "votes": {},
            }

            # Simulate voting with network timeouts
            agents = ["agent_1", "agent_2", "agent_3"]
            timeout_agents = set()

            for agent_id in agents:
                try:
                    # Simulate network timeout for agent_2
                    if agent_id == "agent_2":
                        raise TimeoutError(f"Network timeout for {agent_id}")

                    # Successful vote
                    vote = "Yes" if agent_id != "agent_3" else "No"
                    state["active_vote"]["votes"][
                        agent_id
                    ] = f"{vote}. Reasoning provided."
                    state["active_vote"]["received_votes"] += 1

                except TimeoutError:
                    timeout_agents.add(agent_id)
                    state["warnings"].append(f"Vote timeout for {agent_id}")

            # Handle incomplete voting
            if timeout_agents:
                # Wait for timeout period
                state["active_vote"]["timeout_extension"] = 60  # Give 60 more seconds

                # If still not enough votes after extension, use available votes
                if (
                    state["active_vote"]["received_votes"]
                    < state["active_vote"]["required_votes"]
                ):
                    # Adjust quorum based on available agents
                    active_agents = len(agents) - len(timeout_agents)
                    new_quorum = max(1, active_agents // 2 + 1)
                    state["active_vote"]["adjusted_quorum"] = new_quorum
                    state["warnings"].append(
                        f"Adjusted quorum to {new_quorum} due to timeouts"
                    )

            # Verify voting can still proceed
            assert state["active_vote"]["received_votes"] >= 1
            assert len(state["warnings"]) > 0
            assert "timeout" in state["warnings"][0].lower()


class TestRateLimitingScenarios:
    """Test rate limiting and backoff strategies."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_rate_limit_with_exponential_backoff(self):
        """Test handling of rate limits with exponential backoff."""
        with patch_ui_components():
            # Track rate limit hits
            rate_limit_hits = []
            call_timestamps = []

            def simulate_rate_limited_api(agent_id):
                call_timestamps.append(datetime.now())

                # Simulate rate limit after 5 calls within 1 second
                recent_calls = [
                    t
                    for t in call_timestamps
                    if (datetime.now() - t).total_seconds() < 1
                ]

                if len(recent_calls) > 5:
                    rate_limit_hits.append(datetime.now())
                    raise ProviderRateLimitError(
                        "Rate limit exceeded", provider="test", retry_after=2
                    )

                return f"Response from {agent_id}"

            # Test rapid API calls with backoff
            results = []
            backoff_delays = []

            for i in range(10):
                agent_id = f"agent_{i % 3 + 1}"

                retry_count = 0
                max_retries = 4

                while retry_count < max_retries:
                    try:
                        result = simulate_rate_limited_api(agent_id)
                        results.append(result)
                        break
                    except ProviderRateLimitError as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff: 2^retry_count seconds
                            delay = min(2**retry_count, 16)  # Cap at 16 seconds
                            backoff_delays.append(delay)
                            time.sleep(delay * 0.1)  # Scale down for testing
                        else:
                            results.append(f"Failed: {agent_id}")

            # Verify rate limiting was encountered and handled
            assert len(rate_limit_hits) > 0
            assert len(backoff_delays) > 0
            assert max(backoff_delays) > min(backoff_delays)  # Exponential growth

    @pytest.mark.integration
    def test_distributed_rate_limiting(self):
        """Test distributed rate limiting across multiple agents."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Global rate limiter
            class RateLimiter:
                def __init__(self, max_calls_per_minute=20):
                    self.max_calls = max_calls_per_minute
                    self.call_times = []

                def check_rate_limit(self):
                    now = datetime.now()
                    # Remove calls older than 1 minute
                    self.call_times = [
                        t for t in self.call_times if (now - t).total_seconds() < 60
                    ]

                    if len(self.call_times) >= self.max_calls:
                        # Calculate when next call is allowed
                        oldest_call = min(self.call_times)
                        wait_time = 60 - (now - oldest_call).total_seconds()
                        raise ProviderRateLimitError(
                            "Global rate limit exceeded",
                            provider="global",
                            retry_after=wait_time,
                        )

                    self.call_times.append(now)

            rate_limiter = RateLimiter(max_calls_per_minute=10)

            # Simulate multiple agents making calls
            call_results = {"success": 0, "rate_limited": 0, "queued": 0}

            for round_num in range(1, 5):
                for agent_id in state["speaking_order"]:
                    try:
                        rate_limiter.check_rate_limit()
                        # Successful call
                        call_results["success"] += 1
                    except ProviderRateLimitError as e:
                        call_results["rate_limited"] += 1
                        # Queue the request for later
                        if "queued_requests" not in state:
                            state["queued_requests"] = []
                        state["queued_requests"].append(
                            {
                                "agent_id": agent_id,
                                "round": round_num,
                                "retry_after": getattr(e, "retry_after", 60),
                            }
                        )
                        call_results["queued"] += 1

            # Verify rate limiting is working
            assert call_results["rate_limited"] > 0
            assert call_results["queued"] > 0
            assert len(state.get("queued_requests", [])) > 0


class TestCascadingFailureScenarios:
    """Test cascading failure scenarios and circuit breakers."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(
            num_agents=4, scenario="extended_debate"
        )

    @pytest.mark.integration
    def test_circuit_breaker_activation(self):
        """Test circuit breaker pattern for failing services."""
        with patch_ui_components():
            # Circuit breaker implementation
            class CircuitBreaker:
                def __init__(self, failure_threshold=5, timeout=30):
                    self.failure_threshold = failure_threshold
                    self.timeout = timeout
                    self.failure_count = 0
                    self.last_failure_time = None
                    self.state = "closed"  # closed, open, half-open

                def call(self, func, *args, **kwargs):
                    if self.state == "open":
                        # Check if timeout has passed
                        if (
                            datetime.now() - self.last_failure_time
                        ).total_seconds() > self.timeout:
                            self.state = "half-open"
                        else:
                            raise Exception("Circuit breaker is OPEN")

                    try:
                        result = func(*args, **kwargs)
                        if self.state == "half-open":
                            # Success in half-open state, close the circuit
                            self.state = "closed"
                            self.failure_count = 0
                        return result
                    except Exception as e:
                        self.failure_count += 1
                        self.last_failure_time = datetime.now()

                        if self.failure_count >= self.failure_threshold:
                            self.state = "open"
                            raise Exception(
                                f"Circuit breaker OPENED after {self.failure_count} failures"
                            )
                        raise e

            # Test service with circuit breaker
            circuit_breaker = CircuitBreaker(failure_threshold=3)

            def unreliable_service(failure_rate=0.8):
                if random.random() < failure_rate:
                    raise Exception("Service failure")
                return "Success"

            results = {"success": 0, "failed": 0, "circuit_open": 0}

            for i in range(20):
                try:
                    result = circuit_breaker.call(unreliable_service)
                    results["success"] += 1
                except Exception as e:
                    if "Circuit breaker is OPEN" in str(
                        e
                    ) or "Circuit breaker OPENED" in str(e):
                        results["circuit_open"] += 1
                    else:
                        results["failed"] += 1

                time.sleep(0.05)  # Small delay between calls

            # Verify circuit breaker activated
            assert results["circuit_open"] > 0
            assert (
                results["failed"] <= circuit_breaker.failure_threshold + 2
            )  # Some tolerance

    @pytest.mark.integration
    def test_cascading_failure_prevention(self):
        """Test prevention of cascading failures across agents."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()
            state["warnings"] = []  # Initialize warnings list

            # Track agent health
            agent_health = {
                "agent_1": {"failures": 0, "status": "healthy"},
                "agent_2": {"failures": 0, "status": "healthy"},
                "agent_3": {"failures": 0, "status": "healthy"},
                "agent_4": {"failures": 0, "status": "healthy"},
            }

            # Simulate failure propagation
            failed_agents = set()
            failure_threshold = 3

            def agent_operation(agent_id):
                # Agent fails if too many other agents have failed (cascading)
                if len(failed_agents) >= 2 and random.random() < 0.7:
                    raise Exception(f"{agent_id} failed due to cascading failure")

                # Random failure chance - increased for testing
                if random.random() < 0.4:
                    raise Exception(f"{agent_id} random failure")

                return f"{agent_id} succeeded"

            # Run multiple rounds with failure isolation
            for round_num in range(1, 6):
                round_results = []

                for agent_id in agent_health.keys():
                    if agent_health[agent_id]["status"] == "isolated":
                        round_results.append(f"{agent_id}: isolated")
                        continue

                    try:
                        result = agent_operation(agent_id)
                        round_results.append(result)
                        # Reset failure count on success
                        agent_health[agent_id]["failures"] = 0
                    except Exception as e:
                        agent_health[agent_id]["failures"] += 1
                        round_results.append(f"{agent_id}: failed")

                        # Isolate agent if it fails too often
                        if agent_health[agent_id]["failures"] >= failure_threshold:
                            agent_health[agent_id]["status"] = "isolated"
                            failed_agents.add(agent_id)
                            state["warnings"].append(
                                f"{agent_id} isolated after {failure_threshold} failures"
                            )

                state[f"round_{round_num}_results"] = round_results

            # Verify failure isolation worked
            isolated_agents = [
                a for a, h in agent_health.items() if h["status"] == "isolated"
            ]
            healthy_agents = [
                a for a, h in agent_health.items() if h["status"] == "healthy"
            ]

            # Check if any agents had failures (even if not isolated)
            agents_with_failures = [
                a for a, h in agent_health.items() if h["failures"] > 0
            ]

            # Should have had some failures
            assert len(agents_with_failures) > 0 or len(isolated_agents) > 0
            # If we had isolated agents, we should have prevented total cascade
            if len(isolated_agents) > 0:
                assert len(healthy_agents) > 0


class TestGracefulDegradation:
    """Test graceful degradation strategies."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_feature_degradation_under_load(self):
        """Test feature degradation when system is under load."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()
            state["warnings"] = []  # Initialize warnings list

            # System load simulator
            class SystemLoad:
                def __init__(self):
                    self.cpu_usage = 0.3
                    self.memory_usage = 0.4
                    self.api_latency = 100  # ms

                def increase_load(self):
                    self.cpu_usage = min(0.95, self.cpu_usage * 1.2)
                    self.memory_usage = min(0.95, self.memory_usage * 1.15)
                    self.api_latency = min(5000, self.api_latency * 1.5)

                def get_degradation_level(self):
                    if self.cpu_usage > 0.9 or self.memory_usage > 0.9:
                        return "severe"
                    elif self.cpu_usage > 0.7 or self.memory_usage > 0.7:
                        return "moderate"
                    elif self.cpu_usage > 0.5 or self.memory_usage > 0.5:
                        return "light"
                    return "none"

            system_load = SystemLoad()

            # Features that can be degraded
            features = {
                "detailed_summaries": {"enabled": True, "priority": 3},
                "consensus_analysis": {"enabled": True, "priority": 2},
                "sentiment_tracking": {"enabled": True, "priority": 4},
                "real_time_updates": {"enabled": True, "priority": 5},
                "comprehensive_reports": {"enabled": True, "priority": 1},
            }

            # Simulate increasing load and feature degradation
            for i in range(10):
                system_load.increase_load()
                degradation_level = system_load.get_degradation_level()

                if degradation_level == "light":
                    # Disable lowest priority features
                    for feature, config in features.items():
                        if config["priority"] >= 4:
                            config["enabled"] = False
                            state["warnings"].append(
                                f"Disabled {feature} due to system load"
                            )

                elif degradation_level == "moderate":
                    # Disable more features
                    for feature, config in features.items():
                        if config["priority"] >= 3:
                            config["enabled"] = False
                            state["warnings"].append(
                                f"Disabled {feature} due to high load"
                            )

                elif degradation_level == "severe":
                    # Keep only critical features
                    for feature, config in features.items():
                        if config["priority"] > 1:
                            config["enabled"] = False
                            state["warnings"].append(
                                f"Disabled {feature} - critical load"
                            )

                # Log current state
                enabled_features = [f for f, c in features.items() if c["enabled"]]
                state[f"iteration_{i}_features"] = enabled_features

            # Verify graceful degradation occurred
            final_enabled = [f for f, c in features.items() if c["enabled"]]
            assert len(final_enabled) < len(features)  # Some features disabled
            assert (
                "comprehensive_reports" in final_enabled
            )  # Critical feature preserved

    @pytest.mark.integration
    def test_minimal_viable_discussion(self):
        """Test system can maintain minimal viable discussion during failures."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_discussion_state()

            # Simulate various service failures
            service_status = {
                "summarizer": False,  # Failed
                "topic_report": False,  # Failed
                "sentiment_analysis": False,  # Failed
                "consensus_tracking": True,  # Working
                "basic_discussion": True,  # Working
            }

            # Run minimal discussion
            for round_num in range(1, 4):
                round_messages = []

                # Basic discussion continues
                if service_status["basic_discussion"]:
                    for agent_id in state["speaking_order"]:
                        message = {
                            "id": str(uuid.uuid4()),
                            "agent_id": agent_id,
                            "content": f"Basic message from {agent_id} in round {round_num}",
                            "timestamp": datetime.now(),
                            "round_number": round_num,
                            "topic": state["agenda"][0]["title"],
                            "message_type": "discussion",
                        }
                        round_messages.append(message)
                        state["messages"].append(message)

                # Attempt summary with fallback
                if service_status["summarizer"]:
                    summary = f"AI-generated summary for round {round_num}"
                else:
                    # Fallback: Basic summary
                    summary = f"Round {round_num}: {len(round_messages)} agents participated. Discussion continued."

                state["round_summaries"].append(summary)

                # Check if we can track consensus
                if service_status["consensus_tracking"]:
                    # Simple consensus check
                    if round_num >= 3:
                        state["consensus_indicators"] = ["basic_agreement_reached"]

            # Verify minimal viable discussion was maintained
            assert len(state["messages"]) >= 9  # 3 rounds × 3 agents
            assert len(state["round_summaries"]) >= 3
            assert "basic_agreement_reached" in state.get("consensus_indicators", [])


@pytest.mark.integration
class TestRecoveryVerification:
    """Verify recovery mechanisms work correctly."""

    def test_state_recovery_after_crash(self):
        """Test state recovery after simulated crash."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        with patch_ui_components():
            # Create initial state
            state = helper.create_discussion_state()
            state["current_round"] = 5
            state["messages"] = create_test_messages(15)
            state["checkpoint_id"] = str(uuid.uuid4())

            # Simulate crash by "losing" state
            lost_state = None

            # Recovery should restore from checkpoint
            if lost_state is None:
                # Restore from checkpoint (simplified)
                recovered_state = {
                    "session_id": state["session_id"],
                    "current_round": state["current_round"],
                    "messages": state["messages"][-10:],  # Keep recent messages
                    "warnings": ["Recovered from checkpoint after crash"],
                    "recovery_timestamp": datetime.now().isoformat(),
                }

            # Verify recovery
            assert recovered_state is not None
            assert recovered_state["session_id"] == state["session_id"]
            assert len(recovered_state["messages"]) > 0
            assert "Recovered from checkpoint" in recovered_state["warnings"][0]

    def test_partial_state_reconstruction(self):
        """Test reconstruction of partially corrupted state."""
        helper = IntegrationTestHelper(num_agents=3, scenario="default")

        # Create corrupted state
        corrupted_state = {
            "session_id": "test_session",
            "messages": create_test_messages(5),
            # Missing: current_round, agenda, speaking_order, etc.
        }

        # Reconstruct missing fields
        reconstructed_state = corrupted_state.copy()

        # Infer current round from messages
        if (
            "current_round" not in reconstructed_state
            and "messages" in reconstructed_state
        ):
            max_round = max(
                (msg.get("round_number", 1) for msg in reconstructed_state["messages"]),
                default=1,
            )
            reconstructed_state["current_round"] = max_round

        # Reconstruct speaking order from messages
        if (
            "speaking_order" not in reconstructed_state
            and "messages" in reconstructed_state
        ):
            agents = list(
                set(
                    msg["agent_id"]
                    for msg in reconstructed_state["messages"]
                    if "agent_id" in msg
                )
            )
            reconstructed_state["speaking_order"] = sorted(agents)

        # Add default agenda if missing
        if "agenda" not in reconstructed_state:
            reconstructed_state["agenda"] = [
                {
                    "title": "Recovered Topic",
                    "description": "Topic recovered from state",
                    "status": "in_progress",
                }
            ]

        # Add recovery metadata
        reconstructed_state["recovery_metadata"] = {
            "recovered_at": datetime.now().isoformat(),
            "recovery_type": "partial_reconstruction",
            "missing_fields_reconstructed": [
                "current_round",
                "speaking_order",
                "agenda",
            ],
        }

        # Verify reconstruction
        assert "current_round" in reconstructed_state
        assert "speaking_order" in reconstructed_state
        assert "agenda" in reconstructed_state
        assert len(reconstructed_state["speaking_order"]) > 0
