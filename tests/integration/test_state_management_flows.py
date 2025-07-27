"""Integration tests for State Management & Recovery flows.

This module tests the complete state management workflow including
state persistence, recovery points, rollback functionality, and validation.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta
import uuid

from virtual_agora.state.schema import VirtualAgoraState, FlowControl, HITLState
from virtual_agora.state.manager import StateManager
from virtual_agora.state.recovery import StateRecoveryManager
from virtual_agora.flow.graph import VirtualAgoraFlow
from virtual_agora.flow.persistence import EnhancedMemorySaver
from virtual_agora.utils.exceptions import StateError, ValidationError

from ..helpers.fake_llm import ModeratorFakeLLM, AgentFakeLLM
from ..helpers.integration_utils import (
    IntegrationTestHelper,
    StateTestBuilder,
    TestResponseValidator,
    patch_ui_components,
)


class TestStatePersistenceFlow:
    """Test state persistence and checkpointing workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_state_persistence_across_phases(self):
        """Test state persistence across different discussion phases."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state = self.test_helper.create_basic_state()

            # Simulate progression through phases
            phase_states = []

            # Phase 0: Initialization
            phase_0_state = self._simulate_phase_progression(state, 0)
            phase_states.append(("initialization", phase_0_state))

            # Phase 1: Agenda Setting
            phase_1_state = self._simulate_phase_progression(phase_0_state, 1)
            phase_states.append(("agenda_setting", phase_1_state))

            # Phase 2: Discussion
            phase_2_state = self._simulate_phase_progression(phase_1_state, 2)
            phase_states.append(("discussion", phase_2_state))

            # Verify state persistence through each phase
            for phase_name, phase_state in phase_states:
                # Verify essential state is preserved
                assert "session_id" in phase_state
                assert "current_phase" in phase_state
                assert phase_state["session_id"] == state["session_id"]

                # Verify phase-specific state accumulation
                if phase_name == "agenda_setting":
                    assert "agenda" in phase_state
                elif phase_name == "discussion":
                    assert "messages" in phase_state
                    assert len(phase_state["messages"]) > 0

    @pytest.mark.integration
    def test_enhanced_checkpointing_functionality(self):
        """Test enhanced checkpointing with recovery points."""
        with patch_ui_components():
            # Create flow with enhanced checkpointer
            config = self.test_helper.config
            flow = VirtualAgoraFlow(config, enable_monitoring=True)
            flow.compile()

            session_id = flow.create_session()
            state_manager = flow.get_state_manager()

            # Create multiple recovery points
            recovery_points = []

            # Recovery point 1: After initialization
            state_manager.update_state({"current_phase": 0, "main_topic": "Test Topic"})
            rp1_id = flow.create_recovery_point(
                "After initialization", {"phase": "init"}
            )
            recovery_points.append(rp1_id)

            # Recovery point 2: After agenda setting
            state_manager.update_state(
                {
                    "current_phase": 1,
                    "agenda": [{"title": "Topic 1", "status": "pending"}],
                }
            )
            rp2_id = flow.create_recovery_point(
                "After agenda setting", {"phase": "agenda"}
            )
            recovery_points.append(rp2_id)

            # Recovery point 3: During discussion
            state_manager.update_state(
                {
                    "current_phase": 2,
                    "messages": [
                        {
                            "id": "msg1",
                            "content": "Test message",
                            "timestamp": datetime.now(),
                        }
                    ],
                }
            )
            rp3_id = flow.create_recovery_point(
                "During discussion", {"phase": "discussion"}
            )
            recovery_points.append(rp3_id)

            # Verify all recovery points exist
            all_recovery_points = flow.get_recovery_points()
            assert len(all_recovery_points) == 3

            # Test rollback to specific recovery point
            current_state_before = state_manager.get_snapshot()
            assert current_state_before["current_phase"] == 2

            # Rollback to agenda setting phase
            flow.rollback_to_recovery_point(rp2_id)

            # Verify rollback worked
            current_state_after = state_manager.get_snapshot()
            assert current_state_after["current_phase"] == 1
            assert "agenda" in current_state_after
            assert (
                len(current_state_after.get("messages", [])) == 0
            )  # Messages should be gone

    @pytest.mark.integration
    def test_state_validation_during_persistence(self):
        """Test state validation during persistence operations."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Test valid state update
            valid_update = {
                "current_phase": 1,
                "agenda": [{"title": "Valid Topic", "status": "pending"}],
                "current_round": 1,
            }

            # Should succeed
            state_manager.update_state(valid_update)
            updated_state = state_manager.get_snapshot()
            assert updated_state["current_phase"] == 1

            # Test invalid state update
            invalid_update = {
                "current_phase": "invalid_phase",  # Should be int
                "invalid_field": "should not exist",
            }

            # Should handle gracefully or raise appropriate error
            try:
                state_manager.update_state(invalid_update)
                # If it doesn't raise an error, verify the state wasn't corrupted
                final_state = state_manager.get_snapshot()
                assert isinstance(final_state["current_phase"], int)
            except (ValidationError, ValueError, TypeError):
                # Expected behavior - validation caught the error
                pass

    @pytest.mark.integration
    def test_concurrent_state_access(self):
        """Test concurrent state access and modification."""
        import threading
        import time

        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            results = []
            errors = []

            def update_state_worker(worker_id):
                try:
                    for i in range(5):
                        update = {
                            f"worker_{worker_id}_update_{i}": f"value_{i}",
                            "last_update_by": worker_id,
                            "update_timestamp": datetime.now(),
                        }
                        state_manager.update_state(update)
                        time.sleep(0.01)  # Small delay

                        # Read state
                        snapshot = state_manager.get_snapshot()
                        results.append((worker_id, i, snapshot.get("last_update_by")))

                except Exception as e:
                    errors.append((worker_id, str(e)))

            # Run multiple workers concurrently
            threads = []
            for worker_id in range(3):
                thread = threading.Thread(target=update_state_worker, args=(worker_id,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify no errors occurred
            assert len(errors) == 0, f"Concurrent access errors: {errors}"

            # Verify all updates were processed
            assert len(results) == 15  # 3 workers * 5 updates each

            # Verify final state consistency
            final_state = state_manager.get_snapshot()
            assert "last_update_by" in final_state
            assert final_state["last_update_by"] in [0, 1, 2]

    def _simulate_phase_progression(
        self, state: VirtualAgoraState, target_phase: int
    ) -> VirtualAgoraState:
        """Simulate progression to a specific phase."""
        updated_state = state.copy()
        updated_state["current_phase"] = target_phase
        updated_state["phase_start_time"] = datetime.now()

        if target_phase == 0:
            # Initialization phase
            updated_state["main_topic"] = "Test Discussion Topic"
            updated_state["speaking_order"] = [
                f"agent_{i+1}" for i in range(self.test_helper.num_agents)
            ]
            updated_state["moderator_id"] = "moderator"

        elif target_phase == 1:
            # Agenda setting phase
            updated_state["agenda"] = [
                {
                    "title": "Technical Implementation",
                    "description": "Tech details",
                    "status": "pending",
                },
                {
                    "title": "Business Impact",
                    "description": "ROI analysis",
                    "status": "pending",
                },
                {
                    "title": "Risk Assessment",
                    "description": "Risk mitigation",
                    "status": "pending",
                },
            ]

        elif target_phase == 2:
            # Discussion phase
            updated_state["current_topic_index"] = 0
            updated_state["current_round"] = 1
            updated_state["messages"] = [
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "agent_1",
                    "content": "Opening discussion on technical implementation.",
                    "timestamp": datetime.now(),
                    "round_number": 1,
                    "topic": "Technical Implementation",
                    "message_type": "discussion",
                }
            ]

        return updated_state


class TestStateRecoveryFlow:
    """Test state recovery and error handling workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_state_corruption_recovery(self):
        """Test recovery from state corruption."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Create a good state and recovery point
            good_state = {
                "current_phase": 1,
                "main_topic": "Valid Topic",
                "agenda": [{"title": "Topic 1", "status": "pending"}],
                "session_id": state_manager.state["session_id"],
            }
            state_manager.update_state(good_state)
            recovery_id = flow.create_recovery_point("Good state checkpoint")

            # Simulate state corruption
            corrupted_state = {
                "current_phase": None,  # Invalid
                "invalid_data": {"broken": "structure"},
                "messages": "not_a_list",  # Should be list
            }

            # Attempt to update with corrupted state
            try:
                state_manager.update_state(corrupted_state)
                current_state = state_manager.get_snapshot()

                # Check if state is still valid or if corruption was prevented
                if current_state.get("current_phase") is None:
                    # State was corrupted, trigger recovery
                    flow.rollback_to_recovery_point(recovery_id)

            except (ValidationError, ValueError, TypeError):
                # Validation prevented corruption - this is good
                pass

            # Verify state is now valid
            final_state = state_manager.get_snapshot()
            assert final_state["current_phase"] == 1
            assert final_state["main_topic"] == "Valid Topic"
            assert isinstance(final_state.get("agenda", []), list)

    @pytest.mark.integration
    def test_session_interruption_recovery(self):
        """Test recovery from session interruption."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            session_id = flow.create_session()
            state_manager = flow.get_state_manager()

            # Simulate active discussion session
            active_state = {
                "current_phase": 2,
                "current_round": 3,
                "current_topic_index": 1,
                "agenda": [
                    {"title": "Topic 1", "status": "completed"},
                    {"title": "Topic 2", "status": "active"},
                    {"title": "Topic 3", "status": "pending"},
                ],
                "messages": [
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": "agent_1",
                        "content": "Mid-discussion message",
                        "timestamp": datetime.now(),
                        "round_number": 3,
                        "topic": "Topic 2",
                        "message_type": "discussion",
                    }
                ],
                "session_metadata": {
                    "last_activity": datetime.now(),
                    "interruption_point": "discussion_round_3",
                },
            }
            state_manager.update_state(active_state)

            # Create recovery point before "interruption"
            recovery_id = flow.create_recovery_point("Before interruption")

            # Simulate session interruption (simulate system restart)
            # In real scenario, this would be loading from persistent storage

            # Create new flow instance (simulating restart)
            flow_after_restart = VirtualAgoraFlow(self.test_helper.config)
            flow_after_restart.compile()

            # Restore from recovery point (in real implementation, this would
            # automatically detect and restore the most recent state)
            recovery_points = flow.get_recovery_points()

            # Verify we can continue from where we left off
            assert len(recovery_points) > 0
            latest_recovery = recovery_points[-1]
            assert "Before interruption" in latest_recovery["description"]

    @pytest.mark.integration
    def test_partial_state_recovery(self):
        """Test recovery when only partial state is corrupted."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Create comprehensive state
            comprehensive_state = {
                "current_phase": 2,
                "main_topic": "Important Discussion",
                "agenda": [
                    {"title": "Topic 1", "status": "completed"},
                    {"title": "Topic 2", "status": "active"},
                ],
                "messages": [
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": "agent_1",
                        "content": "Important message 1",
                        "timestamp": datetime.now(),
                        "round_number": 1,
                        "topic": "Topic 1",
                        "message_type": "discussion",
                    },
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": "agent_2",
                        "content": "Important message 2",
                        "timestamp": datetime.now(),
                        "round_number": 1,
                        "topic": "Topic 1",
                        "message_type": "discussion",
                    },
                ],
                "voting_rounds": [
                    {
                        "id": str(uuid.uuid4()),
                        "vote_type": "conclusion",
                        "result": "Yes",
                        "status": "completed",
                    }
                ],
            }
            state_manager.update_state(comprehensive_state)
            recovery_id = flow.create_recovery_point("Complete state")

            # Simulate partial corruption - only messages are corrupted
            recovery_manager = StateRecoveryManager()
            current_state = state_manager.get_snapshot()

            # Corrupt only the messages
            corrupted_messages = "invalid_message_data"
            partially_corrupted_state = current_state.copy()
            partially_corrupted_state["messages"] = corrupted_messages

            # Attempt recovery
            recovered_state = recovery_manager.recover_partial_state(
                current_state=partially_corrupted_state,
                reference_state=comprehensive_state,
                corrupted_fields=["messages"],
            )

            # Verify recovery
            assert recovered_state["current_phase"] == 2
            assert recovered_state["main_topic"] == "Important Discussion"
            assert isinstance(recovered_state["messages"], list)
            assert len(recovered_state["messages"]) == 2
            assert recovered_state["agenda"] == comprehensive_state["agenda"]

    @pytest.mark.integration
    def test_automatic_state_validation_and_repair(self):
        """Test automatic state validation and repair mechanisms."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Create state with some inconsistencies
            inconsistent_state = {
                "current_phase": 2,  # Discussion phase
                "current_topic_index": 5,  # But only 2 topics in agenda
                "agenda": [
                    {"title": "Topic 1", "status": "pending"},
                    {"title": "Topic 2", "status": "pending"},
                ],
                "current_round": 0,  # Invalid (should be >= 1 in discussion)
                "messages": [],
                "speaking_order": ["agent_1", "agent_2", "agent_3"],
            }

            # Apply state with validation
            validator = TestResponseValidator()
            validation_issues = validator.validate_state_consistency(inconsistent_state)

            # Should detect issues
            assert len(validation_issues) > 0

            # Apply automatic repairs
            repaired_state = self._apply_automatic_state_repair(inconsistent_state)

            # Verify repairs
            assert repaired_state["current_topic_index"] < len(repaired_state["agenda"])
            assert repaired_state["current_round"] >= 1

            # Re-validate
            validation_issues_after = validator.validate_state_consistency(
                repaired_state
            )
            assert len(validation_issues_after) < len(validation_issues)

    def _apply_automatic_state_repair(
        self, state: VirtualAgoraState
    ) -> VirtualAgoraState:
        """Apply automatic state repair logic."""
        repaired_state = state.copy()

        # Fix current_topic_index if out of bounds
        agenda_length = len(repaired_state.get("agenda", []))
        if repaired_state.get("current_topic_index", 0) >= agenda_length:
            repaired_state["current_topic_index"] = max(0, agenda_length - 1)

        # Fix current_round if invalid
        if (
            repaired_state.get("current_round", 0) < 1
            and repaired_state.get("current_phase") == 2
        ):
            repaired_state["current_round"] = 1

        # Ensure messages is a list
        if not isinstance(repaired_state.get("messages", []), list):
            repaired_state["messages"] = []

        # Ensure speaking_order exists
        if "speaking_order" not in repaired_state:
            repaired_state["speaking_order"] = [f"agent_{i+1}" for i in range(3)]

        return repaired_state


class TestStateSnapshotFlow:
    """Test state snapshot and comparison workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")

    @pytest.mark.integration
    def test_state_snapshot_creation_and_comparison(self):
        """Test creating and comparing state snapshots."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Create initial snapshot
            initial_state = {
                "current_phase": 1,
                "agenda": [{"title": "Topic 1", "status": "pending"}],
                "messages": [],
            }
            state_manager.update_state(initial_state)
            snapshot1_id = flow.create_state_snapshot({"description": "Initial state"})

            # Make changes
            state_manager.update_state(
                {
                    "current_phase": 2,
                    "messages": [
                        {
                            "id": str(uuid.uuid4()),
                            "agent_id": "agent_1",
                            "content": "First message",
                            "timestamp": datetime.now(),
                            "round_number": 1,
                            "topic": "Topic 1",
                            "message_type": "discussion",
                        }
                    ],
                }
            )

            # Create second snapshot
            snapshot2_id = flow.create_state_snapshot(
                {"description": "After first message"}
            )

            # Make more changes
            state_manager.update_state(
                {
                    "messages": [
                        {
                            "id": str(uuid.uuid4()),
                            "agent_id": "agent_2",
                            "content": "Second message",
                            "timestamp": datetime.now(),
                            "round_number": 1,
                            "topic": "Topic 1",
                            "message_type": "discussion",
                        }
                    ]
                }
            )

            # Compare snapshots
            current_state = state_manager.get_snapshot()

            # Verify progression
            assert len(current_state["messages"]) == 2
            assert current_state["current_phase"] == 2

            # Verify snapshots were created
            assert snapshot1_id is not None
            assert snapshot2_id is not None
            assert snapshot1_id != snapshot2_id

    @pytest.mark.integration
    def test_state_diff_and_change_tracking(self):
        """Test state difference tracking and change analysis."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Baseline state
            baseline_state = {
                "current_phase": 1,
                "agenda": [{"title": "Topic 1", "status": "pending"}],
                "messages": [],
                "current_round": 1,
            }
            state_manager.update_state(baseline_state)
            baseline_snapshot = state_manager.get_snapshot()

            # Make incremental changes
            changes = [
                {"current_phase": 2},  # Phase change
                {"messages": [{"id": "msg1", "content": "Message 1"}]},  # Add message
                {"current_round": 2},  # Round change
                {"agenda": [{"title": "Topic 1", "status": "active"}]},  # Status change
            ]

            snapshots = [baseline_snapshot]
            for change in changes:
                state_manager.update_state(change)
                snapshots.append(state_manager.get_snapshot())

            # Analyze changes between snapshots
            change_analysis = self._analyze_state_changes(snapshots)

            # Verify change tracking
            assert len(change_analysis) == len(changes)
            assert "current_phase" in change_analysis[0]["changed_fields"]
            assert "messages" in change_analysis[1]["changed_fields"]
            assert "current_round" in change_analysis[2]["changed_fields"]
            assert "agenda" in change_analysis[3]["changed_fields"]

    @pytest.mark.integration
    def test_state_version_history(self):
        """Test state version history and timeline tracking."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Create versioned state changes
            versions = []
            base_time = datetime.now()

            for i in range(5):
                version_state = {
                    "current_phase": i // 2,  # Slow phase progression
                    "current_round": i + 1,
                    "version": i + 1,
                    "timestamp": base_time + timedelta(minutes=i * 10),
                    "messages": [
                        {"id": f"msg_{j}", "content": f"Message {j}"}
                        for j in range(i + 1)
                    ],
                }
                state_manager.update_state(version_state)

                # Create snapshot for this version
                snapshot_id = flow.create_state_snapshot(
                    {
                        "version": i + 1,
                        "description": f"Version {i + 1}",
                        "timestamp": version_state["timestamp"],
                    }
                )
                versions.append(snapshot_id)

            # Get version history
            state_history = flow.get_state_history()

            # Verify version history
            assert len(state_history) >= 5

            # Verify chronological ordering
            timestamps = [entry.created_at for entry in state_history]
            assert timestamps == sorted(timestamps)  # Should be chronologically ordered

    def _analyze_state_changes(self, snapshots: list) -> list:
        """Analyze changes between consecutive state snapshots."""
        changes = []

        for i in range(1, len(snapshots)):
            prev_snapshot = snapshots[i - 1]
            curr_snapshot = snapshots[i]

            changed_fields = []
            for key in set(prev_snapshot.keys()) | set(curr_snapshot.keys()):
                prev_value = prev_snapshot.get(key)
                curr_value = curr_snapshot.get(key)

                if prev_value != curr_value:
                    changed_fields.append(key)

            changes.append(
                {
                    "version": i,
                    "changed_fields": changed_fields,
                    "field_count": len(changed_fields),
                }
            )

        return changes


class TestStateValidationFlow:
    """Test state validation and integrity workflows."""

    def setup_method(self):
        """Set up test method."""
        self.test_helper = IntegrationTestHelper(num_agents=3, scenario="default")
        self.validator = TestResponseValidator()

    @pytest.mark.integration
    def test_comprehensive_state_validation(self):
        """Test comprehensive state validation across all phases."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Test Phase 0 validation
            phase_0_state = {
                "current_phase": 0,
                "main_topic": "Test Topic",
                "speaking_order": ["agent_1", "agent_2", "agent_3"],
                "moderator_id": "moderator",
            }
            state_manager.update_state(phase_0_state)
            issues_0 = self.validator.validate_discussion_flow(
                state_manager.get_snapshot()
            )
            assert len(issues_0) == 0  # Should be valid

            # Test Phase 1 validation
            phase_1_state = {
                "current_phase": 1,
                "agenda": [
                    {
                        "title": "Topic 1",
                        "description": "Description",
                        "status": "pending",
                    },
                    {
                        "title": "Topic 2",
                        "description": "Description",
                        "status": "pending",
                    },
                ],
            }
            state_manager.update_state(phase_1_state)
            issues_1 = self.validator.validate_discussion_flow(
                state_manager.get_snapshot()
            )
            assert len(issues_1) == 0  # Should be valid

            # Test Phase 2 validation with invalid state
            invalid_phase_2_state = {
                "current_phase": 2,
                "current_topic_index": 10,  # Out of bounds
                "current_round": 0,  # Invalid (should be >= 1)
                "messages": "not_a_list",  # Should be list
            }
            state_manager.update_state(invalid_phase_2_state)
            issues_2 = self.validator.validate_discussion_flow(
                state_manager.get_snapshot()
            )
            assert len(issues_2) > 0  # Should detect issues

    @pytest.mark.integration
    def test_state_integrity_constraints(self):
        """Test state integrity constraints and validation rules."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Set up valid base state
            base_state = {
                "current_phase": 2,
                "agenda": [
                    {"title": "Topic 1", "status": "active"},
                    {"title": "Topic 2", "status": "pending"},
                ],
                "current_topic_index": 0,
                "current_round": 1,
                "speaking_order": ["agent_1", "agent_2", "agent_3"],
                "messages": [],
            }
            state_manager.update_state(base_state)

            # Test constraint violations
            constraint_tests = [
                # Test 1: Current topic index beyond agenda bounds
                {
                    "update": {"current_topic_index": 5},
                    "should_be_valid": False,
                    "constraint": "topic_index_bounds",
                },
                # Test 2: Invalid phase transition
                {
                    "update": {"current_phase": 5},  # Skip phases
                    "should_be_valid": False,
                    "constraint": "phase_sequence",
                },
                # Test 3: Round number consistency
                {
                    "update": {"current_round": -1},
                    "should_be_valid": False,
                    "constraint": "round_number_positive",
                },
                # Test 4: Valid update
                {
                    "update": {"current_round": 2},
                    "should_be_valid": True,
                    "constraint": "normal_progression",
                },
            ]

            for test in constraint_tests:
                # Reset to base state
                state_manager.update_state(base_state)

                # Apply test update
                state_manager.update_state(test["update"])
                current_state = state_manager.get_snapshot()

                # Validate
                issues = self.validator.validate_discussion_flow(current_state)

                if test["should_be_valid"]:
                    assert (
                        len(issues) == 0
                    ), f"Valid update failed constraint: {test['constraint']}"
                else:
                    # For invalid updates, we should either have validation issues
                    # or the update should have been rejected/corrected
                    state_is_invalid = len(issues) > 0
                    update_was_rejected = all(
                        current_state.get(key) != value
                        for key, value in test["update"].items()
                    )
                    assert (
                        state_is_invalid or update_was_rejected
                    ), f"Invalid update not caught: {test['constraint']}"

    @pytest.mark.integration
    def test_cross_field_validation(self):
        """Test validation of relationships between different state fields."""
        with patch_ui_components():
            flow = self.test_helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Test relationship: messages should match current topic
            state_with_mismatched_messages = {
                "current_phase": 2,
                "agenda": [
                    {"title": "AI Ethics", "status": "active"},
                    {"title": "Climate Change", "status": "pending"},
                ],
                "current_topic_index": 0,  # AI Ethics is active
                "messages": [
                    {
                        "id": str(uuid.uuid4()),
                        "agent_id": "agent_1",
                        "content": "Discussing climate change...",
                        "topic": "Climate Change",  # Mismatched!
                        "round_number": 1,
                        "timestamp": datetime.now(),
                        "message_type": "discussion",
                    }
                ],
            }
            state_manager.update_state(state_with_mismatched_messages)

            # Custom validation for cross-field relationships
            current_state = state_manager.get_snapshot()
            cross_field_issues = self._validate_cross_field_relationships(current_state)

            # Should detect topic mismatch
            assert len(cross_field_issues) > 0
            assert any("topic_mismatch" in issue for issue in cross_field_issues)

    def _validate_cross_field_relationships(self, state: VirtualAgoraState) -> list:
        """Validate relationships between different state fields."""
        issues = []

        # Check message-topic consistency
        if state.get("current_phase") == 2:  # Discussion phase
            agenda = state.get("agenda", [])
            current_topic_index = state.get("current_topic_index", 0)

            if current_topic_index < len(agenda):
                current_topic = agenda[current_topic_index]["title"]
                messages = state.get("messages", [])

                for message in messages:
                    if message.get("topic") != current_topic:
                        issues.append(
                            f"topic_mismatch: Message topic '{message.get('topic')}' "
                            f"doesn't match current topic '{current_topic}'"
                        )

        # Check round-message consistency
        current_round = state.get("current_round", 1)
        messages = state.get("messages", [])

        for message in messages:
            msg_round = message.get("round_number", 1)
            if msg_round > current_round:
                issues.append(
                    f"round_inconsistency: Message from future round {msg_round} "
                    f"when current round is {current_round}"
                )

        # Check speaking order consistency
        speaking_order = state.get("speaking_order", [])
        messages = state.get("messages", [])

        for message in messages:
            agent_id = message.get("agent_id")
            if agent_id and agent_id not in speaking_order:
                issues.append(
                    f"speaker_not_in_order: Agent {agent_id} not in speaking order"
                )

        return issues


@pytest.mark.integration
class TestStateMigrationFlow:
    """Test state migration and schema evolution."""

    def test_state_schema_migration(self):
        """Test migration between different state schema versions."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            # Simulate old state format
            old_state_format = {
                "session_id": str(uuid.uuid4()),
                "phase": 1,  # Old field name
                "topic": "Test Topic",  # Old field name
                "agent_list": ["agent_1", "agent_2"],  # Old field name
                "discussion_messages": [],  # Old field name
            }

            # Migrate to new format
            migrated_state = self._migrate_state_schema(old_state_format)

            # Verify migration
            assert "current_phase" in migrated_state
            assert migrated_state["current_phase"] == 1
            assert "main_topic" in migrated_state
            assert migrated_state["main_topic"] == "Test Topic"
            assert "speaking_order" in migrated_state
            assert migrated_state["speaking_order"] == ["agent_1", "agent_2"]
            assert "messages" in migrated_state
            assert migrated_state["messages"] == []

    def test_backward_compatibility(self):
        """Test backward compatibility with older state versions."""
        helper = IntegrationTestHelper(num_agents=2, scenario="default")

        with patch_ui_components():
            flow = helper.create_test_flow()
            state_manager = flow.get_state_manager()

            # Test with state missing newer fields
            legacy_state = {
                "current_phase": 1,
                "main_topic": "Legacy Topic",
                "speaking_order": ["agent_1", "agent_2"],
                # Missing: hitl_state, flow_control, metadata, etc.
            }

            # Should handle gracefully
            state_manager.update_state(legacy_state)
            current_state = state_manager.get_snapshot()

            # Should have filled in defaults for missing fields
            assert "hitl_state" in current_state
            assert "flow_control" in current_state
            assert isinstance(current_state["hitl_state"], dict)
            assert isinstance(current_state["flow_control"], dict)

    def _migrate_state_schema(self, old_state: dict) -> dict:
        """Migrate state from old schema to new schema."""
        migration_mapping = {
            "phase": "current_phase",
            "topic": "main_topic",
            "agent_list": "speaking_order",
            "discussion_messages": "messages",
        }

        new_state = {}

        # Apply field mappings
        for old_field, new_field in migration_mapping.items():
            if old_field in old_state:
                new_state[new_field] = old_state[old_field]

        # Copy unmapped fields
        for key, value in old_state.items():
            if key not in migration_mapping and key not in new_state:
                new_state[key] = value

        # Add default values for new required fields
        if "hitl_state" not in new_state:
            new_state["hitl_state"] = {"approved": False, "awaiting_approval": False}

        if "flow_control" not in new_state:
            new_state["flow_control"] = {
                "max_rounds_per_topic": 10,
                "auto_conclude_threshold": 3,
            }

        return new_state
