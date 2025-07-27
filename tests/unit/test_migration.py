"""Tests for v1.1 to v1.3 migration utilities."""

import pytest
from datetime import datetime

from virtual_agora.config.migration import (
    migrate_config_v1_to_v3,
    detect_config_version,
    is_migration_needed,
    validate_migrated_config,
    migrate_config_with_validation,
    get_default_model,
)
from virtual_agora.state.migration import (
    migrate_state_v1_to_v3,
    detect_state_version,
    is_migration_needed as is_state_migration_needed,
    validate_migrated_state,
    migrate_state_with_validation,
    create_initial_v13_state,
)


class TestConfigMigration:
    """Test configuration migration from v1.1 to v1.3."""
    
    def test_get_default_model(self):
        """Test default model selection for different providers."""
        # Google defaults
        assert get_default_model('Google', 'summarizer') == 'gemini-2.5-flash-lite'
        assert get_default_model('Google', 'topic_report') == 'gemini-2.5-pro'
        assert get_default_model('Google', 'ecclesia_report') == 'gemini-2.5-pro'
        
        # OpenAI defaults
        assert get_default_model('OpenAI', 'summarizer') == 'gpt-4o'
        assert get_default_model('OpenAI', 'topic_report') == 'gpt-4o'
        
        # Case insensitive
        assert get_default_model('google', 'summarizer') == 'gemini-2.5-flash-lite'
        assert get_default_model('OPENAI', 'summarizer') == 'gpt-4o'
    
    def test_detect_config_version(self):
        """Test configuration version detection."""
        # v1.1 config
        v11_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert detect_config_version(v11_config) == '1.1'
        
        # v1.3 config
        v13_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'summarizer': {'provider': 'OpenAI', 'model': 'gpt-4o'},
            'topic_report': {'provider': 'Anthropic', 'model': 'claude-3'},
            'ecclesia_report': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert detect_config_version(v13_config) == '1.3'
        
        # Unknown config
        unknown_config = {'some_field': 'value'}
        assert detect_config_version(unknown_config) == 'unknown'
    
    def test_is_migration_needed(self):
        """Test migration need detection."""
        v11_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert is_migration_needed(v11_config) is True
        
        v13_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'summarizer': {'provider': 'OpenAI', 'model': 'gpt-4o'},
            'topic_report': {'provider': 'Anthropic', 'model': 'claude-3'},
            'ecclesia_report': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert is_migration_needed(v13_config) is False
    
    def test_migrate_config_v1_to_v3(self):
        """Test basic config migration."""
        old_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [
                {'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2},
                {'provider': 'Anthropic', 'model': 'claude-3', 'count': 1}
            ]
        }
        
        new_config = migrate_config_v1_to_v3(old_config)
        
        # Check all required fields exist
        assert 'moderator' in new_config
        assert 'summarizer' in new_config
        assert 'topic_report' in new_config
        assert 'ecclesia_report' in new_config
        assert 'agents' in new_config
        
        # Check defaults were applied correctly
        assert new_config['summarizer']['provider'] == 'Google'
        assert new_config['summarizer']['model'] == 'gemini-2.5-flash-lite'
        
        # Topic report should prefer Anthropic if available
        assert new_config['topic_report']['provider'] == 'Anthropic'
        assert new_config['topic_report']['model'] == 'claude-3-opus-20240229'
        
        # Ecclesia report should use moderator's provider
        assert new_config['ecclesia_report']['provider'] == 'Google'
        assert new_config['ecclesia_report']['model'] == 'gemini-2.5-pro'
        
        # Original data should be preserved
        assert new_config['moderator'] == old_config['moderator']
        assert new_config['agents'] == old_config['agents']
    
    def test_validate_migrated_config(self):
        """Test validation of migrated configs."""
        # Valid v1.3 config
        valid_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'summarizer': {'provider': 'OpenAI', 'model': 'gpt-4o'},
            'topic_report': {'provider': 'Anthropic', 'model': 'claude-3'},
            'ecclesia_report': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert validate_migrated_config(valid_config) is True
        
        # Missing required field
        invalid_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'summarizer': {'provider': 'OpenAI', 'model': 'gpt-4o'},
            # missing topic_report
            'ecclesia_report': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert validate_migrated_config(invalid_config) is False
        
        # Invalid agent config (missing model)
        invalid_agent_config = {
            'moderator': {'provider': 'Google'},  # missing model
            'summarizer': {'provider': 'OpenAI', 'model': 'gpt-4o'},
            'topic_report': {'provider': 'Anthropic', 'model': 'claude-3'},
            'ecclesia_report': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        assert validate_migrated_config(invalid_agent_config) is False
    
    def test_migrate_config_with_validation(self):
        """Test full migration with validation."""
        # v1.1 config
        old_config = {
            'moderator': {'provider': 'Google', 'model': 'gemini-2.5-pro'},
            'agents': [{'provider': 'OpenAI', 'model': 'gpt-4o', 'count': 2}]
        }
        
        # Should migrate successfully
        new_config = migrate_config_with_validation(old_config)
        assert 'summarizer' in new_config
        assert 'topic_report' in new_config
        assert 'ecclesia_report' in new_config
        
        # v1.3 config should be returned as-is
        result = migrate_config_with_validation(new_config)
        assert result == new_config
        
        # Invalid config should raise error
        invalid_config = {'invalid': 'config'}
        with pytest.raises(ValueError):
            migrate_config_with_validation(invalid_config)


class TestStateMigration:
    """Test state migration from v1.1 to v1.3."""
    
    def test_detect_state_version(self):
        """Test state version detection."""
        # v1.1 state
        v11_state = {
            'session_id': 'test-123',
            'current_phase': 1,
            'agents': {},
            'messages': []
        }
        assert detect_state_version(v11_state) == '1.1'
        
        # v1.3 state
        v13_state = {
            'session_id': 'test-123',
            'current_phase': 1,
            'agents': {},
            'messages': [],
            'specialized_agents': {},
            'agent_invocations': [],
            'round_summaries': [],
            'periodic_stop_counter': 0,
            'user_stop_history': []
        }
        assert detect_state_version(v13_state) == '1.3'
        
        # Unknown state
        unknown_state = {'some_field': 'value'}
        assert detect_state_version(unknown_state) == 'unknown'
    
    def test_migrate_state_v1_to_v3(self):
        """Test basic state migration."""
        old_state = {
            'session_id': 'test-123',
            'current_phase': 2,
            'agents': {'agent1': {'id': 'agent1', 'model': 'gpt-4o'}},
            'messages': [{'content': 'Hello', 'speaker_id': 'agent1'}],
            'hitl_state': {
                'awaiting_approval': False,
                'approval_type': None,
                'prompt_message': None,
                'options': None,
                'approval_history': []
            }
        }
        
        new_state = migrate_state_v1_to_v3(old_state)
        
        # Check new v1.3 fields exist
        assert 'specialized_agents' in new_state
        assert 'agent_invocations' in new_state
        assert 'round_summaries' in new_state
        assert 'agent_contexts' in new_state
        assert 'periodic_stop_counter' in new_state
        assert 'user_stop_history' in new_state
        
        # Check HITL state enhancements
        assert 'last_periodic_stop_round' in new_state['hitl_state']
        assert 'periodic_stop_responses' in new_state['hitl_state']
        
        # Original data should be preserved
        assert new_state['session_id'] == old_state['session_id']
        assert new_state['current_phase'] == old_state['current_phase']
        assert new_state['agents'] == old_state['agents']
        assert new_state['messages'] == old_state['messages']
    
    def test_validate_migrated_state(self):
        """Test validation of migrated state."""
        # Valid v1.3 state
        valid_state = {
            'specialized_agents': {},
            'agent_invocations': [],
            'round_summaries': [],
            'agent_contexts': [],
            'periodic_stop_counter': 0,
            'user_stop_history': [],
            'hitl_state': {
                'last_periodic_stop_round': None,
                'periodic_stop_responses': []
            }
        }
        assert validate_migrated_state(valid_state) is True
        
        # Missing required field
        invalid_state = {
            'specialized_agents': {},
            'agent_invocations': [],
            # missing round_summaries
            'agent_contexts': [],
            'periodic_stop_counter': 0,
            'user_stop_history': []
        }
        assert validate_migrated_state(invalid_state) is False
        
        # Wrong type for field
        invalid_type_state = {
            'specialized_agents': [],  # Should be dict
            'agent_invocations': [],
            'round_summaries': [],
            'agent_contexts': [],
            'periodic_stop_counter': 0,
            'user_stop_history': []
        }
        assert validate_migrated_state(invalid_type_state) is False
    
    def test_create_initial_v13_state(self):
        """Test creation of fresh v1.3 state."""
        state = create_initial_v13_state()
        
        # Check all required fields exist
        assert 'specialized_agents' in state
        assert 'agent_invocations' in state
        assert 'round_summaries' in state
        assert 'agent_contexts' in state
        assert 'periodic_stop_counter' in state
        assert 'user_stop_history' in state
        assert 'hitl_state' in state
        
        # Check HITL state structure
        assert 'last_periodic_stop_round' in state['hitl_state']
        assert 'periodic_stop_responses' in state['hitl_state']
        
        # Validate the created state
        assert validate_migrated_state(state) is True
    
    def test_migrate_state_with_validation(self):
        """Test full state migration with validation."""
        # v1.1 state
        old_state = {
            'session_id': 'test-123',
            'current_phase': 1,
            'agents': {},
            'messages': []
        }
        
        # Should migrate successfully
        new_state = migrate_state_with_validation(old_state)
        assert 'specialized_agents' in new_state
        assert 'periodic_stop_counter' in new_state
        
        # v1.3 state should be returned as-is
        result = migrate_state_with_validation(new_state)
        assert result == new_state
        
        # Invalid state should raise error
        invalid_state = {'invalid': 'state'}
        with pytest.raises(ValueError):
            migrate_state_with_validation(invalid_state)