"""Pytest configuration and shared fixtures for Virtual Agora tests.

This module provides common test fixtures and configuration
that can be used across all test modules.
"""

import os
from pathlib import Path
from typing import Generator

import pytest
from _pytest.config import Config
from _pytest.fixtures import FixtureRequest


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary config file for testing."""
    config_file = tmp_path / "test_config.yml"
    config_content = """
moderator:
  provider: Google
  model: gemini-1.5-pro

agents:
  - provider: OpenAI
    model: gpt-4o
    count: 2
"""
    config_file.write_text(config_content)
    yield config_file


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment variables for testing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")
    monkeypatch.setenv("GROK_API_KEY", "test_grok_key")


@pytest.fixture(autouse=True)
def isolate_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests by changing to a temporary directory."""
    monkeypatch.chdir(tmp_path)


def pytest_configure(config: Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring real API keys"
    )