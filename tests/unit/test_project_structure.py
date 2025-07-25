"""Test project structure and basic imports."""

import importlib
import sys
from pathlib import Path

import pytest


class TestProjectStructure:
    """Test that the project structure is set up correctly."""
    
    def test_source_directory_exists(self):
        """Test that the source directory exists."""
        src_dir = Path(__file__).parent.parent.parent / "src"
        assert src_dir.exists()
        assert src_dir.is_dir()
        
    def test_main_package_importable(self):
        """Test that the main package can be imported."""
        import virtual_agora
        assert hasattr(virtual_agora, "__version__")
        assert hasattr(virtual_agora, "__author__")
        
    def test_submodules_exist(self):
        """Test that all submodules exist and can be imported."""
        submodules = [
            "virtual_agora.core",
            "virtual_agora.agents", 
            "virtual_agora.providers",
            "virtual_agora.config",
            "virtual_agora.ui",
            "virtual_agora.utils",
        ]
        
        for module_name in submodules:
            module = importlib.import_module(module_name)
            assert module is not None
            
    def test_main_entry_point_exists(self):
        """Test that the main entry point exists."""
        from virtual_agora.main import main
        assert callable(main)
        
    def test_utility_modules_importable(self):
        """Test that utility modules can be imported."""
        from virtual_agora.utils.logging import setup_logging, get_logger
        from virtual_agora.utils.exceptions import VirtualAgoraError
        
        assert callable(setup_logging)
        assert callable(get_logger)
        assert issubclass(VirtualAgoraError, Exception)
        
    def test_project_directories_exist(self):
        """Test that required project directories exist."""
        project_root = Path(__file__).parent.parent.parent
        
        required_dirs = [
            "src/virtual_agora",
            "tests",
            "docs",
            "logs",
            "reports",
            "examples",
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Directory {dir_path} does not exist"
            assert full_path.is_dir(), f"{dir_path} is not a directory"
            
    def test_configuration_files_exist(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent.parent
        
        config_files = [
            ".gitignore",
            ".env.example",
            "requirements.txt",
            "pyproject.toml",
            "README.md",
        ]
        
        for file_name in config_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Configuration file {file_name} does not exist"
            assert file_path.is_file(), f"{file_name} is not a file"