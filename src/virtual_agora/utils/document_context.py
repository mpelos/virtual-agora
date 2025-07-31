"""Document context loading utility for Virtual Agora.

This module provides simple functionality to load text documents from a
directory and inject them as context for agents.
"""

import os
from pathlib import Path
from typing import List, Optional

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


def load_context_documents(context_dir: str = "context_docs") -> str:
    """Load all text files from context directory and return as single string.

    Args:
        context_dir: Directory containing context documents (default: "context_docs")

    Returns:
        Concatenated content of all documents, or empty string if no files found
    """
    context_path = Path(context_dir)

    # Check if directory exists
    if not context_path.exists() or not context_path.is_dir():
        logger.debug(f"Context directory not found: {context_path}")
        return ""

    # Supported file extensions
    supported_extensions = {".txt", ".md"}

    # Find all supported files
    context_files: List[Path] = []
    for ext in supported_extensions:
        context_files.extend(context_path.glob(f"*{ext}"))

    if not context_files:
        logger.debug(f"No context files found in {context_path}")
        return ""

    # Sort files by name for consistent ordering
    context_files.sort()

    # Load and concatenate content
    content_parts = []
    loaded_files = []

    for file_path in context_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
                if file_content:  # Only add non-empty files
                    content_parts.append(f"=== {file_path.name} ===\n{file_content}")
                    loaded_files.append(file_path.name)
        except Exception as e:
            logger.warning(f"Failed to load context file {file_path}: {e}")
            continue

    if content_parts:
        logger.info(
            f"Loaded context from {len(loaded_files)} files: {', '.join(loaded_files)}"
        )
        return "\n\n".join(content_parts) + "\n\n" + "=" * 50 + "\n"

    return ""


def get_context_summary(context_dir: str = "context_docs") -> Optional[str]:
    """Get a summary of available context files.

    Args:
        context_dir: Directory containing context documents

    Returns:
        Brief summary of context files, or None if no files found
    """
    context_path = Path(context_dir)

    if not context_path.exists() or not context_path.is_dir():
        return None

    supported_extensions = {".txt", ".md"}
    context_files = []
    for ext in supported_extensions:
        context_files.extend(context_path.glob(f"*{ext}"))

    if not context_files:
        return None

    return f"Context files available: {len(context_files)} files ({', '.join(f.name for f in sorted(context_files))})"
