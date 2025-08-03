"""Document context loading utility for Virtual Agora.

This module provides simple functionality to load text documents from a
directory and inject them as context for agents.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from langchain_text_splitters import CharacterTextSplitter

from virtual_agora.utils.logging import get_logger

logger = get_logger(__name__)


def load_context_documents(context_dir: str = "context") -> str:
    """Load all text files from context directory and return as single string.

    Args:
        context_dir: Directory containing context documents (default: "context")

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


def get_context_summary(context_dir: str = "context") -> Optional[str]:
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


def get_detailed_context_info(context_dir: str = "context") -> Dict[str, any]:
    """Get detailed information about context files including token counts.

    Args:
        context_dir: Directory containing context documents

    Returns:
        Dictionary containing file information and token counts
    """
    context_path = Path(context_dir)

    result = {"files": [], "total_files": 0, "total_tokens": 0, "status": "no_files"}

    # Check if directory exists
    if not context_path.exists() or not context_path.is_dir():
        logger.debug(f"Context directory not found: {context_path}")
        result["status"] = "directory_not_found"
        return result

    # Supported file extensions
    supported_extensions = {".txt", ".md"}

    # Find all supported files
    context_files: List[Path] = []
    for ext in supported_extensions:
        context_files.extend(context_path.glob(f"*{ext}"))

    if not context_files:
        logger.debug(f"No context files found in {context_path}")
        return result

    # Sort files by name for consistent ordering
    context_files.sort()

    # Initialize tokenizer for counting
    try:
        # Use tiktoken through langchain for consistency
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=1,  # We just want to count, not split
            chunk_overlap=0,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize tokenizer: {e}")
        # Fallback to character count approximation (1 token ≈ 4 chars)
        text_splitter = None

    # Process each file
    total_tokens = 0
    file_info = []

    for file_path in context_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                # Count tokens
                if text_splitter:
                    try:
                        # Check if text_splitter has a proper tokenizer method
                        if hasattr(text_splitter, "_tokenizer") and hasattr(
                            text_splitter._tokenizer, "encode"
                        ):
                            tokens = len(text_splitter._tokenizer.encode(content))
                        elif hasattr(text_splitter, "count_tokens"):
                            # Some text splitters have a count_tokens method
                            tokens = text_splitter.count_tokens(content)
                        else:
                            # Fallback to approximation if no tokenizer available
                            tokens = len(content) // 4
                    except Exception as e:
                        logger.debug(f"Token counting failed for {file_path.name}: {e}")
                        # Fallback to approximation
                        tokens = len(content) // 4
                else:
                    # Approximation if tokenizer not available
                    tokens = len(content) // 4

                file_info.append(
                    {
                        "name": file_path.name,
                        "tokens": tokens,
                        "size_kb": file_path.stat().st_size / 1024,
                    }
                )
                total_tokens += tokens
        except Exception as e:
            logger.warning(f"Failed to process context file {file_path}: {e}")
            continue

    result["files"] = file_info
    result["total_files"] = len(file_info)
    result["total_tokens"] = total_tokens
    result["status"] = "success"

    if file_info:
        logger.info(
            f"Analyzed {len(file_info)} context files with {total_tokens:,} total tokens"
        )

    return result
