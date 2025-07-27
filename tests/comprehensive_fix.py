#!/usr/bin/env python3
"""Comprehensive fix for test files."""

import re
from pathlib import Path


def comprehensive_fix(content):
    """Apply comprehensive fixes."""

    # Fix double commas
    content = re.sub(r",\s*,", ",", content)

    # Fix missing quotes in dictionaries
    content = re.sub(r"(\s+)(\w+)=([^=\n]+)(?=\n)", r'\1"\2": \3,', content)

    # Fix timestamp=datetime.now(} -> "timestamp": datetime.now()
    content = re.sub(
        r"timestamp=datetime\.now\(}", '"timestamp": datetime.now()},', content
    )

    # Fix round_number= -> "round_number":
    content = re.sub(r"round_number=(\d+),", r'"round_number": \1,', content)

    # Fix topic= -> "topic":
    content = re.sub(r'topic="([^"]+)",', r'"topic": "\1",', content)

    # Fix message_type= -> "message_type":
    content = re.sub(r'message_type="([^"]+)"', r'"message_type": "\1"', content)

    # Fix round_id=str(uuid.uuid4(}) -> "round_id": str(uuid.uuid4())
    content = re.sub(
        r"round_id=str\(uuid\.uuid4\(}\)", '"round_id": str(uuid.uuid4())', content
    )

    # Fix extra commas before closing braces
    content = re.sub(r",\s*}", "}", content)
    content = re.sub(r",\s*\)", ")", content)

    # Fix f"agent_{(i % 3} + 1}" -> f"agent_{(i % 3) + 1}"
    content = re.sub(r'f"agent_\{\(i % 3} \+ 1\}"', r'f"agent_{(i % 3) + 1}"', content)

    # Fix line continuations with backslash in f-strings
    content = re.sub(r"\\,\n", r'" \\\n                           f"', content)

    # Fix comments after values in dicts
    content = re.sub(r'", # ([^,\n]+),', r'",  # \1', content)
    content = re.sub(r'", # ([^,\n]+)$', r'"  # \1', content, flags=re.MULTILINE)

    return content


def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}...")

    with open(filepath, "r") as f:
        content = f.read()

    content = comprehensive_fix(content)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"  Fixed {filepath}")


def main():
    """Fix test files."""
    test_files = [
        "/Users/mpelos/workspace/personal/virtual-agora/tests/integration/test_agenda_flows.py",
        "/Users/mpelos/workspace/personal/virtual-agora/tests/integration/test_conclusion_flows.py",
    ]

    for filepath in test_files:
        if Path(filepath).exists():
            process_file(filepath)


if __name__ == "__main__":
    main()
