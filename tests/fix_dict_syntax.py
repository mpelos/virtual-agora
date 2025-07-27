#!/usr/bin/env python3
"""Fix dictionary syntax in test files."""

import re
from pathlib import Path


def fix_dict_syntax(content):
    """Fix dictionary syntax issues."""

    # Fix dict key syntax: key=value -> "key": value
    # This regex looks for word=value patterns inside dictionaries
    def fix_dict_line(line):
        # Check if this looks like a dict entry (has = but not ==)
        if "=" in line and "==" not in line and ":=" not in line:
            # Check if it's inside a dict (has leading whitespace and ends with comma or no punctuation)
            if line.strip() and not line.strip().startswith(
                ("def ", "class ", "if ", "for ", "while ", "assert ", "return ")
            ):
                # Match word=value pattern
                match = re.match(r"^(\s+)(\w+)=(.+?)(?:,)?$", line)
                if match:
                    indent, key, value = match.groups()
                    return f'{indent}"{key}": {value},\n'
        return line

    lines = content.split("\n")
    fixed_lines = []
    in_dict = False

    for i, line in enumerate(lines):
        # Simple heuristic to detect if we're in a dict
        if "{" in line:
            in_dict = True
        elif "}" in line:
            in_dict = False

        if in_dict and "=" in line and "==" not in line:
            fixed_lines.append(fix_dict_line(line + "\n").rstrip("\n"))
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}...")

    with open(filepath, "r") as f:
        content = f.read()

    content = fix_dict_syntax(content)

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
