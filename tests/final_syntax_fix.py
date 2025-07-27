#!/usr/bin/env python3
"""Final syntax fixes for test files."""

import re
from pathlib import Path


def fix_syntax_errors(content):
    """Fix remaining syntax errors."""

    # Fix double commas in dictionary entries
    content = re.sub(r"\},,", "},", content)
    content = re.sub(r"\),,$", "),", content, flags=re.MULTILINE)

    # Fix datetime.now(}, -> datetime.now(),
    content = re.sub(r"datetime\.now\(\},", "datetime.now(),", content)

    # Fix str(uuid.uuid4(}), -> str(uuid.uuid4()),
    content = re.sub(r"str\(uuid\.uuid4\(\}\)", "str(uuid.uuid4())", content)

    # Fix multiple commas in dict entries
    content = re.sub(r'",,', '",', content)
    content = re.sub(r",,", ",", content)

    # Fix IntegrationTestHelper initialization
    content = re.sub(
        r'IntegrationTestHelper\(num_agents=(\d+), "scenario": "([\w_]+)"\),',
        r'IntegrationTestHelper(num_agents=\1, scenario="\2")',
        content,
    )

    # Fix helper.IntegrationTestHelper patterns
    content = re.sub(
        r'helper = IntegrationTestHelper\(num_agents=(\d+), "scenario": "([\w_]+)"\),',
        r'helper = IntegrationTestHelper(num_agents=\1, scenario="\2")',
        content,
    )

    # Fix line continuation in f-strings
    content = re.sub(r'"\s*"\s*\\', r'" \\', content)

    # Fix broken f-string continuations
    content = re.sub(r'f"([^"]+)"\s*"\s*\\[\s\n]+f"([^"]+)"', r'f"\1 \2"', content)

    # Fix missing closing braces
    content = re.sub(r'"status": "pending"\)', r'"status": "pending"}', content)
    content = re.sub(r'"status": "completed"\)', r'"status": "completed"}', content)

    # Fix self.test_helper assignments with wrong scenario syntax
    content = re.sub(
        r'self\.test_helper = IntegrationTestHelper\(num_agents=(\d+), "scenario": "([\w_]+)"\),',
        r'self.test_helper = IntegrationTestHelper(num_agents=\1, scenario="\2")',
        content,
    )

    # Fix voting_round.result access
    content = re.sub(r"voting_round\.result", r'voting_round["result"]', content)

    # Fix test_helper
    content = re.sub(
        r'IntegrationTestHelper\(num_agents=4, "scenario": "default"\),',
        r'IntegrationTestHelper(num_agents=4, scenario="default")',
        content,
    )

    content = re.sub(
        r'IntegrationTestHelper\(num_agents=3, "scenario": "minority_dissent"\),',
        r'IntegrationTestHelper(num_agents=3, scenario="minority_dissent")',
        content,
    )

    # Fix specific test patterns
    content = re.sub(
        r'helper = IntegrationTestHelper\(num_agents=3, "scenario": "quick_consensus"\),',
        r'helper = IntegrationTestHelper(num_agents=3, scenario="quick_consensus")',
        content,
    )

    content = re.sub(
        r'helper = IntegrationTestHelper\(num_agents=3, "scenario": "extended_debate"\),',
        r'helper = IntegrationTestHelper(num_agents=3, scenario="extended_debate")',
        content,
    )

    content = re.sub(
        r'helper = IntegrationTestHelper\(num_agents=3, "scenario": "default"\),',
        r'helper = IntegrationTestHelper(num_agents=3, scenario="default")',
        content,
    )

    return content


def process_file(filepath):
    """Process a single file."""
    print(f"Processing {filepath}...")

    with open(filepath, "r") as f:
        content = f.read()

    content = fix_syntax_errors(content)

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
