#!/usr/bin/env python3
"""Script to fix TypedDict attribute access in test files."""

import re
import os
from pathlib import Path


def fix_state_access(content):
    """Fix state attribute access patterns."""

    # Common state attributes that need to be converted
    state_attrs = [
        "agents",
        "agenda",
        "current_phase",
        "current_topic_index",
        "current_round",
        "speaking_order",
        "messages",
        "round_summaries",
        "topic_summaries",
        "voting_rounds",
        "hitl_state",
        "session_id",
        "current_speaker_index",
        "metadata",
        "discussion_rounds",
        "main_topic",
        "active_topic",
        "vote_history",
        "votes",
        "active_vote",
        "current_topic",
        "last_topic_summary",
    ]

    # Replace state.attr with state["attr"]
    for attr in state_attrs:
        # Handle basic access
        pattern = rf"(\w+)\.{attr}\b"
        replacement = rf'\1["{attr}"]'
        content = re.sub(pattern, replacement, content)

        # Handle nested access like state.hitl_state.approved
        if attr == "hitl_state":
            content = re.sub(
                r'(\w+)\["hitl_state"\]\.approved',
                r'\1["hitl_state"]["approved"]',
                content,
            )
            content = re.sub(
                r'(\w+)\["hitl_state"\]\.pending_approval',
                r'\1["hitl_state"]["pending_approval"]',
                content,
            )

        # Handle agenda items access like state.agenda[0].title
        if attr == "agenda":
            content = re.sub(
                r'(\w+)\["agenda"\]\[(\d+)\]\.title',
                r'\1["agenda"][\2]["title"]',
                content,
            )
            content = re.sub(
                r'(\w+)\["agenda"\]\[(\d+)\]\.status',
                r'\1["agenda"][\2]["status"]',
                content,
            )
            content = re.sub(
                r'(\w+)\["agenda"\]\[(\d+)\]\.description',
                r'\1["agenda"][\2]["description"]',
                content,
            )

    # Fix loop variables with attributes
    content = re.sub(
        r'for topic in (\w+)\["agenda"\]:\s*topic\.status',
        r'for topic in \1["agenda"]:\n            topic["status"]',
        content,
    )
    content = re.sub(r"topic\.status", r'topic["status"]', content)
    content = re.sub(r"topic\.title", r'topic["title"]', content)

    # Fix message attributes
    content = re.sub(r"msg\.agent_id", r'msg["agent_id"]', content)
    content = re.sub(r"msg\.content", r'msg["content"]', content)
    content = re.sub(r"msg\.round_number", r'msg["round_number"]', content)

    # Fix vote round attributes
    content = re.sub(r"vr\.vote_type", r'vr["vote_type"]', content)
    content = re.sub(r'vr\.get\("vote_type"\)', r'vr.get("vote_type")', content)

    return content


def fix_typeddict_constructors(content):
    """Fix TypedDict constructor calls."""

    # Convert TopicInfo(...) to {...}
    content = re.sub(
        r"TopicInfo\(([\s\S]*?)\)", lambda m: "{" + m.group(1) + "}", content
    )

    # Convert Message(...) to {...}
    def replace_message(match):
        args = match.group(1)
        # Add id field if not present
        if '"id"' not in args and "'id'" not in args:
            args = f'"id": str(uuid.uuid4()),\n        {args}'
        return "{" + args + "}"

    content = re.sub(r"Message\(([\s\S]*?)\)", replace_message, content)

    # Convert VoteRound(...) to {...}
    content = re.sub(
        r"VoteRound\(([\s\S]*?)\)", lambda m: "{" + m.group(1) + "}", content
    )

    # Convert RoundInfo(...) to {...}
    content = re.sub(
        r"RoundInfo\(([\s\S]*?)\)", lambda m: "{" + m.group(1) + "}", content
    )

    # Convert AgentInfo(...) to {...}
    content = re.sub(
        r"AgentInfo\(([\s\S]*?)\)", lambda m: "{" + m.group(1) + "}", content
    )

    return content


def fix_phase_values(content):
    """Fix phase values to use integers."""

    # Map phase names to integers
    phase_map = {
        '"initialization"': "0",
        '"agenda_setting"': "1",
        '"discussion"': "2",
        '"topic_conclusion"': "3",
        '"agenda_reevaluation"': "4",
        '"final_report"': "5",
        '"completed"': "5",  # Treat completed as final_report
    }

    # Replace in comparisons like state["current_phase"] == "discussion"
    for phase_str, phase_int in phase_map.items():
        # Escape special regex characters in phase_str
        escaped_phase_str = re.escape(phase_str)
        content = re.sub(
            rf'(\["current_phase"\]\s*==\s*){escaped_phase_str}',
            rf"\g<1>{phase_int}",
            content,
        )
        content = re.sub(
            rf'(\["current_phase"\]\s*=\s*){escaped_phase_str}',
            rf"\g<1>{phase_int}",
            content,
        )

    return content


def process_file(filepath):
    """Process a single test file."""
    print(f"Processing {filepath}...")

    with open(filepath, "r") as f:
        content = f.read()

    # Apply fixes
    content = fix_state_access(content)
    content = fix_typeddict_constructors(content)
    content = fix_phase_values(content)

    # Write back
    with open(filepath, "w") as f:
        f.write(content)

    print(f"  Fixed {filepath}")


def main():
    """Fix all test files."""
    test_dir = Path(__file__).parent / "integration"

    test_files = [
        "test_agenda_flows.py",
        "test_conclusion_flows.py",
        # Don't process files we already manually fixed
        # "test_complete_flow.py",
        # "test_discussion_flows.py"
    ]

    for filename in test_files:
        filepath = test_dir / filename
        if filepath.exists():
            process_file(filepath)


if __name__ == "__main__":
    main()
