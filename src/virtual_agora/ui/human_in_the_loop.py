from typing import List, Dict, Any


def get_initial_topic() -> str:
    """Gets the initial discussion topic from the user."""
    print("Please enter the topic you would like the agents to discuss:")
    topic = input()
    while not topic.strip():
        print("Topic cannot be empty. Please enter a topic:")
        topic = input()
    return topic


def get_agenda_approval(agenda: List[str]) -> List[str]:
    """
    Displays the proposed agenda and asks for user approval.
    """
    print("Proposed Agenda:")
    for i, item in enumerate(agenda):
        print(f"{i+1}. {item}")

    action = input("Do you want to (a)pprove, (e)dit, or (r)eject? ").lower()

    if action == "a":
        return agenda
    elif action == "e":
        return edit_agenda(agenda)
    else:
        return []


def edit_agenda(agenda: List[str]) -> List[str]:
    """
    Allows the user to interactively edit the agenda.
    """
    # This is a simplified implementation. A more robust solution would be added later.
    print("Agenda editing is not fully implemented yet. Returning original agenda.")
    return agenda


def get_continuation_approval(completed_topic: str, remaining_topics: List[str]) -> str:
    """
    Asks the user if they want to continue to the next topic.
    """
    print(f"Topic '{completed_topic}' is now concluded.")
    if remaining_topics:
        print("Remaining topics:", ", ".join(remaining_topics))
        action = input("Continue to the next topic? (y/n/m)odify agenda: ").lower()
    else:
        action = input("No more topics. End session? (y/n): ").lower()

    return action


def get_agenda_modifications(agenda: List[str]) -> List[str]:
    """
    Allows the user to modify the agenda.
    """
    # This is a simplified implementation.
    print("Current agenda:", ", ".join(agenda))
    new_agenda_str = input("Enter the new agenda, with items separated by commas: ")
    return [item.strip() for item in new_agenda_str.split(",")]


def display_session_status(status: Dict[str, Any]):
    """
    Displays the current session status.
    """
    print(
        """
--- Session Status ---"""
    )
    for key, value in status.items():
        print(f"{key}: {value}")
    print(
        """----------------------
"""
    )


def handle_emergency_interrupt():
    """

    Handles emergency interrupts from the user.
    """
    # This will be implemented in a future step.
    print("Emergency interrupt requested. Shutting down...")
    exit()
