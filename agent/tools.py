"""
agent/tools.py — AutoStream Lead Agent Tools
Provides lead capture simulation and email validation utilities.
"""

from __future__ import annotations

import re


# Email validator

def validate_email(email: str) -> bool:
    """
    Validate an email address using a basic RFC-style regex.

    Args:
        email: The email string to validate.

    Returns:
        True if the email looks valid, False otherwise.
    """
    pattern = re.compile(
        r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    )
    return bool(pattern.match(email.strip()))


# Mock lead capture

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulate saving a lead to a CRM / database.

    Prints a success message to stdout and returns a confirmation string
    that can be appended to the conversation.

    Args:
        name:     Lead's full name.
        email:    Lead's email address.
        platform: Social platform / channel the lead came from.

    Returns:
        A confirmation string to surface to the user.
    """
    # Simulate CRM write
    print("\n" + "=" * 60)
    print("✅  LEAD CAPTURED — AutoStream CRM")
    print("=" * 60)
    print(f"  Name    : {name}")
    print(f"  Email   : {email}")
    print(f"  Platform: {platform}")
    print("=" * 60 + "\n")

    return (
        f"🎉 Thanks, {name}! Your details have been saved. "
        f"Our team will reach out to {email} shortly to walk you through "
        f"AutoStream and get you set up. Welcome aboard!"
    )
