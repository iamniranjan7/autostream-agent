"""
Mock lead capture tool for the AutoStream agent.

This simulates a backend API call that would store lead information
in a CRM or database in a production environment.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Capture a qualified lead's information.

    This mock function simulates sending lead data to a backend service.
    In production, this would call a CRM API (e.g., HubSpot, Salesforce).

    Args:
        name: The lead's full name.
        email: The lead's email address.
        platform: The creator platform (e.g., YouTube, Instagram, TikTok).

    Returns:
        A confirmation message string.
    """
    print(f"\n{'='*50}")
    print(f"🎯 LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"  Name:     {name}")
    print(f"  Email:    {email}")
    print(f"  Platform: {platform}")
    print(f"{'='*50}\n")

    return f"Lead captured successfully: {name}, {email}, {platform}"
