"""
Shared environment bootstrap for local development.
"""

from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    # Resolve relative to the repository root so entrypoints work from anywhere.
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path, override=False)


load_environment()
