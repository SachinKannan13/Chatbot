"""
Uvicorn launcher for the Employee Survey Insights Chatbot.

Usage:
    python run.py
"""
import uvicorn
from src.config.settings import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1,  # Single worker — caches are in-memory and not shared
    )


if __name__ == "__main__":
    main()
