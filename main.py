"""Main entry point for the TruCareer application.

This module provides the main entry point for running the FastAPI application
directly or through various deployment methods.
"""

import uvicorn
from src.api.main import app
from src.core.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
        log_config=None,  # Use our custom logging
        access_log=False,  # Handled by middleware
        workers=1 if settings.APP_ENV == "development" else 4,
    )