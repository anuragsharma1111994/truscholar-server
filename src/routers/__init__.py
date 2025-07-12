"""API routers for TruScholar application.

This module provides all FastAPI routers that define the API endpoints
for different functional areas of the application.
"""

from src.routers.auth import router as auth_router
from src.routers.users import router as users_router
from src.routers.careers import router as careers_router
from src.routers.questions import router as questions_router
from src.routers.tests import router as tests_router
from src.routers.health import router as health_router

# Version info
__version__ = "1.0.0"

# Export all routers
__all__ = [
    "auth_router",
    "users_router", 
    "careers_router",
    "questions_router",
    "tests_router",
    "health_router",
]

# For convenience, also export individual router modules
auth = auth_router
users = users_router
careers = careers_router
questions = questions_router
tests = tests_router
health = health_router
