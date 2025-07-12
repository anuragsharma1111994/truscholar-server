"""API routers for TruScholar application.

This module provides all FastAPI routers that define the API endpoints
for different functional areas of the application.
"""

from src.routers.auth import router as auth_router
from src.routers.users import router as users_router

# Version info
__version__ = "1.0.0"

# Export all routers
__all__ = [
    "auth_router",
    "users_router",
]
