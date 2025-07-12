"""FastAPI dependency injection for TruCareer system.

This module provides dependency injection for authentication, services,
database connections, and other core components used across the application.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
import redis

from src.core.config import get_settings
from src.services.career_service import CareerService
from src.services.test_service import TestService
from src.services.user_service import UserService
from src.services.question_service import QuestionService
from src.services.scoring_service import ScoringService
from src.services.notification_service import NotificationService, create_notification_service
from src.langchain_handlers.career_recommender import CareerRecommender
from src.database.mongodb import get_database
from src.database.redis_client import get_redis_client
from src.cache.cache_manager import CacheManager
from src.llm.llm_factory import LLMFactory
from src.utils.logger import get_logger
from src.utils.exceptions import AuthenticationError

settings = get_settings()
logger = get_logger(__name__)

# Security
security = HTTPBearer()

# Global service instances (singleton pattern)
_career_service: Optional[CareerService] = None
_test_service: Optional[TestService] = None
_user_service: Optional[UserService] = None
_question_service: Optional[QuestionService] = None
_scoring_service: Optional[ScoringService] = None
_notification_service: Optional[NotificationService] = None
_career_recommender: Optional[CareerRecommender] = None
_cache_manager: Optional[CacheManager] = None


# Authentication Dependencies

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Extract and validate current user from JWT token.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Token missing user ID")
        
        # Extract user information from token
        user_data = {
            "id": user_id,
            "email": payload.get("email"),
            "name": payload.get("name"),
            "role": payload.get("role", "user"),
            "permissions": payload.get("permissions", []),
            "exp": payload.get("exp")
        }
        
        # Check token expiration
        if user_data["exp"] and datetime.utcnow().timestamp() > user_data["exp"]:
            raise AuthenticationError("Token has expired")
        
        # Log successful authentication
        logger.info(f"Authenticated user: {user_id}")
        
        return user_data
        
    except JWTError as e:
        logger.warning(f"JWT decode error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        logger.warning(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Extract user from token if present, otherwise return None.
    Used for endpoints that work with or without authentication.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def require_permissions(required_permissions: list):
    """
    Dependency factory for requiring specific permissions.
    
    Args:
        required_permissions: List of required permission strings
        
    Returns:
        Dependency function that validates permissions
    """
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_permissions = current_user.get("permissions", [])
        
        # Check if user has all required permissions
        missing_permissions = set(required_permissions) - set(user_permissions)
        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {list(missing_permissions)}"
            )
        
        return current_user
    
    return permission_checker


def require_role(required_role: str):
    """
    Dependency factory for requiring specific role.
    
    Args:
        required_role: Required role string
        
    Returns:
        Dependency function that validates role
    """
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role")
        
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}, current role: {user_role}"
            )
        
        return current_user
    
    return role_checker


# Database Dependencies

async def get_mongodb():
    """Get MongoDB database connection."""
    try:
        db = await get_database()
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )


async def get_redis():
    """Get Redis client connection."""
    try:
        redis_client = await get_redis_client()
        return redis_client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache connection failed"
        )


# Service Dependencies

async def get_cache_manager() -> CacheManager:
    """Get cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        try:
            redis_client = await get_redis()
            _cache_manager = CacheManager(redis_client)
        except Exception as e:
            logger.warning(f"Failed to initialize cache manager: {str(e)}")
            # Fallback to in-memory cache
            _cache_manager = CacheManager(None)
    
    return _cache_manager


async def get_career_recommender() -> CareerRecommender:
    """Get career recommender instance."""
    global _career_recommender
    
    if _career_recommender is None:
        try:
            llm = LLMFactory.create_from_settings()
            _career_recommender = CareerRecommender(
                llm=llm,
                enable_caching=True,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to initialize career recommender: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Career recommender service unavailable"
            )
    
    return _career_recommender


async def get_career_service() -> CareerService:
    """Get career service instance."""
    global _career_service
    
    if _career_service is None:
        try:
            career_recommender = await get_career_recommender()
            test_service = await get_test_service()
            cache_manager = await get_cache_manager()
            
            _career_service = CareerService(
                career_recommender=career_recommender,
                test_service=test_service,
                cache_client=cache_manager
            )
        except Exception as e:
            logger.error(f"Failed to initialize career service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Career service unavailable"
            )
    
    return _career_service


async def get_test_service() -> TestService:
    """Get test service instance."""
    global _test_service
    
    if _test_service is None:
        try:
            db = await get_mongodb()
            scoring_service = await get_scoring_service()
            cache_manager = await get_cache_manager()
            
            _test_service = TestService(
                db=db,
                scoring_service=scoring_service,
                cache_client=cache_manager
            )
        except Exception as e:
            logger.error(f"Failed to initialize test service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Test service unavailable"
            )
    
    return _test_service


async def get_user_service() -> UserService:
    """Get user service instance."""
    global _user_service
    
    if _user_service is None:
        try:
            db = await get_mongodb()
            cache_manager = await get_cache_manager()
            
            _user_service = UserService(
                db=db,
                cache_client=cache_manager
            )
        except Exception as e:
            logger.error(f"Failed to initialize user service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="User service unavailable"
            )
    
    return _user_service


async def get_question_service() -> QuestionService:
    """Get question service instance."""
    global _question_service
    
    if _question_service is None:
        try:
            db = await get_mongodb()
            llm = LLMFactory.create_from_settings()
            cache_manager = await get_cache_manager()
            
            _question_service = QuestionService(
                db=db,
                llm=llm,
                cache_client=cache_manager
            )
        except Exception as e:
            logger.error(f"Failed to initialize question service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Question service unavailable"
            )
    
    return _question_service


async def get_scoring_service() -> ScoringService:
    """Get scoring service instance."""
    global _scoring_service
    
    if _scoring_service is None:
        try:
            _scoring_service = ScoringService()
        except Exception as e:
            logger.error(f"Failed to initialize scoring service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scoring service unavailable"
            )
    
    return _scoring_service


async def get_notification_service() -> NotificationService:
    """Get notification service instance."""
    global _notification_service
    
    if _notification_service is None:
        try:
            _notification_service = create_notification_service()
        except Exception as e:
            logger.error(f"Failed to initialize notification service: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Notification service unavailable"
            )
    
    return _notification_service


# Utility Dependencies

def get_pagination_params(
    page: int = 1,
    page_size: int = 20,
    max_page_size: int = 100
) -> Dict[str, int]:
    """
    Get pagination parameters with validation.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Dict with validated pagination parameters
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be >= 1"
        )
    
    if page_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be >= 1"
        )
    
    if page_size > max_page_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Page size must be <= {max_page_size}"
        )
    
    offset = (page - 1) * page_size
    
    return {
        "page": page,
        "page_size": page_size,
        "offset": offset,
        "limit": page_size
    }


def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Extract request context information.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dict containing request context
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_host": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "request_id": request.headers.get("x-request-id"),
        "timestamp": datetime.utcnow().isoformat()
    }


# Health Check Dependencies

async def check_database_health() -> Dict[str, Any]:
    """Check database connection health."""
    try:
        db = await get_mongodb()
        # Simple health check query
        await db.admin.command('ping')
        return {"mongodb": "healthy"}
    except Exception as e:
        return {"mongodb": "unhealthy", "error": str(e)}


async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health."""
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        return {"redis": "healthy"}
    except Exception as e:
        return {"redis": "unhealthy", "error": str(e)}


async def check_llm_health() -> Dict[str, Any]:
    """Check LLM service health."""
    try:
        llm = LLMFactory.create_from_settings()
        # Simple health check (could be expanded)
        return {"llm": "healthy", "provider": llm.provider}
    except Exception as e:
        return {"llm": "unhealthy", "error": str(e)}


# Cleanup Dependencies

async def cleanup_services():
    """Cleanup service instances and connections."""
    global _career_service, _test_service, _user_service
    global _question_service, _scoring_service, _notification_service
    global _career_recommender, _cache_manager
    
    logger.info("Cleaning up service instances...")
    
    # Reset service instances
    _career_service = None
    _test_service = None
    _user_service = None
    _question_service = None
    _scoring_service = None
    _notification_service = None
    _career_recommender = None
    _cache_manager = None
    
    logger.info("Service cleanup completed")


# Development and Testing Dependencies

def get_test_user() -> Dict[str, Any]:
    """Get test user for development/testing."""
    return {
        "id": "test_user_123",
        "email": "test@trucareer.com",
        "name": "Test User",
        "role": "user",
        "permissions": ["test:read", "test:write", "career:read"]
    }


def get_admin_user() -> Dict[str, Any]:
    """Get admin user for development/testing."""
    return {
        "id": "admin_user_123",
        "email": "admin@trucareer.com",
        "name": "Admin User",
        "role": "admin",
        "permissions": ["*"]  # All permissions
    }


# Request Validation Dependencies

def validate_test_id(test_id: str) -> str:
    """Validate test ID format."""
    if not test_id or len(test_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid test ID format"
        )
    return test_id


def validate_user_id(user_id: str) -> str:
    """Validate user ID format."""
    if not user_id or len(user_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    return user_id


def validate_career_id(career_id: str) -> str:
    """Validate career ID format."""
    if not career_id or len(career_id) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid career ID format"
        )
    return career_id