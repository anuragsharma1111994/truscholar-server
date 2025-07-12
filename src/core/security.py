"""Security and authentication utilities for TruCareer system.

This module provides JWT token handling, password hashing, permission management,
and other security-related utilities for the application.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status
from enum import Enum

from src.core.config import get_settings
from src.utils.logger import get_logger
from src.utils.exceptions import AuthenticationError, AuthorizationError

settings = get_settings()
logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(Enum):
    """User role definitions."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"


class Permission(Enum):
    """Permission definitions."""
    # User permissions
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    
    # Test permissions
    TEST_READ = "test:read"
    TEST_WRITE = "test:write"
    TEST_DELETE = "test:delete"
    TEST_ADMIN = "test:admin"
    
    # Career permissions
    CAREER_READ = "career:read"
    CAREER_WRITE = "career:write"
    CAREER_ADMIN = "career:admin"
    
    # Question permissions
    QUESTION_READ = "question:read"
    QUESTION_WRITE = "question:write"
    QUESTION_ADMIN = "question:admin"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    
    # Analytics permissions
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_WRITE = "analytics:write"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.USER_READ, Permission.USER_WRITE, Permission.USER_DELETE,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_DELETE, Permission.TEST_ADMIN,
        Permission.CAREER_READ, Permission.CAREER_WRITE, Permission.CAREER_ADMIN,
        Permission.QUESTION_READ, Permission.QUESTION_WRITE, Permission.QUESTION_ADMIN,
        Permission.SYSTEM_ADMIN, Permission.SYSTEM_MONITOR,
        Permission.ANALYTICS_READ, Permission.ANALYTICS_WRITE
    ],
    UserRole.MODERATOR: [
        Permission.USER_READ, Permission.USER_WRITE,
        Permission.TEST_READ, Permission.TEST_WRITE, Permission.TEST_ADMIN,
        Permission.CAREER_READ, Permission.CAREER_WRITE,
        Permission.QUESTION_READ, Permission.QUESTION_WRITE,
        Permission.SYSTEM_MONITOR,
        Permission.ANALYTICS_READ
    ],
    UserRole.USER: [
        Permission.USER_READ,
        Permission.TEST_READ, Permission.TEST_WRITE,
        Permission.CAREER_READ,
        Permission.QUESTION_READ
    ],
    UserRole.GUEST: [
        Permission.CAREER_READ,
        Permission.QUESTION_READ
    ]
}


class SecurityManager:
    """Main security manager for authentication and authorization."""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    
    # Password Management
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def generate_password_reset_token(self, email: str) -> str:
        """Generate a password reset token."""
        data = {
            "email": email,
            "type": "password_reset",
            "exp": datetime.utcnow() + timedelta(hours=1),  # 1 hour expiry
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        }
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)
    
    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """Verify password reset token and return email."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "password_reset":
                return None
            
            return payload.get("email")
            
        except JWTError:
            return None
    
    # JWT Token Management
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "jti": secrets.token_urlsafe(32)
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Created access token for user: {data.get('sub', 'unknown')}")
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32)
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Created refresh token for user: {user_id}")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise AuthenticationError("Token has expired")
            
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            raise AuthenticationError("Invalid token")
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token."""
        try:
            payload = self.verify_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")
            
            user_id = payload.get("sub")
            if not user_id:
                raise AuthenticationError("Invalid user ID in token")
            
            # Create new access token
            access_token_data = {
                "sub": user_id,
                "email": payload.get("email"),
                "name": payload.get("name"),
                "role": payload.get("role", "user"),
                "permissions": payload.get("permissions", [])
            }
            
            new_access_token = self.create_access_token(access_token_data)
            
            return {
                "access_token": new_access_token,
                "token_type": "bearer"
            }
            
        except Exception as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise AuthenticationError("Token refresh failed")
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token (add to blacklist)."""
        try:
            payload = self.verify_token(token)
            jti = payload.get("jti")
            
            if jti:
                # In a real implementation, you would add JTI to a blacklist
                # For now, we'll just log it
                logger.info(f"Token revoked: {jti}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Token revocation failed: {str(e)}")
            return False
    
    # Permission Management
    
    def get_user_permissions(self, role: Union[str, UserRole]) -> List[Permission]:
        """Get permissions for a user role."""
        if isinstance(role, str):
            try:
                role = UserRole(role)
            except ValueError:
                logger.warning(f"Unknown role: {role}")
                return []
        
        return ROLE_PERMISSIONS.get(role, [])
    
    def has_permission(self, user_permissions: List[str], required_permission: Union[str, Permission]) -> bool:
        """Check if user has required permission."""
        if isinstance(required_permission, Permission):
            required_permission = required_permission.value
        
        # Admin wildcard permission
        if "*" in user_permissions:
            return True
        
        return required_permission in user_permissions
    
    def has_any_permission(self, user_permissions: List[str], required_permissions: List[Union[str, Permission]]) -> bool:
        """Check if user has any of the required permissions."""
        for permission in required_permissions:
            if self.has_permission(user_permissions, permission):
                return True
        return False
    
    def has_all_permissions(self, user_permissions: List[str], required_permissions: List[Union[str, Permission]]) -> bool:
        """Check if user has all required permissions."""
        for permission in required_permissions:
            if not self.has_permission(user_permissions, permission):
                return False
        return True
    
    def check_permission(self, user_permissions: List[str], required_permission: Union[str, Permission]):
        """Check permission and raise exception if not authorized."""
        if not self.has_permission(user_permissions, required_permission):
            permission_str = required_permission.value if isinstance(required_permission, Permission) else required_permission
            raise AuthorizationError(f"Permission denied: {permission_str}")
    
    # API Key Management
    
    def generate_api_key(self, user_id: str, name: str, permissions: List[str] = None) -> Dict[str, Any]:
        """Generate API key for user."""
        api_key = f"tk_{secrets.token_urlsafe(32)}"
        
        # Create API key data
        api_key_data = {
            "key": api_key,
            "user_id": user_id,
            "name": name,
            "permissions": permissions or [],
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "is_active": True
        }
        
        # In a real implementation, you would store this in database
        logger.info(f"Generated API key for user {user_id}: {name}")
        
        return api_key_data
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data."""
        # In a real implementation, you would lookup from database
        # For now, return None (not implemented)
        return None
    
    # Security Utilities
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: str, salt: str = None) -> str:
        """Hash data with optional salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def constant_time_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        return secrets.compare_digest(a.encode(), b.encode())
    
    # Rate Limiting Helpers
    
    def generate_rate_limit_key(self, identifier: str, endpoint: str) -> str:
        """Generate rate limiting key."""
        return f"rate_limit:{identifier}:{endpoint}"
    
    def check_rate_limit(self, key: str, limit: int, window: int = 3600) -> Dict[str, Any]:
        """Check rate limit for a key."""
        # In a real implementation, this would use Redis
        # For now, return a mock response
        return {
            "allowed": True,
            "remaining": limit - 1,
            "reset_time": datetime.utcnow() + timedelta(seconds=window)
        }


# Global security manager instance
security_manager = SecurityManager()


# Convenience functions

def hash_password(password: str) -> str:
    """Hash password using global security manager."""
    return security_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using global security manager."""
    return security_manager.verify_password(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create access token using global security manager."""
    return security_manager.create_access_token(data, expires_delta)


def create_refresh_token(user_id: str) -> str:
    """Create refresh token using global security manager."""
    return security_manager.create_refresh_token(user_id)


def verify_token(token: str) -> Dict[str, Any]:
    """Verify token using global security manager."""
    return security_manager.verify_token(token)


def get_user_permissions(role: Union[str, UserRole]) -> List[Permission]:
    """Get user permissions using global security manager."""
    return security_manager.get_user_permissions(role)


def has_permission(user_permissions: List[str], required_permission: Union[str, Permission]) -> bool:
    """Check permission using global security manager."""
    return security_manager.has_permission(user_permissions, required_permission)


def require_permissions(required_permissions: List[Union[str, Permission]]):
    """Decorator for requiring permissions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented as a proper FastAPI dependency
            # For now, just return the function
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Security middleware helpers

def get_client_ip(request) -> str:
    """Extract client IP from request."""
    # Check for forwarded headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host if request.client else "unknown"


def log_security_event(event_type: str, user_id: str = None, details: Dict[str, Any] = None):
    """Log security-related events."""
    log_data = {
        "event_type": event_type,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {}
    }
    
    logger.warning(f"Security event: {event_type}", extra=log_data)


# Exception handlers for security

class SecurityException(Exception):
    """Base security exception."""
    pass


class TokenExpiredException(SecurityException):
    """Token has expired."""
    pass


class InvalidTokenException(SecurityException):
    """Token is invalid."""
    pass


class InsufficientPermissionsException(SecurityException):
    """User lacks required permissions."""
    pass


def handle_security_exception(exc: SecurityException) -> HTTPException:
    """Convert security exceptions to HTTP exceptions."""
    if isinstance(exc, TokenExpiredException):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    elif isinstance(exc, InvalidTokenException):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    elif isinstance(exc, InsufficientPermissionsException):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security error"
        )