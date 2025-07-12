"""Authentication service for TruScholar application.

This module provides authentication logic including login, token management,
session handling, and user verification.
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import jwt
from pydantic import BaseModel

from src.core.config import get_settings
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.models.user import User, UserSession
from src.schemas.auth_schemas import (
    LoginRequest,
    LoginResponse,
    RefreshTokenResponse,
    TokenInfo,
    UserBasicInfo,
    UserRegistration,
    UserRegistrationResponse,
)
from src.utils.constants import UserAccountType
from src.utils.datetime_utils import utc_now
from src.utils.exceptions import AuthenticationError, AuthorizationError, ValidationError
from src.utils.helpers import generate_random_string, mask_phone_number
from src.utils.logger import get_security_logger
from src.utils.validators import validate_name, validate_phone

settings = get_settings()
logger = get_security_logger()


class TokenData(BaseModel):
    """Token payload data."""

    user_id: str
    phone: str
    name: str
    account_type: str
    permissions: List[str]
    session_id: str
    issued_at: datetime
    expires_at: datetime


class JWTManager:
    """JWT token management utility."""

    def __init__(self):
        """Initialize JWT manager with settings."""
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS

    def create_access_token(
        self,
        user_id: str,
        phone: str,
        name: str,
        account_type: str,
        session_id: str,
        permissions: Optional[List[str]] = None
    ) -> Tuple[str, datetime]:
        """Create JWT access token.

        Args:
            user_id: User ID
            phone: User phone
            name: User name
            account_type: Account type
            session_id: Session ID
            permissions: User permissions

        Returns:
            Tuple[str, datetime]: Token and expiration time
        """
        now = utc_now()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "phone": phone,
            "name": name,
            "account_type": account_type,
            "session_id": session_id,
            "permissions": permissions or [],
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "access"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expire

    def create_refresh_token(
        self,
        user_id: str,
        session_id: str
    ) -> Tuple[str, datetime]:
        """Create JWT refresh token.

        Args:
            user_id: User ID
            session_id: Session ID

        Returns:
            Tuple[str, datetime]: Token and expiration time
        """
        now = utc_now()
        expire = now + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "session_id": session_id,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "type": "refresh"
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token, expire

    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token.

        Args:
            token: JWT token to verify
            token_type: Expected token type (access/refresh)

        Returns:
            TokenData: Decoded token data

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify token type
            if payload.get("type") != token_type:
                raise AuthenticationError("Invalid token type")

            # Check expiration
            exp_timestamp = payload.get("exp")
            if not exp_timestamp or datetime.fromtimestamp(exp_timestamp) < utc_now():
                raise AuthenticationError("Token has expired")

            # For access tokens, return full data
            if token_type == "access":
                return TokenData(
                    user_id=payload["sub"],
                    phone=payload.get("phone", ""),
                    name=payload.get("name", ""),
                    account_type=payload.get("account_type", "free"),
                    permissions=payload.get("permissions", []),
                    session_id=payload.get("session_id", ""),
                    issued_at=datetime.fromtimestamp(payload["iat"]),
                    expires_at=datetime.fromtimestamp(payload["exp"])
                )
            else:
                # For refresh tokens, return minimal data
                return TokenData(
                    user_id=payload["sub"],
                    phone="",
                    name="",
                    account_type="",
                    permissions=[],
                    session_id=payload.get("session_id", ""),
                    issued_at=datetime.fromtimestamp(payload["iat"]),
                    expires_at=datetime.fromtimestamp(payload["exp"])
                )

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
        except KeyError as e:
            raise AuthenticationError(f"Missing token field: {str(e)}")


class SessionManager:
    """Session management utility."""

    def __init__(self):
        """Initialize session manager."""
        self.session_timeout_hours = 24
        self.max_sessions_per_user = 5

    async def create_session(
        self,
        user_id: str,
        device_info: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create new user session.

        Args:
            user_id: User ID
            device_info: Device information
            ip_address: IP address
            user_agent: User agent string

        Returns:
            UserSession: Created session
        """
        # Generate unique session token
        session_token = generate_random_string(32, include_special=False)
        refresh_token = generate_random_string(32, include_special=False)

        # Set expiration
        expires_at = utc_now() + timedelta(hours=self.session_timeout_hours)
        refresh_expires_at = utc_now() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

        # Create session
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            refresh_token=refresh_token,
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info,
            expires_at=expires_at,
            refresh_expires_at=refresh_expires_at,
            is_active=True
        )

        # Clean up old sessions if user has too many
        await self._cleanup_old_sessions(user_id)

        # Save session to database
        await MongoDB.create_document("sessions", session.to_dict())

        # Cache session in Redis
        await RedisClient.set_json(
            f"session:{session_token}",
            {
                "user_id": user_id,
                "session_id": str(session.id),
                "expires_at": expires_at.isoformat()
            },
            ttl=int(timedelta(hours=self.session_timeout_hours).total_seconds())
        )

        logger.info(
            "Session created",
            extra={
                "user_id": user_id,
                "session_id": str(session.id),
                "ip_address": ip_address,
                "device_type": device_info.get("device_type") if device_info else None
            }
        )

        return session

    async def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get session by token.

        Args:
            session_token: Session token

        Returns:
            UserSession or None: Session if found and valid
        """
        # Try cache first
        cached_session = await RedisClient.get_json(f"session:{session_token}")
        if cached_session:
            session_doc = await MongoDB.find_document(
                "sessions",
                {"session_token": session_token, "is_active": True}
            )
            if session_doc:
                session = UserSession(**session_doc)
                if session.is_valid():
                    return session

        # Session not in cache or invalid
        await self._invalidate_session_cache(session_token)
        return None

    async def refresh_session(self, refresh_token: str) -> Optional[UserSession]:
        """Refresh session using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            UserSession or None: Refreshed session if valid
        """
        session_doc = await MongoDB.find_document(
            "sessions",
            {"refresh_token": refresh_token, "is_active": True}
        )

        if not session_doc:
            return None

        session = UserSession(**session_doc)

        # Check if refresh token is still valid
        if not session.refresh_expires_at or utc_now() > session.refresh_expires_at:
            await self.invalidate_session(session.session_token)
            return None

        # Extend session expiration
        new_expires_at = utc_now() + timedelta(hours=self.session_timeout_hours)

        await MongoDB.update_document(
            "sessions",
            {"_id": session.id},
            {
                "expires_at": new_expires_at,
                "last_accessed": utc_now()
            }
        )

        # Update cache
        await RedisClient.set_json(
            f"session:{session.session_token}",
            {
                "user_id": session.user_id,
                "session_id": str(session.id),
                "expires_at": new_expires_at.isoformat()
            },
            ttl=int(timedelta(hours=self.session_timeout_hours).total_seconds())
        )

        session.expires_at = new_expires_at
        session.last_accessed = utc_now()

        return session

    async def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session.

        Args:
            session_token: Session token to invalidate

        Returns:
            bool: True if session was invalidated
        """
        # Update database
        result = await MongoDB.update_document(
            "sessions",
            {"session_token": session_token},
            {"is_active": False}
        )

        # Remove from cache
        await self._invalidate_session_cache(session_token)

        logger.info(
            "Session invalidated",
            extra={"session_token": session_token[:8] + "..."}
        )

        return result is not None

    async def invalidate_user_sessions(self, user_id: str, except_session_id: Optional[str] = None) -> int:
        """Invalidate all sessions for a user.

        Args:
            user_id: User ID
            except_session_id: Session ID to keep active

        Returns:
            int: Number of sessions invalidated
        """
        query = {"user_id": user_id, "is_active": True}
        if except_session_id:
            query["_id"] = {"$ne": except_session_id}

        # Get sessions to invalidate
        sessions = await MongoDB.find_documents("sessions", query)

        # Invalidate in database
        await MongoDB.update_documents(
            "sessions",
            query,
            {"is_active": False}
        )

        # Remove from cache
        for session in sessions:
            await self._invalidate_session_cache(session["session_token"])

        logger.info(
            "User sessions invalidated",
            extra={
                "user_id": user_id,
                "sessions_invalidated": len(sessions),
                "except_session_id": except_session_id
            }
        )

        return len(sessions)

    async def _cleanup_old_sessions(self, user_id: str) -> None:
        """Clean up old sessions for user.

        Args:
            user_id: User ID
        """
        # Get active sessions sorted by last access
        sessions = await MongoDB.find_documents(
            "sessions",
            {"user_id": user_id, "is_active": True},
            sort=[("last_accessed", -1)]
        )

        # If user has too many sessions, deactivate oldest ones
        if len(sessions) >= self.max_sessions_per_user:
            sessions_to_remove = sessions[self.max_sessions_per_user - 1:]
            session_ids = [session["_id"] for session in sessions_to_remove]

            await MongoDB.update_documents(
                "sessions",
                {"_id": {"$in": session_ids}},
                {"is_active": False}
            )

            # Remove from cache
            for session in sessions_to_remove:
                await self._invalidate_session_cache(session["session_token"])

    async def _invalidate_session_cache(self, session_token: str) -> None:
        """Remove session from cache.

        Args:
            session_token: Session token
        """
        await RedisClient.delete(f"session:{session_token}")


class AuthService:
    """Authentication service for user login, registration, and token management."""

    def __init__(self):
        """Initialize authentication service."""
        self.jwt_manager = JWTManager()
        self.session_manager = SessionManager()

    async def register_user(
        self,
        registration_data: UserRegistration,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserRegistrationResponse:
        """Register a new user.

        Args:
            registration_data: User registration data
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            UserRegistrationResponse: Registration response with tokens

        Raises:
            ValidationError: If registration data is invalid
            AuthenticationError: If user already exists
        """
        # Check if user already exists
        existing_user = await MongoDB.find_document(
            "users",
            {"phone": registration_data.phone}
        )

        if existing_user:
            logger.warning(
                "Registration attempt for existing phone",
                extra={"phone": mask_phone_number(registration_data.phone)}
            )
            raise AuthenticationError("User with this phone number already exists")

        # Create user from registration data
        user_data = {
            "phone": registration_data.phone,
            "name": registration_data.name,
            "email": registration_data.email,
            "account_type": UserAccountType.FREE,
            "is_active": True,
            "is_verified": False,
            "terms_accepted": registration_data.terms_accepted,
            "privacy_accepted": registration_data.privacy_accepted,
            "data_consent": registration_data.data_consent,
            "terms_accepted_at": utc_now(),
            "privacy_accepted_at": utc_now(),
        }

        # Add profile data if provided
        if registration_data.age:
            user_data["profile"] = {
                "age": registration_data.age,
                "education_level": registration_data.education_level,
                "current_occupation": registration_data.current_occupation,
                "location_city": registration_data.location_city,
                "location_state": registration_data.location_state,
            }

        # Add preferences
        user_data["preferences"] = {
            "marketing_emails": registration_data.marketing_emails,
        }

        # Create user
        user = User(**user_data)
        user_doc = await MongoDB.create_document("users", user.to_dict())
        user_id = str(user_doc["_id"])

        # Create session
        session = await self.session_manager.create_session(
            user_id=user_id,
            device_info=registration_data.device_info,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Create tokens
        permissions = self._get_user_permissions(user.account_type, user.is_admin)
        access_token, access_expires = self.jwt_manager.create_access_token(
            user_id=user_id,
            phone=user.phone,
            name=user.name,
            account_type=user.account_type.value,
            session_id=str(session.id),
            permissions=permissions
        )

        refresh_token, refresh_expires = self.jwt_manager.create_refresh_token(
            user_id=user_id,
            session_id=str(session.id)
        )

        # Update user login info
        await MongoDB.update_document(
            "users",
            {"_id": user.id},
            {
                "last_login": utc_now(),
                "login_count": 1
            }
        )

        logger.info(
            "User registered successfully",
            extra={
                "user_id": user_id,
                "phone": mask_phone_number(user.phone),
                "name": user.name
            }
        )

        return UserRegistrationResponse(
            user=UserBasicInfo(
                id=user_id,
                phone=mask_phone_number(user.phone),
                name=user.name,
                account_type=user.account_type,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at,
                last_login=utc_now()
            ),
            tokens=TokenInfo(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=int(timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds()),
                expires_at=access_expires
            ),
            session_id=str(session.id),
            welcome_message="Welcome to TruScholar! Let's discover your perfect career path.",
            next_steps=[
                "Complete your profile",
                "Take your first RAISEC assessment",
                "Explore career recommendations"
            ]
        )

    async def authenticate_user(
        self,
        login_data: LoginRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> LoginResponse:
        """Authenticate user with phone and name.

        Args:
            login_data: Login credentials
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            LoginResponse: Authentication response with tokens

        Raises:
            AuthenticationError: If authentication fails
        """
        # Find user by phone and name
        user_doc = await MongoDB.find_document(
            "users",
            {"phone": login_data.phone, "name": login_data.name}
        )

        if not user_doc:
            logger.warning(
                "Authentication failed - user not found",
                extra={
                    "phone": mask_phone_number(login_data.phone),
                    "name": login_data.name
                }
            )
            raise AuthenticationError("Invalid phone number or name")

        user = User(**user_doc)

        # Check if user is active
        if not user.is_active:
            logger.warning(
                "Authentication failed - user inactive",
                extra={"user_id": str(user.id)}
            )
            raise AuthenticationError("User account is inactive")

        # Create session
        session = await self.session_manager.create_session(
            user_id=str(user.id),
            device_info=login_data.device_info,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Create tokens
        permissions = self._get_user_permissions(user.account_type, user.is_admin)
        access_token, access_expires = self.jwt_manager.create_access_token(
            user_id=str(user.id),
            phone=user.phone,
            name=user.name,
            account_type=user.account_type.value,
            session_id=str(session.id),
            permissions=permissions
        )

        refresh_token, refresh_expires = self.jwt_manager.create_refresh_token(
            user_id=str(user.id),
            session_id=str(session.id)
        )

        # Update user login info
        await MongoDB.update_document(
            "users",
            {"_id": user.id},
            {
                "last_login": utc_now(),
                "login_count": user.login_count + 1,
                "last_active": utc_now()
            }
        )

        logger.info(
            "User authenticated successfully",
            extra={
                "user_id": str(user.id),
                "phone": mask_phone_number(user.phone),
                "session_id": str(session.id)
            }
        )

        return LoginResponse(
            user=UserBasicInfo(
                id=str(user.id),
                phone=mask_phone_number(user.phone),
                name=user.name,
                account_type=user.account_type,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at,
                last_login=utc_now()
            ),
            tokens=TokenInfo(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=int(timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds()),
                expires_at=access_expires
            ),
            session_id=str(session.id),
            permissions=permissions
        )

    async def refresh_token(self, refresh_token: str) -> RefreshTokenResponse:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            RefreshTokenResponse: New tokens

        Raises:
            AuthenticationError: If refresh token is invalid
        """
        # Verify refresh token
        try:
            token_data = self.jwt_manager.verify_token(refresh_token, "refresh")
        except AuthenticationError:
            raise AuthenticationError("Invalid refresh token")

        # Refresh session
        session = await self.session_manager.refresh_session(refresh_token)
        if not session:
            raise AuthenticationError("Session expired or invalid")

        # Get user data
        user_doc = await MongoDB.find_document("users", {"_id": token_data.user_id})
        if not user_doc:
            raise AuthenticationError("User not found")

        user = User(**user_doc)

        # Create new tokens
        permissions = self._get_user_permissions(user.account_type, user.is_admin)
        access_token, access_expires = self.jwt_manager.create_access_token(
            user_id=str(user.id),
            phone=user.phone,
            name=user.name,
            account_type=user.account_type.value,
            session_id=str(session.id),
            permissions=permissions
        )

        new_refresh_token, refresh_expires = self.jwt_manager.create_refresh_token(
            user_id=str(user.id),
            session_id=str(session.id)
        )

        logger.info(
            "Token refreshed",
            extra={
                "user_id": str(user.id),
                "session_id": str(session.id)
            }
        )

        return RefreshTokenResponse(
            tokens=TokenInfo(
                access_token=access_token,
                refresh_token=new_refresh_token,
                token_type="bearer",
                expires_in=int(timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES).total_seconds()),
                expires_at=access_expires
            )
        )

    async def logout_user(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        logout_all_sessions: bool = False
    ) -> bool:
        """Logout user by invalidating tokens and sessions.

        Args:
            access_token: Current access token
            refresh_token: Optional refresh token
            logout_all_sessions: Whether to logout from all sessions

        Returns:
            bool: True if logout was successful
        """
        try:
            # Verify access token
            token_data = self.jwt_manager.verify_token(access_token, "access")

            if logout_all_sessions:
                # Invalidate all user sessions
                await self.session_manager.invalidate_user_sessions(token_data.user_id)
            else:
                # Invalidate current session only
                session = await self.session_manager.get_session(token_data.session_id)
                if session:
                    await self.session_manager.invalidate_session(session.session_token)

            logger.info(
                "User logged out",
                extra={
                    "user_id": token_data.user_id,
                    "session_id": token_data.session_id,
                    "logout_all": logout_all_sessions
                }
            )

            return True

        except AuthenticationError:
            # Token invalid but still try to logout if refresh token provided
            if refresh_token:
                try:
                    refresh_data = self.jwt_manager.verify_token(refresh_token, "refresh")
                    await self.session_manager.invalidate_user_sessions(
                        refresh_data.user_id,
                        None if logout_all_sessions else refresh_data.session_id
                    )
                    return True
                except AuthenticationError:
                    pass

            return False

    async def verify_token(self, access_token: str) -> TokenData:
        """Verify access token and return user data.

        Args:
            access_token: Access token to verify

        Returns:
            TokenData: Token data if valid

        Raises:
            AuthenticationError: If token is invalid
        """
        token_data = self.jwt_manager.verify_token(access_token, "access")

        # Verify session is still active
        session = await self.session_manager.get_session(token_data.session_id)
        if not session:
            raise AuthenticationError("Session expired or invalid")

        return token_data

    def _get_user_permissions(self, account_type: UserAccountType, is_admin: bool = False) -> List[str]:
        """Get user permissions based on account type and admin status.

        Args:
            account_type: User account type
            is_admin: Whether user is admin

        Returns:
            List[str]: List of permissions
        """
        permissions = [
            "user:read",
            "user:update",
            "test:create",
            "test:read",
            "test:update",
            "career:read",
            "report:read",
        ]

        if account_type == UserAccountType.PREMIUM:
            permissions.extend([
                "test:unlimited",
                "report:export",
                "career:advanced",
            ])
        elif account_type == UserAccountType.ENTERPRISE:
            permissions.extend([
                "test:unlimited",
                "report:export",
                "career:advanced",
                "analytics:read",
                "bulk:operations",
            ])

        if is_admin:
            permissions.extend([
                "admin:read",
                "admin:write",
                "user:admin",
                "system:monitor",
            ])

        return list(set(permissions))  # Remove duplicates


# Export service classes
__all__ = [
    "AuthService",
    "JWTManager",
    "SessionManager",
    "TokenData",
]
