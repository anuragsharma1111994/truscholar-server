"""Authentication API endpoints for TruScholar application.

This module provides FastAPI routes for user authentication including
login, registration, token management, and session handling.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.schemas.auth_schemas import (
    LoginRequest,
    LoginResponse,
    LogoutRequest,
    RefreshTokenRequest,
    RefreshTokenResponse,
    UserRegistration,
    UserRegistrationResponse,
    UserSessionsResponse,
    VerifyTokenRequest,
    VerifyTokenResponse,
)
from src.schemas.base import (
    BaseResponse,
    ErrorResponse,
    SuccessResponse,
    create_error_response,
    create_success_response,
)
from src.services.auth_service import AuthService, TokenData
from src.utils.exceptions import (
    AuthenticationError,
    ValidationError,
    TruScholarError,
)
from src.utils.logger import get_api_logger, log_api_request, log_api_response

# Initialize router
router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={
        401: {"description": "Authentication failed"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Security scheme
security = HTTPBearer(auto_error=False)

# Initialize services
auth_service = AuthService()

# Logger
logger = get_api_logger()


# Dependency to get client IP
def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Dependency to get user agent
def get_user_agent(request: Request) -> str:
    """Get user agent from request."""
    return request.headers.get("User-Agent", "unknown")


# Dependency to get current user from token
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenData:
    """Get current authenticated user from token.

    Args:
        credentials: Authorization credentials

    Returns:
        TokenData: Current user token data

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token_data = await auth_service.verify_token(credentials.credentials)
        return token_data
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


# Optional dependency to get current user (doesn't raise error if not authenticated)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """Get current user if authenticated, None otherwise.

    Args:
        credentials: Authorization credentials

    Returns:
        TokenData or None: Current user token data if authenticated
    """
    if not credentials:
        return None

    try:
        token_data = await auth_service.verify_token(credentials.credentials)
        return token_data
    except AuthenticationError:
        return None


@router.post(
    "/register",
    response_model=SuccessResponse[UserRegistrationResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Register a new user account with phone number and name"
)
async def register_user(
    registration_data: UserRegistration,
    request: Request,
    client_ip: str = Depends(get_client_ip),
    user_agent: str = Depends(get_user_agent)
) -> SuccessResponse[UserRegistrationResponse]:
    """Register a new user account.

    Args:
        registration_data: User registration information
        request: FastAPI request object
        client_ip: Client IP address
        user_agent: Client user agent

    Returns:
        SuccessResponse[UserRegistrationResponse]: Registration response with tokens

    Raises:
        HTTPException: If registration fails
    """
    log_api_request("POST", "/auth/register", logger=logger)

    try:
        response = await auth_service.register_user(
            registration_data=registration_data,
            ip_address=client_ip,
            user_agent=user_agent
        )

        log_api_response("POST", "/auth/register", status.HTTP_201_CREATED, 0, logger=logger)

        return create_success_response(
            data=response,
            message="User registered successfully"
        )

    except ValidationError as e:
        logger.warning(f"Registration validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except AuthenticationError as e:
        logger.warning(f"Registration authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post(
    "/login",
    response_model=SuccessResponse[LoginResponse],
    summary="User login",
    description="Authenticate user with phone number and name"
)
async def login_user(
    login_data: LoginRequest,
    request: Request,
    client_ip: str = Depends(get_client_ip),
    user_agent: str = Depends(get_user_agent)
) -> SuccessResponse[LoginResponse]:
    """Authenticate user and return access tokens.

    Args:
        login_data: User login credentials
        request: FastAPI request object
        client_ip: Client IP address
        user_agent: Client user agent

    Returns:
        SuccessResponse[LoginResponse]: Login response with tokens

    Raises:
        HTTPException: If login fails
    """
    log_api_request("POST", "/auth/login", logger=logger)

    try:
        response = await auth_service.authenticate_user(
            login_data=login_data,
            ip_address=client_ip,
            user_agent=user_agent
        )

        log_api_response("POST", "/auth/login", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=response,
            message="Login successful"
        )

    except AuthenticationError as e:
        logger.warning(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning(f"Login validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post(
    "/refresh",
    response_model=SuccessResponse[RefreshTokenResponse],
    summary="Refresh access token",
    description="Refresh access token using refresh token"
)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request
) -> SuccessResponse[RefreshTokenResponse]:
    """Refresh access token using refresh token.

    Args:
        refresh_data: Refresh token request
        request: FastAPI request object

    Returns:
        SuccessResponse[RefreshTokenResponse]: New access token

    Raises:
        HTTPException: If token refresh fails
    """
    log_api_request("POST", "/auth/refresh", logger=logger)

    try:
        response = await auth_service.refresh_token(refresh_data.refresh_token)

        log_api_response("POST", "/auth/refresh", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=response,
            message="Token refreshed successfully"
        )

    except AuthenticationError as e:
        logger.warning(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post(
    "/logout",
    response_model=SuccessResponse[dict],
    summary="User logout",
    description="Logout user and invalidate tokens"
)
async def logout_user(
    logout_data: LogoutRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Logout user and invalidate session.

    Args:
        logout_data: Logout request data
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Logout confirmation

    Raises:
        HTTPException: If logout fails
    """
    log_api_request("POST", "/auth/logout", current_user.user_id, logger=logger)

    try:
        # Get access token from authorization header
        auth_header = request.headers.get("authorization", "")
        access_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""

        success = await auth_service.logout_user(
            access_token=access_token,
            refresh_token=logout_data.refresh_token,
            logout_all_sessions=logout_data.logout_all_sessions
        )

        if success:
            log_api_response("POST", "/auth/logout", status.HTTP_200_OK, 0, logger=logger)

            return create_success_response(
                data={"logged_out": True},
                message="Logout successful"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )

    except TruScholarError as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post(
    "/verify",
    response_model=SuccessResponse[VerifyTokenResponse],
    summary="Verify access token",
    description="Verify if access token is valid"
)
async def verify_token(
    verify_data: VerifyTokenRequest,
    request: Request
) -> SuccessResponse[VerifyTokenResponse]:
    """Verify access token validity.

    Args:
        verify_data: Token verification request
        request: FastAPI request object

    Returns:
        SuccessResponse[VerifyTokenResponse]: Token verification result
    """
    log_api_request("POST", "/auth/verify", logger=logger)

    try:
        token_data = await auth_service.verify_token(verify_data.access_token)

        response = VerifyTokenResponse(
            valid=True,
            user_id=token_data.user_id,
            expires_at=token_data.expires_at,
            permissions=token_data.permissions
        )

        log_api_response("POST", "/auth/verify", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=response,
            message="Token is valid"
        )

    except AuthenticationError:
        # Token is invalid, but we return success with valid=False
        response = VerifyTokenResponse(
            valid=False,
            user_id=None,
            expires_at=None,
            permissions=[]
        )

        return create_success_response(
            data=response,
            message="Token is invalid"
        )


@router.get(
    "/me",
    response_model=SuccessResponse[dict],
    summary="Get current user info",
    description="Get information about currently authenticated user"
)
async def get_current_user_info(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Get current user information from token.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Current user information
    """
    log_api_request("GET", "/auth/me", current_user.user_id, logger=logger)

    user_info = {
        "user_id": current_user.user_id,
        "phone": current_user.phone,
        "name": current_user.name,
        "account_type": current_user.account_type,
        "permissions": current_user.permissions,
        "session_id": current_user.session_id,
        "issued_at": current_user.issued_at.isoformat(),
        "expires_at": current_user.expires_at.isoformat()
    }

    log_api_response("GET", "/auth/me", status.HTTP_200_OK, 0, logger=logger)

    return create_success_response(
        data=user_info,
        message="User information retrieved successfully"
    )


@router.get(
    "/sessions",
    response_model=SuccessResponse[UserSessionsResponse],
    summary="Get user sessions",
    description="Get list of active sessions for current user"
)
async def get_user_sessions(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[UserSessionsResponse]:
    """Get active sessions for current user.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[UserSessionsResponse]: User sessions
    """
    log_api_request("GET", "/auth/sessions", current_user.user_id, logger=logger)

    try:
        # TODO: Implement session listing in auth service
        # For now, return basic response
        sessions_response = UserSessionsResponse(
            current_session={
                "session_id": current_user.session_id,
                "user_id": current_user.user_id,
                "device_info": {},
                "created_at": current_user.issued_at,
                "last_accessed": current_user.issued_at,
                "expires_at": current_user.expires_at,
                "is_active": True
            },
            other_sessions=[],
            total_sessions=1
        )

        log_api_response("GET", "/auth/sessions", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=sessions_response,
            message="Sessions retrieved successfully"
        )

    except TruScholarError as e:
        logger.error(f"Get sessions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@router.delete(
    "/sessions/{session_id}",
    response_model=SuccessResponse[dict],
    summary="Invalidate session",
    description="Invalidate a specific session"
)
async def invalidate_session(
    session_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Invalidate a specific session.

    Args:
        session_id: Session ID to invalidate
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Invalidation result
    """
    log_api_request("DELETE", f"/auth/sessions/{session_id}", current_user.user_id, logger=logger)

    try:
        # TODO: Implement session invalidation in auth service
        success = await auth_service.session_manager.invalidate_session(session_id)

        if success:
            log_api_response("DELETE", f"/auth/sessions/{session_id}", status.HTTP_200_OK, 0, logger=logger)

            return create_success_response(
                data={"invalidated": True},
                message="Session invalidated successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

    except TruScholarError as e:
        logger.error(f"Session invalidation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to invalidate session"
        )


# Health check endpoint
@router.get(
    "/health",
    response_model=SuccessResponse[dict],
    summary="Authentication service health check",
    description="Check if authentication service is healthy"
)
async def auth_health_check() -> SuccessResponse[dict]:
    """Check authentication service health.

    Returns:
        SuccessResponse[dict]: Health status
    """
    return create_success_response(
        data={
            "service": "authentication",
            "status": "healthy",
            "version": "1.0.0"
        },
        message="Authentication service is healthy"
    )


# Export router and dependencies
__all__ = [
    "router",
    "get_current_user",
    "get_current_user_optional",
    "get_client_ip",
    "get_user_agent",
]
