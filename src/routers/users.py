"""User management API endpoints for TruScholar application.

This module provides FastAPI routes for user CRUD operations, profile management,
preferences, and user search functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from typing import List, Optional

from src.routers.auth import get_current_user, TokenData
from src.schemas.base import (
    PaginatedResponse,
    SuccessResponse,
    create_paginated_response,
    create_success_response,
)
from src.schemas.user_schemas import (
    UserCreate,
    UserProfileUpdate,
    UserPreferencesUpdate,
    UserResponse,
    UserSearch,
    UserSummary,
    UserUpdate,
)
from src.services.user_service import UserService, UserProfileService, UserStatsService
from src.utils.exceptions import (
    ResourceNotFoundError,
    ValidationError,
    BusinessLogicError,
    TruScholarError,
)
from src.utils.logger import get_api_logger, log_api_request, log_api_response

# Initialize router
router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Permission denied"},
        404: {"description": "User not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Initialize services
user_service = UserService()
profile_service = UserProfileService()
stats_service = UserStatsService()

# Logger
logger = get_api_logger()


# Dependency to check admin permissions
async def require_admin_permissions(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Require admin permissions for endpoint access.

    Args:
        current_user: Current authenticated user

    Returns:
        TokenData: Current user if admin

    Raises:
        HTTPException: If user is not admin
    """
    if "admin:read" not in current_user.permissions and "user:admin" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    return current_user


# Dependency to check if user can access another user's data
async def check_user_access(
    user_id: str,
    current_user: TokenData = Depends(get_current_user)
) -> str:
    """Check if current user can access specified user's data.

    Args:
        user_id: Target user ID
        current_user: Current authenticated user

    Returns:
        str: User ID if access is allowed

    Raises:
        HTTPException: If access is denied
    """
    # Users can access their own data
    if current_user.user_id == user_id:
        return user_id

    # Admins can access any user's data
    if "admin:read" in current_user.permissions or "user:admin" in current_user.permissions:
        return user_id

    # Otherwise, access denied
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Cannot access other user's data"
    )


@router.post(
    "/",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create user",
    description="Create a new user account (Admin only)"
)
async def create_user(
    user_data: UserCreate,
    request: Request,
    current_user: TokenData = Depends(require_admin_permissions)
) -> SuccessResponse[UserResponse]:
    """Create a new user account.

    Args:
        user_data: User creation data
        request: FastAPI request object
        current_user: Current authenticated admin user

    Returns:
        SuccessResponse[UserResponse]: Created user data

    Raises:
        HTTPException: If creation fails
    """
    log_api_request("POST", "/users", current_user.user_id, logger=logger)

    try:
        user = await user_service.create_user(user_data)

        log_api_response("POST", "/users", status.HTTP_201_CREATED, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User created successfully"
        )

    except ValidationError as e:
        logger.warning(f"User creation validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except BusinessLogicError as e:
        logger.warning(f"User creation business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"User creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get(
    "/me",
    response_model=SuccessResponse[UserResponse],
    summary="Get current user",
    description="Get current user's complete profile"
)
async def get_current_user_profile(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[UserResponse]:
    """Get current user's profile.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[UserResponse]: User profile data

    Raises:
        HTTPException: If user not found
    """
    log_api_request("GET", "/users/me", current_user.user_id, logger=logger)

    try:
        user = await user_service.get_user_by_id(current_user.user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Update last activity
        await user_service.update_user_activity(current_user.user_id)

        log_api_response("GET", "/users/me", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User profile retrieved successfully"
        )

    except TruScholarError as e:
        logger.error(f"Get current user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.get(
    "/{user_id}",
    response_model=SuccessResponse[UserResponse],
    summary="Get user by ID",
    description="Get user profile by ID"
)
async def get_user(
    user_id: str = Depends(check_user_access),
    request: Request = None,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[UserResponse]:
    """Get user by ID.

    Args:
        user_id: User ID (validated by dependency)
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[UserResponse]: User profile data

    Raises:
        HTTPException: If user not found
    """
    log_api_request("GET", f"/users/{user_id}", current_user.user_id, logger=logger)

    try:
        user = await user_service.get_user_by_id(user_id)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        log_api_response("GET", f"/users/{user_id}", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User profile retrieved successfully"
        )

    except TruScholarError as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.put(
    "/me",
    response_model=SuccessResponse[UserResponse],
    summary="Update current user",
    description="Update current user's basic information"
)
async def update_current_user(
    user_data: UserUpdate,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[UserResponse]:
    """Update current user's information.

    Args:
        user_data: User update data
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[UserResponse]: Updated user data

    Raises:
        HTTPException: If update fails
    """
    log_api_request("PUT", "/users/me", current_user.user_id, logger=logger)

    try:
        user = await user_service.update_user(current_user.user_id, user_data)

        log_api_response("PUT", "/users/me", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User profile updated successfully"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except ValidationError as e:
        logger.warning(f"User update validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"User update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.put(
    "/me/profile",
    response_model=SuccessResponse[UserResponse],
    summary="Update user profile",
    description="Update current user's profile information"
)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[UserResponse]:
    """Update current user's profile.

    Args:
        profile_data: Profile update data
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[UserResponse]: Updated user data

    Raises:
        HTTPException: If update fails
    """
    log_api_request("PUT", "/users/me/profile", current_user.user_id, logger=logger)

    try:
        user = await user_service.update_user_profile(current_user.user_id, profile_data)

        log_api_response("PUT", "/users/me/profile", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User profile updated successfully"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except ValidationError as e:
        logger.warning(f"Profile update validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Profile update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.put(
    "/me/preferences",
    response_model=SuccessResponse[UserResponse],
    summary="Update user preferences",
    description="Update current user's preferences and settings"
)
async def update_user_preferences(
    preferences_data: UserPreferencesUpdate,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[UserResponse]:
    """Update current user's preferences.

    Args:
        preferences_data: Preferences update data
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[UserResponse]: Updated user data

    Raises:
        HTTPException: If update fails
    """
    log_api_request("PUT", "/users/me/preferences", current_user.user_id, logger=logger)

    try:
        user = await user_service.update_user_preferences(current_user.user_id, preferences_data)

        log_api_response("PUT", "/users/me/preferences", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User preferences updated successfully"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except ValidationError as e:
        logger.warning(f"Preferences update validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Preferences update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )


@router.delete(
    "/me",
    response_model=SuccessResponse[dict],
    summary="Delete current user account",
    description="Delete current user's account (soft delete)"
)
async def delete_current_user(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
    hard_delete: bool = Query(False, description="Whether to perform hard delete")
) -> SuccessResponse[dict]:
    """Delete current user's account.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        hard_delete: Whether to perform hard delete

    Returns:
        SuccessResponse[dict]: Deletion confirmation

    Raises:
        HTTPException: If deletion fails
    """
    log_api_request("DELETE", "/users/me", current_user.user_id, logger=logger)

    try:
        success = await user_service.delete_user(
            current_user.user_id,
            soft_delete=not hard_delete
        )

        if success:
            log_api_response("DELETE", "/users/me", status.HTTP_200_OK, 0, logger=logger)

            return create_success_response(
                data={"deleted": True, "hard_delete": hard_delete},
                message="User account deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user account"
            )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except TruScholarError as e:
        logger.error(f"User deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user account"
        )


@router.get(
    "/",
    response_model=PaginatedResponse[UserSummary],
    summary="Search users",
    description="Search and list users with filters (Admin only)"
)
async def search_users(
    request: Request,
    search_params: UserSearch = Depends(),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: TokenData = Depends(require_admin_permissions)
) -> PaginatedResponse[UserSummary]:
    """Search users with filters and pagination.

    Args:
        request: FastAPI request object
        search_params: Search filter parameters
        page: Page number
        limit: Items per page
        current_user: Current authenticated admin user

    Returns:
        PaginatedResponse[UserSummary]: Paginated user list

    Raises:
        HTTPException: If search fails
    """
    log_api_request("GET", "/users", current_user.user_id, logger=logger)

    try:
        users, total = await user_service.search_users(
            search_params=search_params,
            page=page,
            limit=limit
        )

        log_api_response("GET", "/users", status.HTTP_200_OK, 0, logger=logger)

        return create_paginated_response(
            data=users,
            page=page,
            limit=limit,
            total=total,
            message=f"Found {total} users"
        )

    except TruScholarError as e:
        logger.error(f"User search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search users"
        )


@router.put(
    "/{user_id}",
    response_model=SuccessResponse[UserResponse],
    summary="Update user by ID",
    description="Update user by ID (Admin only)"
)
async def update_user_by_id(
    user_id: str,
    user_data: UserUpdate,
    request: Request,
    current_user: TokenData = Depends(require_admin_permissions)
) -> SuccessResponse[UserResponse]:
    """Update user by ID (Admin only).

    Args:
        user_id: User ID to update
        user_data: User update data
        request: FastAPI request object
        current_user: Current authenticated admin user

    Returns:
        SuccessResponse[UserResponse]: Updated user data

    Raises:
        HTTPException: If update fails
    """
    log_api_request("PUT", f"/users/{user_id}", current_user.user_id, logger=logger)

    try:
        user = await user_service.update_user(user_id, user_data)

        log_api_response("PUT", f"/users/{user_id}", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User updated successfully"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except ValidationError as e:
        logger.warning(f"User update validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"User update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete(
    "/{user_id}",
    response_model=SuccessResponse[dict],
    summary="Delete user by ID",
    description="Delete user by ID (Admin only)"
)
async def delete_user_by_id(
    user_id: str,
    request: Request,
    current_user: TokenData = Depends(require_admin_permissions),
    hard_delete: bool = Query(False, description="Whether to perform hard delete")
) -> SuccessResponse[dict]:
    """Delete user by ID (Admin only).

    Args:
        user_id: User ID to delete
        request: FastAPI request object
        current_user: Current authenticated admin user
        hard_delete: Whether to perform hard delete

    Returns:
        SuccessResponse[dict]: Deletion confirmation

    Raises:
        HTTPException: If deletion fails
    """
    log_api_request("DELETE", f"/users/{user_id}", current_user.user_id, logger=logger)

    try:
        success = await user_service.delete_user(user_id, soft_delete=not hard_delete)

        if success:
            log_api_response("DELETE", f"/users/{user_id}", status.HTTP_200_OK, 0, logger=logger)

            return create_success_response(
                data={"deleted": True, "user_id": user_id, "hard_delete": hard_delete},
                message="User deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    except TruScholarError as e:
        logger.error(f"User deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get(
    "/me/profile/completion",
    response_model=SuccessResponse[dict],
    summary="Get profile completion",
    description="Get current user's profile completion percentage"
)
async def get_profile_completion(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Get current user's profile completion percentage.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Profile completion data
    """
    log_api_request("GET", "/users/me/profile/completion", current_user.user_id, logger=logger)

    try:
        completion_percentage = await profile_service.update_profile_completion(current_user.user_id)

        log_api_response("GET", "/users/me/profile/completion", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data={
                "completion_percentage": completion_percentage,
                "is_complete": completion_percentage >= 90.0,
                "next_steps": _get_profile_next_steps(completion_percentage)
            },
            message="Profile completion calculated successfully"
        )

    except TruScholarError as e:
        logger.error(f"Profile completion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate profile completion"
        )


@router.get(
    "/me/stats",
    response_model=SuccessResponse[dict],
    summary="Get user statistics",
    description="Get current user's activity statistics"
)
async def get_user_stats(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Get current user's statistics.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: User statistics
    """
    log_api_request("GET", "/users/me/stats", current_user.user_id, logger=logger)

    try:
        user = await user_service.get_user_by_id(current_user.user_id)

        if not user or not user.stats:
            stats_data = {
                "total_tests_taken": 0,
                "tests_completed": 0,
                "tests_abandoned": 0,
                "completion_rate": 0.0,
                "average_test_duration_minutes": 0.0,
                "career_paths_viewed": 0,
                "reports_generated": 0,
                "last_test_date": None
            }
        else:
            stats_data = {
                "total_tests_taken": user.stats.total_tests_taken,
                "tests_completed": user.stats.tests_completed,
                "tests_abandoned": user.stats.tests_abandoned,
                "completion_rate": user.stats.completion_rate,
                "average_test_duration_minutes": user.stats.average_test_duration_minutes,
                "career_paths_viewed": user.stats.career_paths_viewed,
                "reports_generated": user.stats.reports_generated,
                "last_test_date": user.stats.last_test_date.isoformat() if user.stats.last_test_date else None
            }

        log_api_response("GET", "/users/me/stats", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=stats_data,
            message="User statistics retrieved successfully"
        )

    except TruScholarError as e:
        logger.error(f"Get user stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics"
        )


@router.post(
    "/me/activity",
    response_model=SuccessResponse[dict],
    summary="Update user activity",
    description="Update current user's last activity timestamp"
)
async def update_user_activity(
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Update current user's activity timestamp.

    Args:
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Activity update confirmation
    """
    log_api_request("POST", "/users/me/activity", current_user.user_id, logger=logger)

    try:
        await user_service.update_user_activity(current_user.user_id)

        log_api_response("POST", "/users/me/activity", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data={"activity_updated": True},
            message="User activity updated successfully"
        )

    except TruScholarError as e:
        logger.error(f"Update activity error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user activity"
        )


@router.get(
    "/phone/{phone}",
    response_model=SuccessResponse[UserResponse],
    summary="Get user by phone",
    description="Get user by phone number (Admin only)"
)
async def get_user_by_phone(
    phone: str,
    request: Request,
    current_user: TokenData = Depends(require_admin_permissions)
) -> SuccessResponse[UserResponse]:
    """Get user by phone number (Admin only).

    Args:
        phone: Phone number to search
        request: FastAPI request object
        current_user: Current authenticated admin user

    Returns:
        SuccessResponse[UserResponse]: User data

    Raises:
        HTTPException: If user not found
    """
    log_api_request("GET", f"/users/phone/{phone}", current_user.user_id, logger=logger)

    try:
        # Validate phone number format
        from src.utils.validators import validate_phone
        phone_result = validate_phone(phone)
        if not phone_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid phone number format"
            )

        user = await user_service.get_user_by_phone(phone_result.cleaned_value)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        log_api_response("GET", f"/users/phone/{phone}", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=user,
            message="User found successfully"
        )

    except ValidationError as e:
        logger.warning(f"Phone validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Get user by phone error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )


# Health check endpoint
@router.get(
    "/health",
    response_model=SuccessResponse[dict],
    summary="User service health check",
    description="Check if user service is healthy"
)
async def users_health_check() -> SuccessResponse[dict]:
    """Check user service health.

    Returns:
        SuccessResponse[dict]: Health status
    """
    return create_success_response(
        data={
            "service": "users",
            "status": "healthy",
            "version": "1.0.0"
        },
        message="User service is healthy"
    )


# Helper functions

def _get_profile_next_steps(completion_percentage: float) -> List[str]:
    """Get next steps for profile completion.

    Args:
        completion_percentage: Current completion percentage

    Returns:
        List[str]: List of next steps
    """
    if completion_percentage >= 90:
        return ["Profile is complete! Take your first RAISEC test."]

    next_steps = []

    if completion_percentage < 30:
        next_steps.extend([
            "Add your age and location",
            "Provide your education level",
            "Add your current occupation"
        ])
    elif completion_percentage < 60:
        next_steps.extend([
            "Complete your bio",
            "Add your interests",
            "Verify your email address"
        ])
    else:
        next_steps.extend([
            "Add more interests",
            "Complete remaining profile fields",
            "Take your first assessment"
        ])

    return next_steps


# Export router and dependencies
__all__ = [
    "router",
    "require_admin_permissions",
    "check_user_access",
]
