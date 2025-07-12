"""User management service for TruScholar application.

This module provides business logic for user operations including CRUD operations,
profile management, preferences, statistics, and user search functionality.
"""

from typing import Dict, List, Optional, Tuple

from bson import ObjectId

from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.models.user import User, UserProfile, UserPreferences, UserStats
from src.schemas.user_schemas import (
    UserCreate,
    UserProfileUpdate,
    UserPreferencesUpdate,
    UserResponse,
    UserSearch,
    UserSummary,
    UserUpdate,
)
from src.utils.constants import AgeGroup, UserAccountType
from src.utils.datetime_utils import utc_now
from src.utils.exceptions import (
    BusinessLogicError,
    ResourceNotFoundError,
    ValidationError,
)
from src.utils.helper import mask_phone_number
from src.utils.logger import get_logger
from src.utils.validators import validate_age, validate_email, validate_name, validate_phone

logger = get_logger(__name__)


class UserService:
    """Service for user management operations."""

    def __init__(self):
        """Initialize user service."""
        self.collection = "users"
        self.cache_ttl = 3600  # 1 hour

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user.

        Args:
            user_data: User creation data

        Returns:
            UserResponse: Created user data

        Raises:
            ValidationError: If user data is invalid
            BusinessLogicError: If user already exists
        """
        # Check if user already exists
        existing_user = await MongoDB.find_document(
            self.collection,
            {"phone": user_data.phone}
        )

        if existing_user:
            logger.warning(
                "Attempt to create duplicate user",
                extra={"phone": mask_phone_number(user_data.phone)}
            )
            raise BusinessLogicError(
                "User with this phone number already exists",
                operation="create_user",
                resource_id=user_data.phone
            )

        # Create user model
        user = User(
            phone=user_data.phone,
            name=user_data.name,
            email=user_data.email,
            account_type=user_data.account_type,
            is_active=user_data.is_active,
            terms_accepted=user_data.terms_accepted,
            privacy_accepted=user_data.privacy_accepted,
            data_consent=user_data.data_consent,
            terms_accepted_at=utc_now() if user_data.terms_accepted else None,
            privacy_accepted_at=utc_now() if user_data.privacy_accepted else None,
            tags=user_data.tags,
            notes=user_data.notes,
            referral_code=user_data.referral_code
        )

        # Add profile if provided
        if user_data.profile:
            user.profile = UserProfile(**user_data.profile.model_dump())

        # Add preferences if provided
        if user_data.preferences:
            user.preferences = UserPreferences(**user_data.preferences.model_dump())

        # Save to database
        user_doc = await MongoDB.create_document(self.collection, user.to_dict())
        user_id = str(user_doc["_id"])

        # Cache user data
        await self._cache_user(user_id, user_doc)

        logger.info(
            "User created successfully",
            extra={
                "user_id": user_id,
                "phone": mask_phone_number(user.phone),
                "name": user.name
            }
        )

        return await self._convert_to_response(user_doc)

    async def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID.

        Args:
            user_id: User ID

        Returns:
            UserResponse or None: User data if found
        """
        # Try cache first
        cached_user = await RedisClient.get_json(f"user:{user_id}")
        if cached_user:
            return await self._convert_to_response(cached_user)

        # Get from database
        user_doc = await MongoDB.find_document(
            self.collection,
            {"_id": ObjectId(user_id)}
        )

        if not user_doc:
            return None

        # Cache user data
        await self._cache_user(user_id, user_doc)

        return await self._convert_to_response(user_doc)

    async def get_user_by_phone(self, phone: str) -> Optional[UserResponse]:
        """Get user by phone number.

        Args:
            phone: Phone number

        Returns:
            UserResponse or None: User data if found
        """
        user_doc = await MongoDB.find_document(
            self.collection,
            {"phone": phone}
        )

        if not user_doc:
            return None

        # Cache user data
        user_id = str(user_doc["_id"])
        await self._cache_user(user_id, user_doc)

        return await self._convert_to_response(user_doc)

    async def update_user(self, user_id: str, user_data: UserUpdate) -> UserResponse:
        """Update user information.

        Args:
            user_id: User ID
            user_data: User update data

        Returns:
            UserResponse: Updated user data

        Raises:
            ResourceNotFoundError: If user not found
            ValidationError: If update data is invalid
        """
        # Check if user exists
        existing_user = await MongoDB.find_document(
            self.collection,
            {"_id": ObjectId(user_id)}
        )

        if not existing_user:
            raise ResourceNotFoundError(
                "User not found",
                resource_type="user",
                resource_id=user_id
            )

        # Prepare update data
        update_data = {}

        if user_data.name is not None:
            update_data["name"] = user_data.name

        if user_data.email is not None:
            update_data["email"] = user_data.email

        # Update timestamps
        update_data["updated_at"] = utc_now()

        # Perform update
        updated_doc = await MongoDB.update_document(
            self.collection,
            {"_id": ObjectId(user_id)},
            update_data
        )

        if not updated_doc:
            raise BusinessLogicError(
                "Failed to update user",
                operation="update_user",
                resource_id=user_id
            )

        # Clear cache
        await self._clear_user_cache(user_id)

        logger.info(
            "User updated successfully",
            extra={"user_id": user_id, "updated_fields": list(update_data.keys())}
        )

        return await self._convert_to_response(updated_doc)

    async def update_user_profile(self, user_id: str, profile_data: UserProfileUpdate) -> UserResponse:
        """Update user profile.

        Args:
            user_id: User ID
            profile_data: Profile update data

        Returns:
            UserResponse: Updated user data

        Raises:
            ResourceNotFoundError: If user not found
        """
        # Check if user exists
        existing_user = await MongoDB.find_document(
            self.collection,
            {"_id": ObjectId(user_id)}
        )

        if not existing_user:
            raise ResourceNotFoundError(
                "User not found",
                resource_type="user",
                resource_id=user_id
            )

        # Prepare profile update data
        profile_updates = {}

        for field, value in profile_data.model_dump(exclude_unset=True).items():
            if value is not None:
                profile_updates[f"profile.{field}"] = value

        # Set age group if age is updated
        if profile_data.age is not None:
            try:
                from src.utils.constants import get_age_group_from_age
                age_group = get_age_group_from_age(profile_data.age)
                profile_updates["profile.age_group"] = age_group.value
            except ValueError:
                # Age outside supported range
                profile_updates["profile.age_group"] = None

        # Update timestamps
        profile_updates["updated_at"] = utc_now()

        # Perform update
        updated_doc = await MongoDB.update_document(
            self.collection,
            {"_id": ObjectId(user_id)},
            profile_updates
        )

        if not updated_doc:
            raise BusinessLogicError(
                "Failed to update user profile",
                operation="update_user_profile",
                resource_id=user_id
            )

        # Clear cache
        await self._clear_user_cache(user_id)

        logger.info(
            "User profile updated successfully",
            extra={"user_id": user_id, "updated_fields": list(profile_data.model_dump(exclude_unset=True).keys())}
        )

        return await self._convert_to_response(updated_doc)

    async def update_user_preferences(self, user_id: str, preferences_data: UserPreferencesUpdate) -> UserResponse:
        """Update user preferences.

        Args:
            user_id: User ID
            preferences_data: Preferences update data

        Returns:
            UserResponse: Updated user data

        Raises:
            ResourceNotFoundError: If user not found
        """
        # Check if user exists
        existing_user = await MongoDB.find_document(
            self.collection,
            {"_id": ObjectId(user_id)}
        )

        if not existing_user:
            raise ResourceNotFoundError(
                "User not found",
                resource_type="user",
                resource_id=user_id
            )

        # Prepare preferences update data
        preferences_updates = {}

        for field, value in preferences_data.model_dump(exclude_unset=True).items():
            if value is not None:
                preferences_updates[f"preferences.{field}"] = value

        # Update timestamps
        preferences_updates["updated_at"] = utc_now()

        # Perform update
        updated_doc = await MongoDB.update_document(
            self.collection,
            {"_id": ObjectId(user_id)},
            preferences_updates
        )

        if not updated_doc:
            raise BusinessLogicError(
                "Failed to update user preferences",
                operation="update_user_preferences",
                resource_id=user_id
            )

        # Clear cache
        await self._clear_user_cache(user_id)

        logger.info(
            "User preferences updated successfully",
            extra={"user_id": user_id}
        )

        return await self._convert_to_response(updated_doc)

    async def delete_user(self, user_id: str, soft_delete: bool = True) -> bool:
        """Delete user account.

        Args:
            user_id: User ID
            soft_delete: Whether to perform soft delete (default) or hard delete

        Returns:
            bool: True if user was deleted

        Raises:
            ResourceNotFoundError: If user not found
        """
        # Check if user exists
        existing_user = await MongoDB.find_document(
            self.collection,
            {"_id": ObjectId(user_id)}
        )

        if not existing_user:
            raise ResourceNotFoundError(
                "User not found",
                resource_type="user",
                resource_id=user_id
            )

        if soft_delete:
            # Soft delete - mark as inactive and set deleted timestamp
            update_data = {
                "is_active": False,
                "deleted_at": utc_now(),
                "updated_at": utc_now()
            }

            result = await MongoDB.update_document(
                self.collection,
                {"_id": ObjectId(user_id)},
                update_data
            )

            success = result is not None
        else:
            # Hard delete - remove from database
            result = await MongoDB.delete_document(
                self.collection,
                {"_id": ObjectId(user_id)}
            )

            success = result > 0

        if success:
            # Clear cache
            await self._clear_user_cache(user_id)

            # TODO: Invalidate all user sessions
            # TODO: Clean up related data (tests, reports, etc.)

            logger.info(
                "User deleted successfully",
                extra={
                    "user_id": user_id,
                    "soft_delete": soft_delete,
                    "phone": mask_phone_number(existing_user.get("phone", ""))
                }
            )

        return success

    async def search_users(
        self,
        search_params: UserSearch,
        page: int = 1,
        limit: int = 20
    ) -> Tuple[List[UserSummary], int]:
        """Search users with filters and pagination.

        Args:
            search_params: Search parameters
            page: Page number
            limit: Items per page

        Returns:
            Tuple[List[UserSummary], int]: Users and total count
        """
        # Build query
        query = {}

        # Text search
        if search_params.search:
            query["$or"] = [
                {"name": {"$regex": search_params.search, "$options": "i"}},
                {"email": {"$regex": search_params.search, "$options": "i"}}
            ]

        # Specific field filters
        if search_params.name:
            query["name"] = {"$regex": search_params.name, "$options": "i"}

        if search_params.phone:
            query["phone"] = {"$regex": search_params.phone}

        if search_params.email:
            query["email"] = {"$regex": search_params.email, "$options": "i"}

        if search_params.account_type:
            query["account_type"] = search_params.account_type.value

        if search_params.is_active is not None:
            query["is_active"] = search_params.is_active

        if search_params.is_verified is not None:
            query["is_verified"] = search_params.is_verified

        if search_params.age_group:
            query["profile.age_group"] = search_params.age_group.value

        if search_params.location_city:
            query["profile.location_city"] = {"$regex": search_params.location_city, "$options": "i"}

        if search_params.location_state:
            query["profile.location_state"] = {"$regex": search_params.location_state, "$options": "i"}

        # Date filters
        date_filters = {}

        if search_params.created_after:
            date_filters["$gte"] = search_params.created_after

        if search_params.created_before:
            date_filters["$lte"] = search_params.created_before

        if date_filters:
            query["created_at"] = date_filters

        # Last active filters
        if search_params.last_active_after or search_params.last_active_before:
            last_active_filters = {}

            if search_params.last_active_after:
                last_active_filters["$gte"] = search_params.last_active_after

            if search_params.last_active_before:
                last_active_filters["$lte"] = search_params.last_active_before

            query["last_active"] = last_active_filters

        # Test completion filter
        if search_params.has_completed_tests is not None:
            if search_params.has_completed_tests:
                query["stats.tests_completed"] = {"$gt": 0}
            else:
                query["$or"] = [
                    {"stats.tests_completed": {"$exists": False}},
                    {"stats.tests_completed": 0}
                ]

        # Get total count
        total = await MongoDB.count_documents(self.collection, query)

        # Get paginated results
        skip = (page - 1) * limit
        users = await MongoDB.find_documents(
            self.collection,
            query,
            skip=skip,
            limit=limit,
            sort=[("created_at", -1)]
        )

        # Convert to summaries
        user_summaries = []
        for user_doc in users:
            summary = UserSummary(
                id=str(user_doc["_id"]),
                name=user_doc["name"],
                phone=mask_phone_number(user_doc["phone"]),
                account_type=UserAccountType(user_doc["account_type"]),
                is_active=user_doc["is_active"],
                last_active=user_doc["last_active"],
                tests_completed=user_doc.get("stats", {}).get("tests_completed", 0)
            )
            user_summaries.append(summary)

        logger.info(
            "User search completed",
            extra={
                "query_filters": len(query),
                "results_found": len(user_summaries),
                "total_count": total,
                "page": page
            }
        )

        return user_summaries, total

    async def update_user_activity(self, user_id: str) -> None:
        """Update user's last activity timestamp.

        Args:
            user_id: User ID
        """
        await MongoDB.update_document(
            self.collection,
            {"_id": ObjectId(user_id)},
            {"last_active": utc_now()}
        )

        # Clear cache to ensure fresh data
        await self._clear_user_cache(user_id)

    async def _convert_to_response(self, user_doc: Dict) -> UserResponse:
        """Convert user document to response schema.

        Args:
            user_doc: User document from database

        Returns:
            UserResponse: Formatted response
        """
        # Convert ObjectId to string
        user_doc["id"] = str(user_doc["_id"])

        # Mask phone number
        user_doc["phone"] = mask_phone_number(user_doc["phone"])

        # Count related documents
        user_doc["test_ids_count"] = len(user_doc.get("test_ids", []))
        user_doc["report_ids_count"] = len(user_doc.get("report_ids", []))

        return UserResponse(**user_doc)

    async def _cache_user(self, user_id: str, user_doc: Dict) -> None:
        """Cache user data in Redis.

        Args:
            user_id: User ID
            user_doc: User document
        """
        try:
            # Prepare data for caching (remove ObjectId)
            cache_data = user_doc.copy()
            cache_data["_id"] = str(cache_data["_id"])

            await RedisClient.set_json(
                f"user:{user_id}",
                cache_data,
                ttl=self.cache_ttl
            )
        except Exception as e:
            logger.warning(
                "Failed to cache user data",
                extra={"user_id": user_id, "error": str(e)}
            )

    async def _clear_user_cache(self, user_id: str) -> None:
        """Clear user cache.

        Args:
            user_id: User ID
        """
        try:
            await RedisClient.delete(f"user:{user_id}")
        except Exception as e:
            logger.warning(
                "Failed to clear user cache",
                extra={"user_id": user_id, "error": str(e)}
            )


class UserProfileService:
    """Service for user profile specific operations."""

    def __init__(self):
        """Initialize user profile service."""
        self.user_service = UserService()

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile data.

        Args:
            user_id: User ID

        Returns:
            UserProfile or None: User profile if found
        """
        user = await self.user_service.get_user_by_id(user_id)
        if user and user.profile:
            return user.profile
        return None

    async def update_profile_completion(self, user_id: str) -> float:
        """Calculate and update profile completion percentage.

        Args:
            user_id: User ID

        Returns:
            float: Profile completion percentage
        """
        user = await self.user_service.get_user_by_id(user_id)
        if not user:
            return 0.0

        # Calculate completion based on filled fields
        total_fields = 10  # Total profile fields
        completed_fields = 0

        if user.profile:
            if user.profile.age:
                completed_fields += 1
            if user.profile.education_level:
                completed_fields += 1
            if user.profile.current_occupation:
                completed_fields += 1
            if user.profile.location_city:
                completed_fields += 1
            if user.profile.location_state:
                completed_fields += 1
            if user.profile.bio:
                completed_fields += 1
            if user.profile.interests:
                completed_fields += 1

        if user.email:
            completed_fields += 1

        # Always have phone and name
        completed_fields += 2

        completion_percentage = (completed_fields / total_fields) * 100

        logger.info(
            "Profile completion calculated",
            extra={
                "user_id": user_id,
                "completion_percentage": completion_percentage,
                "completed_fields": completed_fields
            }
        )

        return completion_percentage


class UserStatsService:
    """Service for user statistics and metrics."""

    def __init__(self):
        """Initialize user stats service."""
        self.user_service = UserService()

    async def update_test_stats(
        self,
        user_id: str,
        test_completed: bool = False,
        test_abandoned: bool = False,
        duration_minutes: Optional[float] = None
    ) -> None:
        """Update user test statistics.

        Args:
            user_id: User ID
            test_completed: Whether test was completed
            test_abandoned: Whether test was abandoned
            duration_minutes: Test duration in minutes
        """
        user_doc = await MongoDB.find_document(
            "users",
            {"_id": ObjectId(user_id)}
        )

        if not user_doc:
            return

        # Get current stats or create new
        current_stats = user_doc.get("stats", {})

        # Update counters
        if test_completed:
            current_stats["tests_completed"] = current_stats.get("tests_completed", 0) + 1
            current_stats["last_test_date"] = utc_now()

            # Update average duration
            if duration_minutes:
                current_completed = current_stats["tests_completed"]
                current_avg = current_stats.get("average_test_duration_minutes", 0.0)

                # Calculate new average
                total_duration = (current_avg * (current_completed - 1)) + duration_minutes
                new_avg = total_duration / current_completed
                current_stats["average_test_duration_minutes"] = round(new_avg, 2)

        if test_abandoned:
            current_stats["tests_abandoned"] = current_stats.get("tests_abandoned", 0) + 1

        # Update total tests taken
        current_stats["total_tests_taken"] = (
            current_stats.get("tests_completed", 0) +
            current_stats.get("tests_abandoned", 0)
        )

        # Update in database
        await MongoDB.update_document(
            "users",
            {"_id": ObjectId(user_id)},
            {"stats": current_stats, "updated_at": utc_now()}
        )

        # Clear cache
        await self.user_service._clear_user_cache(user_id)

        logger.info(
            "User test stats updated",
            extra={
                "user_id": user_id,
                "test_completed": test_completed,
                "test_abandoned": test_abandoned,
                "duration_minutes": duration_minutes
            }
        )

    async def increment_career_views(self, user_id: str) -> None:
        """Increment user's career paths viewed count.

        Args:
            user_id: User ID
        """
        await MongoDB.update_document(
            "users",
            {"_id": ObjectId(user_id)},
            {
                "$inc": {"stats.career_paths_viewed": 1},
                "$set": {"updated_at": utc_now()}
            }
        )

        # Clear cache
        await self.user_service._clear_user_cache(user_id)

    async def increment_reports_generated(self, user_id: str) -> None:
        """Increment user's reports generated count.

        Args:
            user_id: User ID
        """
        await MongoDB.update_document(
            "users",
            {"_id": ObjectId(user_id)},
            {
                "$inc": {"stats.reports_generated": 1},
                "$set": {"updated_at": utc_now()}
            }
        )

        # Clear cache
        await self.user_service._clear_user_cache(user_id)


# Export service classes
__all__ = [
    "UserService",
    "UserProfileService",
    "UserStatsService",
]
