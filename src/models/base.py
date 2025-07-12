"""Base model classes and utilities for MongoDB documents.

This module provides base classes and utilities for all MongoDB models in the
TruScholar application, including common fields, validation, and serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from src.utils.datetime_utils import utc_now


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models.

    This class enables proper serialization and validation of MongoDB ObjectIds
    in Pydantic models.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        """Get the Pydantic core schema for PyObjectId."""
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(cls.validate),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x), return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def validate(cls, value: Any) -> ObjectId:
        """Validate and convert a value to ObjectId.

        Args:
            value: The value to validate

        Returns:
            ObjectId: A valid ObjectId instance

        Raises:
            ValueError: If the value is not a valid ObjectId
        """
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, str) and ObjectId.is_valid(value):
            return ObjectId(value)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: Any
    ) -> JsonSchemaValue:
        """Get JSON schema for PyObjectId."""
        return handler(core_schema.str_schema())


class BaseDocument(BaseModel):
    """Base model for all MongoDB documents.

    Provides common fields and configuration for all database models.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
            }
        },
    )

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def __init__(self, **data: Any) -> None:
        """Initialize a BaseDocument instance."""
        if "id" not in data and "_id" not in data:
            data["_id"] = ObjectId()
        super().__init__(**data)

    @field_validator("id", mode="before")
    @classmethod
    def validate_object_id(cls, value: Any) -> Optional[PyObjectId]:
        """Validate ObjectId field."""
        if value is None:
            return None
        return PyObjectId.validate(value)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        """Convert model to dictionary.

        Args:
            **kwargs: Additional arguments for model_dump

        Returns:
            Dict[str, Any]: Dictionary representation of the model
        """
        data = self.model_dump(by_alias=True, **kwargs)
        # Ensure _id is string in output
        if "_id" in data and data["_id"] is not None:
            data["_id"] = str(data["_id"])
        return data

    def to_json(self, **kwargs: Any) -> str:
        """Convert model to JSON string.

        Args:
            **kwargs: Additional arguments for model_dump_json

        Returns:
            str: JSON string representation of the model
        """
        return self.model_dump_json(by_alias=True, **kwargs)

    @classmethod
    def from_dict(cls: Type["T"], data: Dict[str, Any]) -> "T":
        """Create model instance from dictionary.

        Args:
            data: Dictionary containing model data

        Returns:
            Model instance
        """
        return cls(**data)

    def update_timestamps(self) -> None:
        """Update the updated_at timestamp to current time."""
        self.updated_at = utc_now()

    def create_index_keys(self) -> List[tuple]:
        """Return list of index keys for this model.

        Override in subclasses to define model-specific indexes.

        Returns:
            List[tuple]: List of index key specifications
        """
        return []


T = TypeVar("T", bound=BaseDocument)


class EmbeddedDocument(BaseModel):
    """Base model for embedded documents (subdocuments).

    Used for documents that are embedded within other documents rather than
    stored in their own collection.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
    )


class TimestampedDocument(BaseDocument):
    """Base model with automatic timestamp management.

    Extends BaseDocument with additional timestamp tracking for soft deletes
    and other time-based operations.
    """

    deleted_at: Optional[datetime] = Field(default=None)
    is_active: bool = Field(default=True)

    def soft_delete(self) -> None:
        """Mark document as deleted without removing from database."""
        self.deleted_at = utc_now()
        self.is_active = False
        self.update_timestamps()

    def restore(self) -> None:
        """Restore a soft-deleted document."""
        self.deleted_at = None
        self.is_active = True
        self.update_timestamps()

    @property
    def is_deleted(self) -> bool:
        """Check if document is soft-deleted."""
        return self.deleted_at is not None


class VersionedDocument(TimestampedDocument):
    """Base model with version tracking.

    Extends TimestampedDocument with version tracking for optimistic
    concurrency control.
    """

    version: int = Field(default=1, ge=1)
    version_history: List[Dict[str, Any]] = Field(default_factory=list)

    def increment_version(self) -> None:
        """Increment document version and save current state to history."""
        # Save current state to history before incrementing
        current_state = self.to_dict(exclude={"version_history"})
        self.version_history.append(
            {
                "version": self.version,
                "data": current_state,
                "timestamp": utc_now(),
            }
        )
        self.version += 1
        self.update_timestamps()

    def get_version(self, version_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific version from history.

        Args:
            version_number: The version number to retrieve

        Returns:
            Optional[Dict[str, Any]]: The version data if found
        """
        for version in self.version_history:
            if version["version"] == version_number:
                return version["data"]
        return None


class PaginationParams(BaseModel):
    """Parameters for pagination queries."""

    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)
    sort_by: str = Field(default="created_at")
    sort_order: int = Field(default=-1, ge=-1, le=1)  # -1 for desc, 1 for asc

    @property
    def skip(self) -> int:
        """Calculate skip value for MongoDB queries."""
        return (self.page - 1) * self.limit

    def to_mongo_options(self) -> Dict[str, Any]:
        """Convert to MongoDB query options.

        Returns:
            Dict[str, Any]: MongoDB query options
        """
        return {
            "skip": self.skip,
            "limit": self.limit,
            "sort": [(self.sort_by, self.sort_order)],
        }


class BulkWriteResult(BaseModel):
    """Result of a bulk write operation."""

    inserted_count: int = 0
    updated_count: int = 0
    deleted_count: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if bulk operation was successful."""
        return len(self.errors) == 0

    @property
    def total_count(self) -> int:
        """Get total number of affected documents."""
        return self.inserted_count + self.updated_count + self.deleted_count


# Metadata models for API responses
class CollectionMetadata(BaseModel):
    """Metadata for collection responses."""

    total_count: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool


class ResponseMetadata(BaseModel):
    """General response metadata."""

    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
    version: str = "1.0.0"
    status: str = "success"


# Type aliases for common field types
ObjectIdStr = str  # String representation of ObjectId
ISODateTime = str  # ISO format datetime string
