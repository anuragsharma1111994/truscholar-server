"""MongoDB database connection and management for TruScholar.

This module provides MongoDB connection management, database operations,
and helper functions for the application.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING, DESCENDING, IndexModel, errors

from src.core.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

T = TypeVar("T")


class MongoDB:
    """MongoDB connection manager and database operations."""

    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None
    _initialized: bool = False
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def connect(
        cls,
        url: Optional[str] = None,
        db_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Connect to MongoDB with connection pooling.

        Args:
            url: MongoDB connection URL
            db_name: Database name
            **kwargs: Additional connection parameters
        """
        async with cls._lock:
            if cls._initialized:
                logger.warning("MongoDB already connected")
                return

            try:
                # Use provided values or defaults from settings
                connection_url = url or settings.get_database_url()
                database_name = db_name or settings.MONGODB_DB_NAME

                # Connection parameters
                connection_params = {
                    "maxPoolSize": kwargs.get("max_pool_size", settings.MONGODB_MAX_POOL_SIZE),
                    "minPoolSize": kwargs.get("min_pool_size", settings.MONGODB_MIN_POOL_SIZE),
                    "maxIdleTimeMS": kwargs.get("max_idle_time_ms", settings.MONGODB_MAX_IDLE_TIME_MS),
                    "connectTimeoutMS": kwargs.get("connect_timeout_ms", settings.MONGODB_CONNECT_TIMEOUT_MS),
                    "serverSelectionTimeoutMS": kwargs.get("server_selection_timeout_ms", 5000),
                    "retryWrites": kwargs.get("retry_writes", True),
                    "retryReads": kwargs.get("retry_reads", True),
                    "w": kwargs.get("write_concern", "majority"),
                    "readPreference": kwargs.get("read_preference", "primaryPreferred"),
                }

                # Create motor client
                cls._client = AsyncIOMotorClient(connection_url, **connection_params)
                cls._database = cls._client[database_name]

                # Test connection
                await cls._client.server_info()

                cls._initialized = True
                logger.info(
                    f"MongoDB connected successfully",
                    extra={
                        "database": database_name,
                        "pool_size": connection_params["maxPoolSize"],
                    }
                )

            except Exception as e:
                cls._client = None
                cls._database = None
                cls._initialized = False
                logger.error(f"MongoDB connection failed: {str(e)}", exc_info=True)
                raise

    @classmethod
    async def disconnect(cls) -> None:
        """Disconnect from MongoDB."""
        async with cls._lock:
            if cls._client:
                try:
                    cls._client.close()
                    cls._client = None
                    cls._database = None
                    cls._initialized = False
                    logger.info("MongoDB disconnected successfully")
                except Exception as e:
                    logger.error(f"Error disconnecting from MongoDB: {str(e)}", exc_info=True)

    @classmethod
    async def ping(cls) -> bool:
        """Check if MongoDB connection is alive.

        Returns:
            bool: True if connection is alive
        """
        if not cls._initialized or not cls._client:
            return False

        try:
            await cls._client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {str(e)}")
            return False

    @classmethod
    def get_database(cls) -> Optional[AsyncIOMotorDatabase]:
        """Get the current database instance.

        Returns:
            Database instance or None
        """
        return cls._database

    @classmethod
    def get_collection(cls, name: str) -> Optional[AsyncIOMotorCollection]:
        """Get a collection by name.

        Args:
            name: Collection name

        Returns:
            Collection instance or None
        """
        if not cls._database:
            return None
        return cls._database[name]

    @classmethod
    async def create_index(
        cls,
        collection_name: str,
        keys: List[Tuple[str, int]],
        **kwargs: Any,
    ) -> bool:
        """Create an index on a collection.

        Args:
            collection_name: Name of the collection
            keys: List of (field, direction) tuples
            **kwargs: Additional index options

        Returns:
            bool: True if index was created
        """
        collection = cls.get_collection(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return False

        try:
            index_name = await collection.create_index(keys, **kwargs)
            logger.info(
                f"Created index {index_name} on {collection_name}",
                extra={"keys": keys, "options": kwargs}
            )
            return True
        except errors.DuplicateKeyError:
            logger.debug(f"Index already exists on {collection_name}")
            return False
        except Exception as e:
            logger.error(
                f"Failed to create index on {collection_name}: {str(e)}",
                extra={"keys": keys, "options": kwargs}
            )
            return False

    @classmethod
    async def create_indexes(cls, collection_name: str, indexes: List[IndexModel]) -> List[str]:
        """Create multiple indexes on a collection.

        Args:
            collection_name: Name of the collection
            indexes: List of IndexModel instances

        Returns:
            List of created index names
        """
        collection = cls.get_collection(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return []

        try:
            result = await collection.create_indexes(indexes)
            logger.info(
                f"Created {len(result)} indexes on {collection_name}",
                extra={"indexes": result}
            )
            return result
        except Exception as e:
            logger.error(
                f"Failed to create indexes on {collection_name}: {str(e)}",
                exc_info=True
            )
            return []

    @classmethod
    async def drop_index(cls, collection_name: str, index_name: str) -> bool:
        """Drop an index from a collection.

        Args:
            collection_name: Name of the collection
            index_name: Name of the index to drop

        Returns:
            bool: True if index was dropped
        """
        collection = cls.get_collection(collection_name)
        if not collection:
            return False

        try:
            await collection.drop_index(index_name)
            logger.info(f"Dropped index {index_name} from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop index {index_name}: {str(e)}")
            return False

    @classmethod
    async def get_collection_stats(cls, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection statistics or None
        """
        if not cls._database:
            return None

        try:
            stats = await cls._database.command("collStats", collection_name)
            return {
                "count": stats.get("count", 0),
                "size": stats.get("size", 0),
                "avgObjSize": stats.get("avgObjSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("nindexes", 0),
                "indexSize": stats.get("totalIndexSize", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {str(e)}")
            return None

    @classmethod
    async def get_database_stats(cls) -> Optional[Dict[str, Any]]:
        """Get database statistics.

        Returns:
            Database statistics or None
        """
        if not cls._database:
            return None

        try:
            stats = await cls._database.command("dbStats")
            return {
                "collections": stats.get("collections", 0),
                "objects": stats.get("objects", 0),
                "avgObjSize": stats.get("avgObjSize", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0),
                "indexSize": stats.get("indexSize", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return None


class MongoDBOperations:
    """Helper class for common MongoDB operations."""

    @staticmethod
    async def find_one_by_id(
        collection_name: str,
        document_id: str,
        projection: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a document by its ID.

        Args:
            collection_name: Name of the collection
            document_id: Document ID (string or ObjectId)
            projection: Fields to include/exclude

        Returns:
            Document or None
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return None

        try:
            # Convert string to ObjectId if needed
            if isinstance(document_id, str):
                document_id = ObjectId(document_id)

            document = await collection.find_one(
                {"_id": document_id},
                projection=projection
            )
            return document
        except Exception as e:
            logger.error(f"Error finding document by ID: {str(e)}")
            return None

    @staticmethod
    async def find_many(
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, int]] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        skip: int = 0,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """Find multiple documents with filtering and pagination.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            projection: Fields to include/exclude
            sort: Sort specification
            skip: Number of documents to skip
            limit: Maximum number of documents to return

        Returns:
            List of documents
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return []

        try:
            cursor = collection.find(
                filter_dict or {},
                projection=projection
            )

            if sort:
                cursor = cursor.sort(sort)
            if skip > 0:
                cursor = cursor.skip(skip)
            if limit > 0:
                cursor = cursor.limit(limit)

            documents = await cursor.to_list(length=None)
            return documents
        except Exception as e:
            logger.error(f"Error finding documents: {str(e)}")
            return []

    @staticmethod
    async def count_documents(
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count documents matching a filter.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter

        Returns:
            Document count
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return 0

        try:
            count = await collection.count_documents(filter_dict or {})
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0

    @staticmethod
    async def insert_one(
        collection_name: str,
        document: Dict[str, Any],
    ) -> Optional[str]:
        """Insert a single document.

        Args:
            collection_name: Name of the collection
            document: Document to insert

        Returns:
            Inserted document ID or None
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return None

        try:
            # Add timestamps if not present
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()
            if "updated_at" not in document:
                document["updated_at"] = datetime.utcnow()

            result = await collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return None

    @staticmethod
    async def insert_many(
        collection_name: str,
        documents: List[Dict[str, Any]],
    ) -> List[str]:
        """Insert multiple documents.

        Args:
            collection_name: Name of the collection
            documents: List of documents to insert

        Returns:
            List of inserted document IDs
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection or not documents:
            return []

        try:
            # Add timestamps to all documents
            now = datetime.utcnow()
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = now
                if "updated_at" not in doc:
                    doc["updated_at"] = now

            result = await collection.insert_many(documents)
            return [str(id_) for id_ in result.inserted_ids]
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            return []

    @staticmethod
    async def update_one(
        collection_name: str,
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        """Update a single document.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            update_dict: Update operations
            upsert: Whether to insert if not found

        Returns:
            bool: True if document was modified
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return False

        try:
            # Add updated_at timestamp
            if "$set" in update_dict:
                update_dict["$set"]["updated_at"] = datetime.utcnow()
            else:
                update_dict["$set"] = {"updated_at": datetime.utcnow()}

            result = await collection.update_one(
                filter_dict,
                update_dict,
                upsert=upsert
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    @staticmethod
    async def update_many(
        collection_name: str,
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any],
    ) -> int:
        """Update multiple documents.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter
            update_dict: Update operations

        Returns:
            Number of modified documents
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return 0

        try:
            # Add updated_at timestamp
            if "$set" in update_dict:
                update_dict["$set"]["updated_at"] = datetime.utcnow()
            else:
                update_dict["$set"] = {"updated_at": datetime.utcnow()}

            result = await collection.update_many(filter_dict, update_dict)
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            return 0

    @staticmethod
    async def delete_one(
        collection_name: str,
        filter_dict: Dict[str, Any],
    ) -> bool:
        """Delete a single document.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter

        Returns:
            bool: True if document was deleted
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return False

        try:
            result = await collection.delete_one(filter_dict)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    @staticmethod
    async def delete_many(
        collection_name: str,
        filter_dict: Dict[str, Any],
    ) -> int:
        """Delete multiple documents.

        Args:
            collection_name: Name of the collection
            filter_dict: Query filter

        Returns:
            Number of deleted documents
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return 0

        try:
            result = await collection.delete_many(filter_dict)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return 0

    @staticmethod
    async def aggregate(
        collection_name: str,
        pipeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute an aggregation pipeline.

        Args:
            collection_name: Name of the collection
            pipeline: Aggregation pipeline stages

        Returns:
            List of aggregation results
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection:
            return []

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            return results
        except Exception as e:
            logger.error(f"Error executing aggregation: {str(e)}")
            return []

    @staticmethod
    async def bulk_write(
        collection_name: str,
        operations: List[Dict[str, Any]],
        ordered: bool = True,
    ) -> Dict[str, int]:
        """Execute bulk write operations.

        Args:
            collection_name: Name of the collection
            operations: List of write operations
            ordered: Whether to execute operations in order

        Returns:
            Dictionary with operation counts
        """
        collection = MongoDB.get_collection(collection_name)
        if not collection or not operations:
            return {
                "inserted_count": 0,
                "updated_count": 0,
                "deleted_count": 0,
                "errors": 0,
            }

        try:
            result = await collection.bulk_write(operations, ordered=ordered)
            return {
                "inserted_count": result.inserted_count,
                "updated_count": result.modified_count,
                "deleted_count": result.deleted_count,
                "errors": 0,
            }
        except errors.BulkWriteError as e:
            logger.error(f"Bulk write error: {str(e)}")
            return {
                "inserted_count": e.details.get("nInserted", 0),
                "updated_count": e.details.get("nModified", 0),
                "deleted_count": e.details.get("nRemoved", 0),
                "errors": len(e.details.get("writeErrors", [])),
            }
        except Exception as e:
            logger.error(f"Error executing bulk write: {str(e)}")
            return {
                "inserted_count": 0,
                "updated_count": 0,
                "deleted_count": 0,
                "errors": 1,
            }


class MongoDBTransaction:
    """Context manager for MongoDB transactions."""

    def __init__(self, session=None):
        """Initialize transaction context.

        Args:
            session: Existing session or None to create new
        """
        self.session = session
        self.owns_session = session is None

    async def __aenter__(self):
        """Enter transaction context."""
        if self.owns_session:
            self.session = await MongoDB._client.start_session()
        await self.session.start_transaction()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context."""
        if exc_type:
            await self.session.abort_transaction()
        else:
            await self.session.commit_transaction()

        if self.owns_session:
            await self.session.end_session()


# Helper functions for database access

async def get_database() -> Optional[AsyncIOMotorDatabase]:
    """Get the current database instance.

    Returns:
        Database instance or None
    """
    return MongoDB.get_database()


async def get_collection(name: str) -> Optional[AsyncIOMotorCollection]:
    """Get a collection by name.

    Args:
        name: Collection name

    Returns:
        Collection instance or None
    """
    return MongoDB.get_collection(name)


@asynccontextmanager
async def get_session():
    """Get a database session for transactions.

    Yields:
        Database session
    """
    if not MongoDB._client:
        raise RuntimeError("MongoDB not connected")

    session = await MongoDB._client.start_session()
    try:
        yield session
    finally:
        await session.end_session()


# Export main classes and functions
__all__ = [
    "MongoDB",
    "MongoDBOperations",
    "MongoDBTransaction",
    "get_database",
    "get_collection",
    "get_session",
]
