"""Cache key definitions for TruScholar application.

This module defines all cache key patterns and helper functions for generating
consistent cache keys across the application.
"""

from typing import Optional
from enum import Enum


class CacheNamespace(str, Enum):
    """Cache namespaces for different data types."""
    
    USER = "user"
    TEST = "test"
    QUESTION = "question"
    ANSWER = "answer"
    REPORT = "report"
    CAREER = "career"
    SESSION = "session"
    RATE_LIMIT = "rate_limit"
    PROMPT = "prompt"
    LLM = "llm"


class CacheKeys:
    """Cache key generation and management."""
    
    # Separator for key components
    SEPARATOR = ":"
    
    # Key patterns
    PATTERNS = {
        # User related
        "user_by_id": "{namespace}:{user_id}",
        "user_by_phone": "{namespace}:phone:{phone}",
        "user_sessions": "{namespace}:{user_id}:sessions",
        "user_tests": "{namespace}:{user_id}:tests",
        "user_active_test": "{namespace}:{user_id}:active_test",
        
        # Test related
        "test_by_id": "{namespace}:{test_id}",
        "test_questions": "{namespace}:{test_id}:questions",
        "test_answers": "{namespace}:{test_id}:answers",
        "test_progress": "{namespace}:{test_id}:progress",
        "test_score": "{namespace}:{test_id}:score",
        "test_lock": "{namespace}:{test_id}:lock",
        
        # Question related
        "question_by_id": "{namespace}:{question_id}",
        "question_by_test": "{namespace}:test:{test_id}:q:{question_number}",
        "question_generation": "{namespace}:generation:{age_group}:{question_type}",
        
        # Answer related
        "answer_by_id": "{namespace}:{answer_id}",
        "answer_by_question": "{namespace}:test:{test_id}:question:{question_id}",
        
        # Report related
        "report_by_test": "{namespace}:{test_id}",
        "report_generation_lock": "{namespace}:{test_id}:generation:lock",
        
        # Career related
        "career_recommendations": "{namespace}:test:{test_id}:recommendations",
        "career_validation_questions": "{namespace}:raisec:{raisec_code}:validation",
        
        # Session related
        "session_token": "{namespace}:{token}",
        "session_user": "{namespace}:user:{user_id}",
        
        # Rate limiting
        "rate_limit_api": "{namespace}:api:{user_id}:{endpoint}",
        "rate_limit_test": "{namespace}:test:create:{user_id}",
        "rate_limit_llm": "{namespace}:llm:{user_id}",
        
        # Prompt caching
        "prompt_version": "{namespace}:version:{version}",
        "prompt_by_type": "{namespace}:type:{prompt_type}:{version}",
        
        # LLM response caching
        "llm_response": "{namespace}:response:{hash_key}",
        "llm_generation": "{namespace}:generation:{context_hash}",
    }
    
    @classmethod
    def user_by_id(cls, user_id: str) -> str:
        """Get cache key for user by ID."""
        return cls.PATTERNS["user_by_id"].format(
            namespace=CacheNamespace.USER,
            user_id=user_id
        )
    
    @classmethod
    def user_by_phone(cls, phone: str) -> str:
        """Get cache key for user by phone number."""
        return cls.PATTERNS["user_by_phone"].format(
            namespace=CacheNamespace.USER,
            phone=phone
        )
    
    @classmethod
    def user_sessions(cls, user_id: str) -> str:
        """Get cache key for user sessions."""
        return cls.PATTERNS["user_sessions"].format(
            namespace=CacheNamespace.USER,
            user_id=user_id
        )
    
    @classmethod
    def user_tests(cls, user_id: str) -> str:
        """Get cache key for user's test list."""
        return cls.PATTERNS["user_tests"].format(
            namespace=CacheNamespace.USER,
            user_id=user_id
        )
    
    @classmethod
    def user_active_test(cls, user_id: str) -> str:
        """Get cache key for user's active test."""
        return cls.PATTERNS["user_active_test"].format(
            namespace=CacheNamespace.USER,
            user_id=user_id
        )
    
    @classmethod
    def test_by_id(cls, test_id: str) -> str:
        """Get cache key for test by ID."""
        return cls.PATTERNS["test_by_id"].format(
            namespace=CacheNamespace.TEST,
            test_id=test_id
        )
    
    @classmethod
    def test_questions(cls, test_id: str) -> str:
        """Get cache key for test questions."""
        return cls.PATTERNS["test_questions"].format(
            namespace=CacheNamespace.TEST,
            test_id=test_id
        )
    
    @classmethod
    def test_answers(cls, test_id: str) -> str:
        """Get cache key for test answers."""
        return cls.PATTERNS["test_answers"].format(
            namespace=CacheNamespace.TEST,
            test_id=test_id
        )
    
    @classmethod
    def test_progress(cls, test_id: str) -> str:
        """Get cache key for test progress."""
        return cls.PATTERNS["test_progress"].format(
            namespace=CacheNamespace.TEST,
            test_id=test_id
        )
    
    @classmethod
    def test_score(cls, test_id: str) -> str:
        """Get cache key for test score."""
        return cls.PATTERNS["test_score"].format(
            namespace=CacheNamespace.TEST,
            test_id=test_id
        )
    
    @classmethod
    def test_lock(cls, test_id: str) -> str:
        """Get cache key for test lock (prevents concurrent modifications)."""
        return cls.PATTERNS["test_lock"].format(
            namespace=CacheNamespace.TEST,
            test_id=test_id
        )
    
    @classmethod
    def question_by_id(cls, question_id: str) -> str:
        """Get cache key for question by ID."""
        return cls.PATTERNS["question_by_id"].format(
            namespace=CacheNamespace.QUESTION,
            question_id=question_id
        )
    
    @classmethod
    def question_by_test(cls, test_id: str, question_number: int) -> str:
        """Get cache key for question by test and number."""
        return cls.PATTERNS["question_by_test"].format(
            namespace=CacheNamespace.QUESTION,
            test_id=test_id,
            question_number=question_number
        )
    
    @classmethod
    def question_generation(cls, age_group: str, question_type: str) -> str:
        """Get cache key for question generation context."""
        return cls.PATTERNS["question_generation"].format(
            namespace=CacheNamespace.QUESTION,
            age_group=age_group,
            question_type=question_type
        )
    
    @classmethod
    def answer_by_id(cls, answer_id: str) -> str:
        """Get cache key for answer by ID."""
        return cls.PATTERNS["answer_by_id"].format(
            namespace=CacheNamespace.ANSWER,
            answer_id=answer_id
        )
    
    @classmethod
    def answer_by_question(cls, test_id: str, question_id: str) -> str:
        """Get cache key for answer by test and question."""
        return cls.PATTERNS["answer_by_question"].format(
            namespace=CacheNamespace.ANSWER,
            test_id=test_id,
            question_id=question_id
        )
    
    @classmethod
    def report_by_test(cls, test_id: str) -> str:
        """Get cache key for report by test ID."""
        return cls.PATTERNS["report_by_test"].format(
            namespace=CacheNamespace.REPORT,
            test_id=test_id
        )
    
    @classmethod
    def report_generation_lock(cls, test_id: str) -> str:
        """Get cache key for report generation lock."""
        return cls.PATTERNS["report_generation_lock"].format(
            namespace=CacheNamespace.REPORT,
            test_id=test_id
        )
    
    @classmethod
    def career_recommendations(cls, test_id: str) -> str:
        """Get cache key for career recommendations."""
        return cls.PATTERNS["career_recommendations"].format(
            namespace=CacheNamespace.CAREER,
            test_id=test_id
        )
    
    @classmethod
    def career_validation_questions(cls, raisec_code: str) -> str:
        """Get cache key for career validation questions."""
        return cls.PATTERNS["career_validation_questions"].format(
            namespace=CacheNamespace.CAREER,
            raisec_code=raisec_code
        )
    
    @classmethod
    def session_token(cls, token: str) -> str:
        """Get cache key for session token."""
        return cls.PATTERNS["session_token"].format(
            namespace=CacheNamespace.SESSION,
            token=token
        )
    
    @classmethod
    def session_user(cls, user_id: str) -> str:
        """Get cache key for user's sessions."""
        return cls.PATTERNS["session_user"].format(
            namespace=CacheNamespace.SESSION,
            user_id=user_id
        )
    
    @classmethod
    def rate_limit_api(cls, user_id: str, endpoint: str) -> str:
        """Get cache key for API rate limiting."""
        return cls.PATTERNS["rate_limit_api"].format(
            namespace=CacheNamespace.RATE_LIMIT,
            user_id=user_id,
            endpoint=endpoint.replace("/", "_")
        )
    
    @classmethod
    def rate_limit_test(cls, user_id: str) -> str:
        """Get cache key for test creation rate limiting."""
        return cls.PATTERNS["rate_limit_test"].format(
            namespace=CacheNamespace.RATE_LIMIT,
            user_id=user_id
        )
    
    @classmethod
    def rate_limit_llm(cls, user_id: str) -> str:
        """Get cache key for LLM usage rate limiting."""
        return cls.PATTERNS["rate_limit_llm"].format(
            namespace=CacheNamespace.RATE_LIMIT,
            user_id=user_id
        )
    
    @classmethod
    def prompt_version(cls, version: str) -> str:
        """Get cache key for prompt version."""
        return cls.PATTERNS["prompt_version"].format(
            namespace=CacheNamespace.PROMPT,
            version=version
        )
    
    @classmethod
    def prompt_by_type(cls, prompt_type: str, version: str) -> str:
        """Get cache key for prompt by type and version."""
        return cls.PATTERNS["prompt_by_type"].format(
            namespace=CacheNamespace.PROMPT,
            prompt_type=prompt_type,
            version=version
        )
    
    @classmethod
    def llm_response(cls, hash_key: str) -> str:
        """Get cache key for LLM response caching."""
        return cls.PATTERNS["llm_response"].format(
            namespace=CacheNamespace.LLM,
            hash_key=hash_key
        )
    
    @classmethod
    def llm_generation(cls, context_hash: str) -> str:
        """Get cache key for LLM generation context."""
        return cls.PATTERNS["llm_generation"].format(
            namespace=CacheNamespace.LLM,
            context_hash=context_hash
        )
    
    @classmethod
    def pattern_for_namespace(cls, namespace: CacheNamespace) -> str:
        """Get pattern for clearing all keys in a namespace."""
        return f"{namespace}:*"
    
    @classmethod
    def pattern_for_user(cls, user_id: str) -> str:
        """Get pattern for clearing all user-related cache."""
        return f"*:{user_id}:*"
    
    @classmethod
    def pattern_for_test(cls, test_id: str) -> str:
        """Get pattern for clearing all test-related cache."""
        return f"*:{test_id}*"


# Convenience functions
def get_user_cache_key(user_id: str) -> str:
    """Get user cache key."""
    return CacheKeys.user_by_id(user_id)


def get_test_cache_key(test_id: str) -> str:
    """Get test cache key."""
    return CacheKeys.test_by_id(test_id)


def get_question_cache_key(question_id: str) -> str:
    """Get question cache key."""
    return CacheKeys.question_by_id(question_id)


def get_session_cache_key(token: str) -> str:
    """Get session cache key."""
    return CacheKeys.session_token(token)