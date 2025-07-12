"""Test management service for TruScholar application.

This service handles the complete test lifecycle including creation, question generation,
progress tracking, scoring, and completion.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId

from src.core.config import get_settings
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.models.answer import Answer
from src.models.question import Question
from src.models.test import Test, TestProgress, TestScores, TestStatus, CareerInterests
from src.models.user import User
from src.schemas.test_schemas import (
    AnswerSubmissionRequest,
    CareerInterestsRequest,
    QuestionGenerationContext,
    TestStartRequest,
    TestConfiguration,
    ValidationQuestionsRequest,
)
from src.utils.constants import AgeGroup, QuestionType, RaisecDimension
from src.utils.exceptions import (
    BusinessLogicError,
    ResourceNotFoundError,
    TruScholarError,
    ValidationError,
)
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class TestService:
    """Service for managing RAISEC tests."""

    def __init__(self):
        """Initialize test service."""
        self.db = MongoDB
        self.cache = RedisClient
        self.config = TestConfiguration()

    async def create_test(
        self,
        user_id: str,
        request: TestStartRequest
    ) -> Test:
        """Create a new test for a user.

        Args:
            user_id: ID of the user taking the test
            request: Test creation request

        Returns:
            Test: Created test instance

        Raises:
            ValidationError: If request is invalid
            BusinessLogicError: If user cannot take test
        """
        logger.info(f"Creating test for user {user_id}", extra={"user_id": user_id})

        # Validate user exists and can take test
        user = await self._get_user(user_id)
        await self._validate_user_can_take_test(user)

        # Determine age group
        try:
            age_group = self._determine_age_group(request.age)
        except ValueError as e:
            raise ValidationError(str(e))

        # Create test document
        test = Test(
            user_id=ObjectId(user_id),
            age_group=age_group,
            status=TestStatus.CREATED,
            is_practice=request.is_practice,
            expires_at=datetime.utcnow() + timedelta(hours=self.config.time_limit_hours)
        )

        # Save test
        test_dict = test.to_dict(exclude={"id"})
        result = await self.db.insert_one("tests", test_dict)
        test.id = result.inserted_id

        # Add test to user's test list
        await self._add_test_to_user(user_id, test.id)

        # Cache test for quick access
        await self._cache_test(test)

        logger.info(f"Test created: {test.id}", extra={"test_id": str(test.id), "user_id": user_id})
        return test

    async def start_test(self, test_id: str, user_id: str) -> Test:
        """Start a test by generating questions.

        Args:
            test_id: ID of the test to start
            user_id: ID of the user starting the test

        Returns:
            Test: Updated test with questions

        Raises:
            ResourceNotFoundError: If test not found
            BusinessLogicError: If test cannot be started
        """
        logger.info(f"Starting test {test_id}", extra={"test_id": test_id, "user_id": user_id})

        # Get and validate test
        test = await self.get_test(test_id, user_id)
        if test.status != TestStatus.CREATED:
            raise BusinessLogicError(f"Test {test_id} cannot be started. Current status: {test.status}")

        if datetime.utcnow() > test.expires_at:
            raise BusinessLogicError("Test has expired")

        try:
            # Generate questions
            questions = await self._generate_questions(test)

            # Update test with questions
            test.question_ids = [q.id for q in questions]
            test.start_test()
            test.mark_questions_ready()

            # Save updated test
            await self._update_test(test)

            # Cache questions for quick access
            await self._cache_questions(test_id, questions)

            logger.info(f"Test started with {len(questions)} questions",
                       extra={"test_id": test_id, "question_count": len(questions)})

            return test

        except Exception as e:
            logger.error(f"Failed to start test {test_id}: {str(e)}",
                        extra={"test_id": test_id, "error": str(e)})
            # Mark test as failed
            test.status = TestStatus.FAILED
            await self._update_test(test)
            raise TruScholarError(f"Failed to start test: {str(e)}")

    async def get_test(self, test_id: str, user_id: str) -> Test:
        """Get test by ID with user validation.

        Args:
            test_id: ID of the test
            user_id: ID of the user requesting the test

        Returns:
            Test: Test instance

        Raises:
            ResourceNotFoundError: If test not found
            ValidationError: If user doesn't own test
        """
        # Try cache first
        cached_test = await self._get_cached_test(test_id)
        if cached_test:
            if str(cached_test.user_id) != user_id:
                raise ValidationError("User does not have access to this test")
            return cached_test

        # Get from database
        test_data = await self.db.find_one("tests", {"_id": ObjectId(test_id)})
        if not test_data:
            raise ResourceNotFoundError(f"Test {test_id} not found")

        test = Test.from_dict(test_data)

        # Validate user owns test
        if str(test.user_id) != user_id:
            raise ValidationError("User does not have access to this test")

        # Cache for future use
        await self._cache_test(test)

        return test

    async def get_questions(self, test_id: str, user_id: str) -> List[Question]:
        """Get questions for a test.

        Args:
            test_id: ID of the test
            user_id: ID of the user

        Returns:
            List[Question]: List of questions

        Raises:
            ResourceNotFoundError: If test or questions not found
            BusinessLogicError: If questions not ready
        """
        # Validate test access
        test = await self.get_test(test_id, user_id)

        if test.status not in [TestStatus.QUESTIONS_READY, TestStatus.IN_PROGRESS, TestStatus.SCORING, TestStatus.SCORED]:
            raise BusinessLogicError("Questions are not ready yet")

        # Try cache first
        cached_questions = await self._get_cached_questions(test_id)
        if cached_questions:
            return cached_questions

        # Get from database
        question_filter = {"test_id": ObjectId(test_id)}
        question_docs = await self.db.find("questions", question_filter, sort=[("question_number", 1)])

        questions = [Question.from_dict(doc) for doc in question_docs]

        if not questions:
            raise ResourceNotFoundError(f"No questions found for test {test_id}")

        # Cache questions
        await self._cache_questions(test_id, questions)

        return questions

    async def submit_answer(
        self,
        test_id: str,
        user_id: str,
        request: AnswerSubmissionRequest
    ) -> Answer:
        """Submit an answer to a question.

        Args:
            test_id: ID of the test
            user_id: ID of the user
            request: Answer submission request

        Returns:
            Answer: Created answer

        Raises:
            ValidationError: If answer is invalid
            BusinessLogicError: If test state doesn't allow answers
        """
        logger.info(f"Submitting answer for test {test_id}",
                   extra={"test_id": test_id, "question_id": request.question_id})

        # Validate test can accept answers
        test = await self.get_test(test_id, user_id)
        if not test.can_answer_questions():
            raise BusinessLogicError("Test is not accepting answers")

        # Get question
        question = await self._get_question(request.question_id, test_id)

        # Validate answer format
        if not self._validate_answer_format(question.question_type, request.answer_data):
            raise ValidationError(f"Invalid answer format for {question.question_type}")

        # Check if answer already exists (update scenario)
        existing_answer = await self._get_existing_answer(test_id, request.question_id)

        if existing_answer:
            # Update existing answer
            answer = await self._update_answer(existing_answer, request)
        else:
            # Create new answer
            answer = await self._create_answer(test, question, request)

            # Add answer to test
            test.add_answer(answer.id)
            await self._update_test(test)

        # Update test progress
        await self._update_test_progress(test_id)

        logger.info(f"Answer submitted for question {question.question_number}",
                   extra={"test_id": test_id, "answer_id": str(answer.id)})

        return answer

    async def submit_test(self, test_id: str, user_id: str) -> Test:
        """Submit a completed test for scoring.

        Args:
            test_id: ID of the test
            user_id: ID of the user

        Returns:
            Test: Updated test with scores

        Raises:
            BusinessLogicError: If test cannot be submitted
        """
        logger.info(f"Submitting test {test_id} for scoring", extra={"test_id": test_id})

        # Get test and validate
        test = await self.get_test(test_id, user_id)

        if test.progress.questions_answered < 12:
            raise BusinessLogicError("All questions must be answered before submission")

        if test.status not in [TestStatus.IN_PROGRESS, TestStatus.QUESTIONS_READY]:
            raise BusinessLogicError(f"Test {test_id} cannot be submitted. Current status: {test.status}")

        # Start scoring
        test.start_scoring()
        await self._update_test(test)

        try:
            # Calculate scores
            scores = await self._calculate_scores(test)

            # Update test with scores
            test.complete_scoring(scores)
            await self._update_test(test)

            # Update user stats
            await self._update_user_test_completion(user_id, test)

            logger.info(f"Test scored successfully: RAISEC {scores.raisec_code}",
                       extra={"test_id": test_id, "raisec_code": scores.raisec_code})

            return test

        except Exception as e:
            logger.error(f"Failed to score test {test_id}: {str(e)}",
                        extra={"test_id": test_id, "error": str(e)})
            # Reset test status
            test.status = TestStatus.IN_PROGRESS
            await self._update_test(test)
            raise TruScholarError(f"Failed to score test: {str(e)}")

    async def submit_career_interests(
        self,
        test_id: str,
        user_id: str,
        request: CareerInterestsRequest
    ) -> Tuple[Test, List[str]]:
        """Submit career interests and get validation questions.

        Args:
            test_id: ID of the test
            user_id: ID of the user
            request: Career interests request

        Returns:
            Tuple[Test, List[str]]: Updated test and validation questions

        Raises:
            BusinessLogicError: If test is not ready for interests
        """
        logger.info(f"Submitting career interests for test {test_id}", extra={"test_id": test_id})

        # Get test and validate
        test = await self.get_test(test_id, user_id)

        if not test.can_submit_interests():
            raise BusinessLogicError("Test is not ready for career interests")

        # Create career interests
        interests = CareerInterests(
            interests_text=request.interests_text,
            current_status=request.current_status
        )

        # Generate validation questions based on RAISEC code
        validation_questions = await self._generate_validation_questions(test.scores.raisec_code)
        interests.validation_questions = validation_questions

        # Update test
        test.add_career_interests(interests)
        await self._update_test(test)

        logger.info(f"Career interests submitted for test {test_id}", extra={"test_id": test_id})

        return test, validation_questions

    async def submit_validation_responses(
        self,
        test_id: str,
        user_id: str,
        request: ValidationQuestionsRequest
    ) -> Test:
        """Submit validation question responses.

        Args:
            test_id: ID of the test
            user_id: ID of the user
            request: Validation responses

        Returns:
            Test: Updated test

        Raises:
            BusinessLogicError: If validation not ready
        """
        logger.info(f"Submitting validation responses for test {test_id}", extra={"test_id": test_id})

        # Get test and validate
        test = await self.get_test(test_id, user_id)

        if not test.career_interests or not test.career_interests.validation_questions:
            raise BusinessLogicError("Test is not ready for validation responses")

        if len(request.responses) != len(test.career_interests.validation_questions):
            raise ValidationError("Number of responses must match number of questions")

        # Update career interests with responses
        test.career_interests.validation_responses = request.responses
        test.status = TestStatus.VALIDATION_COMPLETED
        await self._update_test(test)

        logger.info(f"Validation responses submitted for test {test_id}", extra={"test_id": test_id})

        return test

    async def extend_test(self, test_id: str, user_id: str, hours: int = 1) -> Test:
        """Extend test expiry time.

        Args:
            test_id: ID of the test
            user_id: ID of the user
            hours: Hours to extend

        Returns:
            Test: Updated test

        Raises:
            BusinessLogicError: If test cannot be extended
        """
        test = await self.get_test(test_id, user_id)

        if test.status not in [TestStatus.IN_PROGRESS, TestStatus.QUESTIONS_READY]:
            raise BusinessLogicError("Only active tests can be extended")

        # Extend test
        test.extend_expiry(hours)
        await self._update_test(test)

        logger.info(f"Test {test_id} extended by {hours} hours",
                   extra={"test_id": test_id, "hours": hours})

        return test

    async def get_user_tests(self, user_id: str, limit: int = 10) -> List[Test]:
        """Get user's test history.

        Args:
            user_id: ID of the user
            limit: Maximum number of tests to return

        Returns:
            List[Test]: List of user's tests
        """
        test_filter = {"user_id": ObjectId(user_id)}
        test_docs = await self.db.find(
            "tests",
            test_filter,
            sort=[("created_at", -1)],
            limit=limit
        )

        return [Test.from_dict(doc) for doc in test_docs]

    # Private helper methods

    async def _get_user(self, user_id: str) -> User:
        """Get user by ID."""
        user_data = await self.db.find_one("users", {"_id": ObjectId(user_id)})
        if not user_data:
            raise ResourceNotFoundError(f"User {user_id} not found")
        return User.from_dict(user_data)

    async def _validate_user_can_take_test(self, user: User) -> None:
        """Validate user can take a new test."""
        if not user.can_take_test():
            raise BusinessLogicError("User cannot take tests")

        # Check daily limit if enabled
        today = datetime.utcnow().date()
        daily_tests = await self.db.count_documents(
            "tests",
            {
                "user_id": user.id,
                "created_at": {
                    "$gte": datetime.combine(today, datetime.min.time()),
                    "$lt": datetime.combine(today + timedelta(days=1), datetime.min.time())
                }
            }
        )

        if daily_tests >= 3:  # Max test attempts per day
            raise BusinessLogicError("Daily test limit reached")

    def _determine_age_group(self, age: int) -> AgeGroup:
        """Determine age group from age."""
        age_group_ranges = {
            "13-17": (13, 17),
            "18-25": (18, 25),
            "26-35": (26, 35),
            "36+": (36, 99)
        }
        for age_group, (min_age, max_age) in age_group_ranges.items():
            if min_age <= age <= max_age:
                return age_group
        raise ValueError(f"Age {age} is not supported")

    async def _add_test_to_user(self, user_id: str, test_id: ObjectId) -> None:
        """Add test ID to user's test list."""
        await self.db.update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$push": {"test_ids": test_id}}
        )

    async def _generate_questions(self, test: Test) -> List[Question]:
        """Generate questions for a test."""
        from src.services.question_service import QuestionService

        question_service = QuestionService()
        return await question_service.generate_test_questions(
            test_id=test.id,
            age_group=test.age_group,
            distribution={"standard": 70, "scenario": 20, "interactive": 10}  # Question distribution
        )

    async def _get_question(self, question_id: str, test_id: str) -> Question:
        """Get question by ID and validate it belongs to test."""
        question_data = await self.db.find_one(
            "questions",
            {"_id": ObjectId(question_id), "test_id": ObjectId(test_id)}
        )
        if not question_data:
            raise ResourceNotFoundError(f"Question {question_id} not found in test {test_id}")
        return Question.from_dict(question_data)

    def _validate_answer_format(self, question_type: QuestionType, answer_data: Dict[str, Any]) -> bool:
        """Validate answer format for question type."""
        from src.schemas.test_schemas import TestRequestValidator
        return TestRequestValidator.validate_answer_format(question_type, answer_data)

    async def _get_existing_answer(self, test_id: str, question_id: str) -> Optional[Answer]:
        """Check if answer already exists."""
        answer_data = await self.db.find_one(
            "answers",
            {"test_id": ObjectId(test_id), "question_id": ObjectId(question_id)}
        )
        return Answer.from_dict(answer_data) if answer_data else None

    async def _create_answer(
        self,
        test: Test,
        question: Question,
        request: AnswerSubmissionRequest
    ) -> Answer:
        """Create new answer."""
        answer = Answer(
            test_id=test.id,
            question_id=ObjectId(request.question_id),
            user_id=test.user_id,
            question_number=question.question_number,
            question_type=question.question_type,
            answer_data=request.answer_data,
            time_spent_seconds=request.time_spent_seconds
        )

        # Validate answer
        answer.validate_answer(question)

        # Calculate scores
        answer.calculate_scores(question)

        # Save answer
        answer_dict = answer.to_dict(exclude={"id"})
        result = await self.db.insert_one("answers", answer_dict)
        answer.id = result.inserted_id

        return answer

    async def _update_answer(self, answer: Answer, request: AnswerSubmissionRequest) -> Answer:
        """Update existing answer."""
        answer.answer_data = request.answer_data
        answer.time_spent_seconds = request.time_spent_seconds
        answer.changed_count += 1
        answer.update_timestamps()

        # Save updated answer
        await self.db.update_one(
            "answers",
            {"_id": answer.id},
            {"$set": answer.to_dict(exclude={"id", "_id"})}
        )

        return answer

    async def _calculate_scores(self, test: Test) -> TestScores:
        """Calculate test scores."""
        from src.services.scoring_service import ScoringService

        scoring_service = ScoringService()
        return await scoring_service.calculate_test_scores(test.id)

    async def _generate_validation_questions(self, raisec_code: str) -> List[str]:
        """Generate validation questions based on RAISEC code."""
        # This would use LLM to generate personalized validation questions
        # For now, return static questions based on primary dimension
        primary_dim = raisec_code[0]

        base_questions = {
            "R": "Do you enjoy working with your hands and building things?",
            "A": "Do you find yourself drawn to creative and artistic activities?",
            "I": "Do you prefer analyzing problems and finding logical solutions?",
            "S": "Do you feel energized when helping others?",
            "E": "Do you enjoy taking leadership roles in group situations?",
            "C": "Do you prefer organized, structured work environments?"
        }

        # Generate 3 questions - could be enhanced with LLM
        questions = [
            base_questions.get(primary_dim, "Do you feel this assessment reflects your interests?"),
            "Would you be comfortable working in the career areas suggested by your results?",
            "Do you think your personality type matches the description provided?"
        ]

        return questions

    async def _update_test(self, test: Test) -> None:
        """Update test in database."""
        test.update_timestamps()
        await self.db.update_one(
            "tests",
            {"_id": test.id},
            {"$set": test.to_dict(exclude={"id", "_id"})}
        )

        # Update cache
        await self._cache_test(test)

    async def _update_test_progress(self, test_id: str) -> None:
        """Update test progress tracking."""
        # This could include more sophisticated progress tracking
        pass

    async def _update_user_test_completion(self, user_id: str, test: Test) -> None:
        """Update user statistics after test completion."""
        duration = test.completion_time_minutes or 0

        await self.db.update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {
                "$inc": {"stats.tests_completed": 1},
                "$set": {"stats.last_test_date": datetime.utcnow()},
                "$push": {"stats.average_test_duration_minutes": duration}
            }
        )

    # Caching methods

    async def _cache_test(self, test: Test) -> None:
        """Cache test data."""
        if not settings.ENABLE_CACHE:
            return

        cache_key = f"test:{test.id}"
        await self.cache.set_json(cache_key, test.to_dict(), ttl=7200)  # 2 hour TTL

    async def _get_cached_test(self, test_id: str) -> Optional[Test]:
        """Get cached test data."""
        if not settings.ENABLE_CACHE:
            return None

        cache_key = f"test:{test_id}"
        cached_data = await self.cache.get_json(cache_key)
        return Test.from_dict(cached_data) if cached_data else None

    async def _cache_questions(self, test_id: str, questions: List[Question]) -> None:
        """Cache questions for a test."""
        if not settings.ENABLE_CACHE:
            return

        cache_key = f"test:{test_id}:questions"
        questions_data = [q.to_dict() for q in questions]
        await self.cache.set_json(cache_key, questions_data, ttl=3600)  # 1 hour TTL

    async def _get_cached_questions(self, test_id: str) -> Optional[List[Question]]:
        """Get cached questions for a test."""
        if not settings.ENABLE_CACHE:
            return None

        cache_key = f"test:{test_id}:questions"
        cached_data = await self.cache.get_json(cache_key)
        return [Question.from_dict(q) for q in cached_data] if cached_data else None
