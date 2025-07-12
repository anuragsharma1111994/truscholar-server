"""Unit tests for TestService.

This module contains unit tests for the test management service,
covering test creation, question generation, answer submission, and scoring.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from src.services.test_service import TestService
from src.models.test import Test, TestStatus, TestProgress, TestScores
from src.models.user import User
from src.models.question import Question
from src.models.answer import Answer
from src.schemas.test_schemas import (
    TestStartRequest,
    AnswerSubmissionRequest,
    CareerInterestsRequest,
    ValidationQuestionsRequest
)
from src.utils.constants import AgeGroup, QuestionType, RaisecDimension
from src.utils.exceptions import (
    ValidationError,
    BusinessLogicError,
    ResourceNotFoundError,
    TruScholarError
)


@pytest.fixture
def test_service():
    """Create TestService instance with mocked dependencies."""
    service = TestService()
    service.db = AsyncMock()
    service.cache = AsyncMock()
    return service


@pytest.fixture
def mock_user():
    """Create mock user."""
    user_data = {
        "_id": ObjectId(),
        "name": "Test User",
        "phone": "9876543210",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    return User.from_dict(user_data)


@pytest.fixture
def mock_test():
    """Create mock test."""
    test = Test(
        user_id=ObjectId(),
        age_group=AgeGroup.AGE_18_25,
        status=TestStatus.CREATED,
        is_practice=False
    )
    test.id = ObjectId()
    return test


@pytest.fixture
def mock_questions():
    """Create mock questions."""
    questions = []
    for i in range(12):
        question = Question(
            test_id=ObjectId(),
            question_number=i + 1,
            question_type=QuestionType.MCQ,
            question_text=f"Test question {i + 1}",
            raisec_dimensions=[RaisecDimension.REALISTIC]
        )
        question.id = ObjectId()
        questions.append(question)
    return questions


class TestTestService:
    """Test cases for TestService."""
    
    @pytest.mark.asyncio
    async def test_create_test_success(self, test_service, mock_user):
        """Test successful test creation."""
        # Arrange
        user_id = str(mock_user.id)
        request = TestStartRequest(age=20, test_type="raisec", is_practice=False)
        
        test_service._get_user = AsyncMock(return_value=mock_user)
        test_service._validate_user_can_take_test = AsyncMock()
        test_service.db.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id=ObjectId())
        )
        test_service._add_test_to_user = AsyncMock()
        test_service._cache_test = AsyncMock()
        
        # Act
        result = await test_service.create_test(user_id, request)
        
        # Assert
        assert isinstance(result, Test)
        assert result.age_group == AgeGroup.AGE_18_25
        assert result.status == TestStatus.CREATED
        assert result.is_practice == request.is_practice
        test_service.db.insert_one.assert_called_once()
        test_service._add_test_to_user.assert_called_once()
        test_service._cache_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_test_invalid_age(self, test_service, mock_user):
        """Test test creation with invalid age."""
        # Arrange
        user_id = str(mock_user.id)
        request = TestStartRequest(age=10, test_type="raisec", is_practice=False)
        
        test_service._get_user = AsyncMock(return_value=mock_user)
        test_service._validate_user_can_take_test = AsyncMock()
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await test_service.create_test(user_id, request)
    
    @pytest.mark.asyncio
    async def test_create_test_user_cannot_take_test(self, test_service, mock_user):
        """Test test creation when user cannot take test."""
        # Arrange
        user_id = str(mock_user.id)
        request = TestStartRequest(age=20, test_type="raisec", is_practice=False)
        
        test_service._get_user = AsyncMock(return_value=mock_user)
        test_service._validate_user_can_take_test = AsyncMock(
            side_effect=BusinessLogicError("User cannot take tests")
        )
        
        # Act & Assert
        with pytest.raises(BusinessLogicError):
            await test_service.create_test(user_id, request)
    
    @pytest.mark.asyncio
    async def test_start_test_success(self, test_service, mock_test, mock_questions):
        """Test successful test start."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        test_service._generate_questions = AsyncMock(return_value=mock_questions)
        test_service._update_test = AsyncMock()
        test_service._cache_questions = AsyncMock()
        
        # Act
        result = await test_service.start_test(test_id, user_id)
        
        # Assert
        assert result.status == TestStatus.QUESTIONS_READY
        assert len(result.question_ids) == 12
        test_service._generate_questions.assert_called_once()
        test_service._update_test.assert_called_once()
        test_service._cache_questions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_test_already_started(self, test_service, mock_test):
        """Test starting an already started test."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        mock_test.status = TestStatus.IN_PROGRESS
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        
        # Act & Assert
        with pytest.raises(BusinessLogicError):
            await test_service.start_test(test_id, user_id)
    
    @pytest.mark.asyncio
    async def test_start_test_expired(self, test_service, mock_test):
        """Test starting an expired test."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        mock_test.expires_at = datetime.utcnow() - timedelta(hours=1)
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        
        # Act & Assert
        with pytest.raises(BusinessLogicError, match="Test has expired"):
            await test_service.start_test(test_id, user_id)
    
    @pytest.mark.asyncio
    async def test_get_test_success(self, test_service, mock_test):
        """Test successful test retrieval."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        test_service._get_cached_test = AsyncMock(return_value=None)
        test_service.db.find_one = AsyncMock(return_value=mock_test.to_dict())
        test_service._cache_test = AsyncMock()
        
        # Act
        result = await test_service.get_test(test_id, user_id)
        
        # Assert
        assert isinstance(result, Test)
        assert str(result.id) == test_id
        test_service.db.find_one.assert_called_once()
        test_service._cache_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_test_from_cache(self, test_service, mock_test):
        """Test test retrieval from cache."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        test_service._get_cached_test = AsyncMock(return_value=mock_test)
        
        # Act
        result = await test_service.get_test(test_id, user_id)
        
        # Assert
        assert result == mock_test
        test_service.db.find_one.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_test_not_found(self, test_service):
        """Test test retrieval when not found."""
        # Arrange
        test_id = str(ObjectId())
        user_id = str(ObjectId())
        
        test_service._get_cached_test = AsyncMock(return_value=None)
        test_service.db.find_one = AsyncMock(return_value=None)
        
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await test_service.get_test(test_id, user_id)
    
    @pytest.mark.asyncio
    async def test_get_test_wrong_user(self, test_service, mock_test):
        """Test test retrieval by wrong user."""
        # Arrange
        test_id = str(mock_test.id)
        wrong_user_id = str(ObjectId())
        
        test_service._get_cached_test = AsyncMock(return_value=None)
        test_service.db.find_one = AsyncMock(return_value=mock_test.to_dict())
        
        # Act & Assert
        with pytest.raises(ValidationError, match="User does not have access"):
            await test_service.get_test(test_id, wrong_user_id)
    
    @pytest.mark.asyncio
    async def test_submit_answer_success(self, test_service, mock_test, mock_questions):
        """Test successful answer submission."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        question = mock_questions[0]
        
        request = AnswerSubmissionRequest(
            question_id=str(question.id),
            answer_data={"selected_option": "A"},
            time_spent_seconds=30
        )
        
        mock_test.status = TestStatus.IN_PROGRESS
        test_service.get_test = AsyncMock(return_value=mock_test)
        test_service._get_question = AsyncMock(return_value=question)
        test_service._validate_answer_format = MagicMock(return_value=True)
        test_service._get_existing_answer = AsyncMock(return_value=None)
        test_service._create_answer = AsyncMock(return_value=Answer(
            test_id=mock_test.id,
            question_id=question.id,
            user_id=mock_test.user_id,
            question_number=1,
            question_type=QuestionType.MCQ,
            answer_data=request.answer_data
        ))
        test_service._update_test = AsyncMock()
        test_service._update_test_progress = AsyncMock()
        
        # Act
        result = await test_service.submit_answer(test_id, user_id, request)
        
        # Assert
        assert isinstance(result, Answer)
        test_service._create_answer.assert_called_once()
        test_service._update_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_answer_test_not_accepting(self, test_service, mock_test):
        """Test answer submission when test not accepting answers."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        mock_test.status = TestStatus.SCORED
        
        request = AnswerSubmissionRequest(
            question_id=str(ObjectId()),
            answer_data={"selected_option": "A"},
            time_spent_seconds=30
        )
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        
        # Act & Assert
        with pytest.raises(BusinessLogicError, match="not accepting answers"):
            await test_service.submit_answer(test_id, user_id, request)
    
    @pytest.mark.asyncio
    async def test_submit_test_success(self, test_service, mock_test):
        """Test successful test submission."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        mock_test.status = TestStatus.IN_PROGRESS
        mock_test.progress.questions_answered = 12
        
        mock_scores = TestScores()
        mock_scores.raisec_code = "RIA"
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        test_service._calculate_scores = AsyncMock(return_value=mock_scores)
        test_service._update_test = AsyncMock()
        test_service._update_user_test_completion = AsyncMock()
        
        # Act
        result = await test_service.submit_test(test_id, user_id)
        
        # Assert
        assert result.status == TestStatus.SCORED
        assert result.scores.raisec_code == "RIA"
        test_service._calculate_scores.assert_called_once()
        test_service._update_user_test_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_test_incomplete(self, test_service, mock_test):
        """Test test submission with incomplete answers."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        mock_test.status = TestStatus.IN_PROGRESS
        mock_test.progress.questions_answered = 8
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        
        # Act & Assert
        with pytest.raises(BusinessLogicError, match="All questions must be answered"):
            await test_service.submit_test(test_id, user_id)
    
    @pytest.mark.asyncio
    async def test_submit_career_interests_success(self, test_service, mock_test):
        """Test successful career interests submission."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        request = CareerInterestsRequest(
            interests_text="I want to work in technology",
            current_status="Student"
        )
        
        mock_test.status = TestStatus.SCORED
        mock_test.scores = TestScores()
        mock_test.scores.raisec_code = "IRA"
        
        validation_questions = [
            "Do you prefer analyzing problems?",
            "Would you enjoy research work?",
            "Do you like creative problem solving?"
        ]
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        test_service._generate_validation_questions = AsyncMock(
            return_value=validation_questions
        )
        test_service._update_test = AsyncMock()
        
        # Act
        result_test, result_questions = await test_service.submit_career_interests(
            test_id, user_id, request
        )
        
        # Assert
        assert result_test.career_interests is not None
        assert result_test.career_interests.interests_text == request.interests_text
        assert result_questions == validation_questions
        test_service._update_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_validation_responses_success(self, test_service, mock_test):
        """Test successful validation responses submission."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        
        request = ValidationQuestionsRequest(responses=[True, False, True])
        
        mock_test.career_interests = MagicMock()
        mock_test.career_interests.validation_questions = [
            "Question 1", "Question 2", "Question 3"
        ]
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        test_service._update_test = AsyncMock()
        
        # Act
        result = await test_service.submit_validation_responses(
            test_id, user_id, request
        )
        
        # Assert
        assert result.status == TestStatus.VALIDATION_COMPLETED
        assert result.career_interests.validation_responses == request.responses
        test_service._update_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extend_test_success(self, test_service, mock_test):
        """Test successful test extension."""
        # Arrange
        test_id = str(mock_test.id)
        user_id = str(mock_test.user_id)
        hours = 2
        
        mock_test.status = TestStatus.IN_PROGRESS
        original_expiry = mock_test.expires_at
        
        test_service.get_test = AsyncMock(return_value=mock_test)
        test_service._update_test = AsyncMock()
        
        # Act
        result = await test_service.extend_test(test_id, user_id, hours)
        
        # Assert
        assert result.expires_at > original_expiry
        test_service._update_test.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_tests_success(self, test_service):
        """Test successful user tests retrieval."""
        # Arrange
        user_id = str(ObjectId())
        limit = 5
        
        mock_test_docs = [
            {"_id": ObjectId(), "user_id": ObjectId(user_id), "status": "completed"},
            {"_id": ObjectId(), "user_id": ObjectId(user_id), "status": "in_progress"}
        ]
        
        test_service.db.find = AsyncMock(return_value=mock_test_docs)
        
        # Act
        result = await test_service.get_user_tests(user_id, limit)
        
        # Assert
        assert len(result) == 2
        assert all(isinstance(test, Test) for test in result)
        test_service.db.find.assert_called_once_with(
            "tests",
            {"user_id": ObjectId(user_id)},
            sort=[("created_at", -1)],
            limit=limit
        )