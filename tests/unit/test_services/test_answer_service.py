"""Unit tests for answer service."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from src.services.answer_service import AnswerService
from src.schemas.answer_schemas import (
    AnswerSubmitRequest,
    BulkAnswerSubmitRequest,
    MCQAnswerData,
    StatementSetAnswerData,
    ScenarioMCQAnswerData,
    ScenarioMultiSelectAnswerData,
    ThisOrThatAnswerData,
    ScaleRatingAnswerData,
    PlotDayAnswerData
)
from src.utils.enums import QuestionType, RaisecDimension, TestStatus
from src.models.question import Question
from src.models.test import Test


@pytest.fixture
def mock_db():
    """Create mock database."""
    db = MagicMock()
    db.get_collection = MagicMock()
    return db


@pytest.fixture
def mock_cache():
    """Create mock cache."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    return cache


@pytest.fixture
def answer_service(mock_db, mock_cache):
    """Create answer service instance."""
    return AnswerService(mock_db, mock_cache)


@pytest.fixture
def sample_user_id():
    """Sample user ID."""
    return str(ObjectId())


@pytest.fixture
def sample_test_id():
    """Sample test ID."""
    return str(ObjectId())


@pytest.fixture
def sample_question_id():
    """Sample question ID."""
    return str(ObjectId())


@pytest.fixture
def sample_mcq_question(sample_test_id):
    """Sample MCQ question."""
    return Question(
        id=str(ObjectId()),
        test_id=sample_test_id,
        question_number=1,
        question_type=QuestionType.MCQ,
        content={
            "question_text": "What is your preferred work style?",
            "options": [
                {"id": "a", "text": "Independent", "dimensions": {"I": 1.0}},
                {"id": "b", "text": "Collaborative", "dimensions": {"S": 1.0}},
                {"id": "c", "text": "Creative", "dimensions": {"A": 1.0}},
                {"id": "d", "text": "Structured", "dimensions": {"C": 1.0}}
            ]
        },
        dimensions=[RaisecDimension.I, RaisecDimension.S, RaisecDimension.A, RaisecDimension.C],
        is_ai_generated=True
    )


@pytest.fixture
def sample_test(sample_test_id, sample_user_id):
    """Sample test."""
    return Test(
        id=sample_test_id,
        user_id=sample_user_id,
        test_type="career_assessment",
        status=TestStatus.IN_PROGRESS,
        age_group="18-25",
        question_count=12,
        questions_answered=0,
        started_at=datetime.utcnow()
    )


class TestAnswerSubmission:
    """Test answer submission functionality."""
    
    @pytest.mark.asyncio
    async def test_submit_mcq_answer_success(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_question_id,
        sample_mcq_question,
        sample_test
    ):
        """Test successful MCQ answer submission."""
        # Setup mocks
        answer_service.question_collection.find_one = AsyncMock(
            return_value=sample_mcq_question.to_db()
        )
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        answer_service.answer_collection.find_one = AsyncMock(return_value=None)
        answer_service.answer_collection.insert_one = AsyncMock(
            return_value=MagicMock(inserted_id=ObjectId())
        )
        answer_service.question_collection.find = MagicMock(
            return_value=AsyncMock(__aiter__=AsyncMock(return_value=iter([
                {"_id": ObjectId(sample_question_id), "created_at": datetime.utcnow()}
            ])))
        )
        answer_service.answer_collection.count_documents = AsyncMock(return_value=1)
        answer_service.question_collection.count_documents = AsyncMock(return_value=12)
        answer_service.test_collection.update_one = AsyncMock()
        
        # Create request
        request = AnswerSubmitRequest(
            question_id=sample_question_id,
            question_type=QuestionType.MCQ,
            answer_data=MCQAnswerData(selected_option="a"),
            time_spent_seconds=45
        )
        
        # Submit answer
        response = await answer_service.submit_answer(sample_user_id, request)
        
        # Assertions
        assert response.question_id == sample_question_id
        assert response.question_type == QuestionType.MCQ
        assert response.answer_data["selected_option"] == "a"
        assert response.time_spent_seconds == 45
        assert response.validation.is_valid is True
        assert response.score is not None
        assert response.score.dimension_scores[RaisecDimension.I] == 1.0
        
    @pytest.mark.asyncio
    async def test_submit_answer_invalid_question(
        self,
        answer_service,
        sample_user_id,
        sample_question_id
    ):
        """Test answer submission with invalid question."""
        # Setup mocks
        answer_service.question_collection.find_one = AsyncMock(return_value=None)
        
        # Create request
        request = AnswerSubmitRequest(
            question_id=sample_question_id,
            question_type=QuestionType.MCQ,
            answer_data=MCQAnswerData(selected_option="a"),
            time_spent_seconds=45
        )
        
        # Submit answer should raise error
        with pytest.raises(ValueError, match="not found"):
            await answer_service.submit_answer(sample_user_id, request)
            
    @pytest.mark.asyncio
    async def test_submit_answer_unauthorized(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_question_id,
        sample_mcq_question,
        sample_test
    ):
        """Test answer submission by unauthorized user."""
        # Setup mocks with different user ID
        sample_test.user_id = str(ObjectId())  # Different user
        
        answer_service.question_collection.find_one = AsyncMock(
            return_value=sample_mcq_question.to_db()
        )
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        
        # Create request
        request = AnswerSubmitRequest(
            question_id=sample_question_id,
            question_type=QuestionType.MCQ,
            answer_data=MCQAnswerData(selected_option="a"),
            time_spent_seconds=45
        )
        
        # Submit answer should raise error
        with pytest.raises(ValueError, match="unauthorized"):
            await answer_service.submit_answer(sample_user_id, request)
            
    @pytest.mark.asyncio
    async def test_submit_answer_completed_test(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_question_id,
        sample_mcq_question,
        sample_test
    ):
        """Test answer submission to completed test."""
        # Setup mocks with completed test
        sample_test.status = TestStatus.COMPLETED
        
        answer_service.question_collection.find_one = AsyncMock(
            return_value=sample_mcq_question.to_db()
        )
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        
        # Create request
        request = AnswerSubmitRequest(
            question_id=sample_question_id,
            question_type=QuestionType.MCQ,
            answer_data=MCQAnswerData(selected_option="a"),
            time_spent_seconds=45
        )
        
        # Submit answer should raise error
        with pytest.raises(ValueError, match="completed test"):
            await answer_service.submit_answer(sample_user_id, request)
            
    @pytest.mark.asyncio
    async def test_update_existing_answer(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_question_id,
        sample_mcq_question,
        sample_test
    ):
        """Test updating existing answer."""
        # Setup mocks
        existing_answer = {
            "_id": ObjectId(),
            "user_id": ObjectId(sample_user_id),
            "test_id": ObjectId(sample_test_id),
            "question_id": ObjectId(sample_question_id),
            "question_number": 1,
            "answer_data": {"selected_option": "b"},
            "changed_count": 1,
            "submitted_at": datetime.utcnow()
        }
        
        answer_service.question_collection.find_one = AsyncMock(
            return_value=sample_mcq_question.to_db()
        )
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        answer_service.answer_collection.find_one = AsyncMock(
            return_value=existing_answer
        )
        answer_service.answer_collection.update_one = AsyncMock()
        answer_service.answer_collection.count_documents = AsyncMock(return_value=1)
        answer_service.question_collection.count_documents = AsyncMock(return_value=12)
        answer_service.test_collection.update_one = AsyncMock()
        
        # Create request with different answer
        request = AnswerSubmitRequest(
            question_id=sample_question_id,
            question_type=QuestionType.MCQ,
            answer_data=MCQAnswerData(selected_option="c"),
            time_spent_seconds=60
        )
        
        # Submit answer
        response = await answer_service.submit_answer(sample_user_id, request)
        
        # Assertions
        assert response.answer_data["selected_option"] == "c"
        assert response.changed_count == 2  # Incremented
        assert answer_service.answer_collection.update_one.called
        

class TestAnswerValidation:
    """Test answer validation functionality."""
    
    @pytest.mark.asyncio
    async def test_validate_statement_set_answer(self, answer_service):
        """Test statement set answer validation."""
        # Create question with 5 statements
        question = Question(
            id=str(ObjectId()),
            test_id=str(ObjectId()),
            question_number=1,
            question_type=QuestionType.STATEMENT_SET,
            content={
                "question_text": "Rate these statements",
                "statements": [
                    {"id": "1", "text": "Statement 1", "dimensions": {"R": 1.0}},
                    {"id": "2", "text": "Statement 2", "dimensions": {"A": 1.0}},
                    {"id": "3", "text": "Statement 3", "dimensions": {"I": 1.0}},
                    {"id": "4", "text": "Statement 4", "dimensions": {"S": 1.0}},
                    {"id": "5", "text": "Statement 5", "dimensions": {"E": 1.0}}
                ]
            }
        )
        
        # Valid answer
        valid_request = AnswerSubmitRequest(
            question_id=str(question.id),
            question_type=QuestionType.STATEMENT_SET,
            answer_data=StatementSetAnswerData(
                ratings={"1": 5, "2": 4, "3": 3, "4": 2, "5": 1}
            ),
            time_spent_seconds=120
        )
        
        result = await answer_service._validate_answer(valid_request, question)
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Invalid answer - missing ratings
        invalid_request = AnswerSubmitRequest(
            question_id=str(question.id),
            question_type=QuestionType.STATEMENT_SET,
            answer_data=StatementSetAnswerData(
                ratings={"1": 5, "2": 4}  # Missing 3, 4, 5
            ),
            time_spent_seconds=120
        )
        
        result = await answer_service._validate_answer(invalid_request, question)
        assert result.is_valid is False
        assert any("Missing ratings" in error for error in result.errors)
        
    @pytest.mark.asyncio
    async def test_validate_plot_day_answer(self, answer_service):
        """Test plot day answer validation."""
        # Create question with tasks
        question = Question(
            id=str(ObjectId()),
            test_id=str(ObjectId()),
            question_number=1,
            question_type=QuestionType.PLOT_DAY,
            content={
                "question_text": "Plan your day",
                "tasks": [
                    {"id": "task1", "title": "Task 1", "primary_dimension": "R"},
                    {"id": "task2", "title": "Task 2", "primary_dimension": "A"},
                    {"id": "task3", "title": "Task 3", "primary_dimension": "I"},
                    {"id": "task4", "title": "Task 4", "primary_dimension": "S"}
                ]
            }
        )
        
        # Valid answer - all tasks placed
        valid_request = AnswerSubmitRequest(
            question_id=str(question.id),
            question_type=QuestionType.PLOT_DAY,
            answer_data=PlotDayAnswerData(
                placements={
                    "morning": ["task1", "task2"],
                    "afternoon": ["task3"]
                },
                not_interested=["task4"]
            ),
            time_spent_seconds=180
        )
        
        result = await answer_service._validate_answer(valid_request, question)
        assert result.is_valid is True
        
        # Invalid answer - duplicate task
        invalid_request = AnswerSubmitRequest(
            question_id=str(question.id),
            question_type=QuestionType.PLOT_DAY,
            answer_data=PlotDayAnswerData(
                placements={
                    "morning": ["task1", "task2"],
                    "afternoon": ["task1"]  # Duplicate
                },
                not_interested=["task3", "task4"]
            ),
            time_spent_seconds=180
        )
        
        with pytest.raises(ValueError, match="Duplicate"):
            await answer_service._validate_answer(invalid_request, question)
            

class TestAnswerScoring:
    """Test answer scoring functionality."""
    
    @pytest.mark.asyncio
    async def test_score_mcq_answer(self, answer_service, sample_mcq_question):
        """Test MCQ answer scoring."""
        request = AnswerSubmitRequest(
            question_id=str(sample_mcq_question.id),
            question_type=QuestionType.MCQ,
            answer_data=MCQAnswerData(selected_option="a"),
            time_spent_seconds=45
        )
        
        score = await answer_service._calculate_score(request, sample_mcq_question)
        
        assert score is not None
        assert score.dimension_scores[RaisecDimension.I] == 1.0
        assert score.total_score == 1.0
        assert score.weighted_score == 1.0  # MCQ weight is 1.0
        assert 0 <= score.confidence <= 1.0
        
    @pytest.mark.asyncio
    async def test_score_scale_rating_answer(self, answer_service):
        """Test scale rating answer scoring."""
        question = Question(
            id=str(ObjectId()),
            test_id=str(ObjectId()),
            question_number=1,
            question_type=QuestionType.SCALE_RATING,
            content={
                "question_text": "Rate your interest",
                "scale_min": 1,
                "scale_max": 10,
                "scale_labels": {"min": "Low", "max": "High"}
            },
            dimensions=[RaisecDimension.R, RaisecDimension.I]
        )
        
        request = AnswerSubmitRequest(
            question_id=str(question.id),
            question_type=QuestionType.SCALE_RATING,
            answer_data=ScaleRatingAnswerData(rating=8),
            time_spent_seconds=30
        )
        
        score = await answer_service._calculate_score(request, question)
        
        assert score is not None
        # Rating 8 out of 1-10 = 0.777...
        expected_normalized = (8 - 1) / (10 - 1)
        assert score.dimension_scores[RaisecDimension.R] == pytest.approx(expected_normalized, 0.01)
        assert score.dimension_scores[RaisecDimension.I] == pytest.approx(expected_normalized, 0.01)
        

class TestBulkOperations:
    """Test bulk answer operations."""
    
    @pytest.mark.asyncio
    async def test_submit_bulk_answers(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_test
    ):
        """Test bulk answer submission."""
        # Setup mocks
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        
        # Mock submit_answer to return successful responses
        async def mock_submit(user_id, request):
            return MagicMock(
                id=str(ObjectId()),
                question_id=request.question_id,
                question_type=request.question_type
            )
        
        answer_service.submit_answer = mock_submit
        
        # Create bulk request
        request = BulkAnswerSubmitRequest(
            test_id=sample_test_id,
            answers=[
                AnswerSubmitRequest(
                    question_id=str(ObjectId()),
                    question_type=QuestionType.MCQ,
                    answer_data=MCQAnswerData(selected_option="a"),
                    time_spent_seconds=45
                ),
                AnswerSubmitRequest(
                    question_id=str(ObjectId()),
                    question_type=QuestionType.SCALE_RATING,
                    answer_data=ScaleRatingAnswerData(rating=7),
                    time_spent_seconds=30
                )
            ]
        )
        
        # Submit bulk answers
        responses = await answer_service.submit_bulk_answers(sample_user_id, request)
        
        # Assertions
        assert len(responses) == 2
        assert responses[0].question_type == QuestionType.MCQ
        assert responses[1].question_type == QuestionType.SCALE_RATING
        

class TestAnswerRetrieval:
    """Test answer retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_test_answers(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_test
    ):
        """Test retrieving all answers for a test."""
        # Setup mocks
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        
        # Mock answers
        mock_answers = [
            {
                "_id": ObjectId(),
                "test_id": ObjectId(sample_test_id),
                "question_id": ObjectId(),
                "question_number": i,
                "question_type": "mcq",
                "answer_data": {"selected_option": "a"},
                "validation": {"is_valid": True, "errors": [], "warnings": []},
                "score": {
                    "dimension_scores": {"R": 1.0},
                    "total_score": 1.0,
                    "weighted_score": 1.0,
                    "confidence": 0.9
                },
                "time_spent_seconds": 45,
                "changed_count": 0,
                "submitted_at": datetime.utcnow()
            }
            for i in range(1, 6)
        ]
        
        # Create async iterator for cursor
        async def mock_cursor():
            for answer in mock_answers:
                yield answer
        
        mock_find = MagicMock()
        mock_find.sort = MagicMock(return_value=mock_cursor())
        answer_service.answer_collection.find = MagicMock(return_value=mock_find)
        
        answer_service.question_collection.count_documents = AsyncMock(return_value=12)
        
        # Get answers
        response = await answer_service.get_test_answers(sample_user_id, sample_test_id)
        
        # Assertions
        assert response.total == 5
        assert response.questions_answered == 5
        assert response.questions_remaining == 7
        assert len(response.answers) == 5
        
    @pytest.mark.asyncio
    async def test_get_answer_statistics(
        self,
        answer_service,
        sample_user_id,
        sample_test_id,
        sample_test
    ):
        """Test getting answer statistics."""
        # Setup mocks
        answer_service.test_collection.find_one = AsyncMock(
            return_value=sample_test.to_db()
        )
        
        # Mock answers with different times and types
        mock_answers = [
            {
                "time_spent_seconds": 30,
                "changed_count": 0,
                "question_type": "mcq"
            },
            {
                "time_spent_seconds": 120,
                "changed_count": 2,
                "question_type": "statement_set"
            },
            {
                "time_spent_seconds": 60,
                "changed_count": 1,
                "question_type": "mcq"
            }
        ]
        
        async def mock_cursor():
            for answer in mock_answers:
                yield answer
        
        answer_service.answer_collection.find = MagicMock(return_value=mock_cursor())
        answer_service.question_collection.count_documents = AsyncMock(return_value=12)
        
        # Get statistics
        stats = await answer_service.get_answer_statistics(sample_user_id, sample_test_id)
        
        # Assertions
        assert stats.total_time_seconds == 210  # 30 + 120 + 60
        assert stats.average_time_per_question == 70  # 210 / 3
        assert stats.fastest_answer_seconds == 30
        assert stats.slowest_answer_seconds == 120
        assert stats.total_changes == 3  # 0 + 2 + 1
        assert stats.questions_by_type["mcq"] == 2
        assert stats.questions_by_type["statement_set"] == 1
        assert stats.completion_rate == 0.25  # 3 / 12