"""Unit tests for QuestionService.

Tests the question service functionality including LLM integration,
static question fallback, and question management operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from bson import ObjectId
from typing import Dict, Any, List

from src.services.question_service import QuestionService
from src.models.question import Question, QuestionOption, LikertStatement, PlotDayTask, ScoringRule
from src.utils.constants import QuestionType, AgeGroup, RaisecDimension
from src.utils.exceptions import TruScholarError, ValidationError, ResourceNotFoundError
from src.schemas.question_schemas import (
    QuestionGenerationStatusResponse,
    QuestionValidationResponse,
    QuestionResponse
)


class TestQuestionService:
    """Test suite for QuestionService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        db = Mock()
        db.insert_many = AsyncMock(return_value=Mock(inserted_ids=[ObjectId(), ObjectId()]))
        db.find_one = AsyncMock(return_value=None)
        db.find = AsyncMock(return_value=[])
        db.count_documents = AsyncMock(return_value=0)
        db.update_one = AsyncMock()
        return db
    
    @pytest.fixture
    def mock_cache(self):
        """Mock cache."""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.delete = AsyncMock()
        return cache
    
    @pytest.fixture
    def question_service(self, mock_db, mock_cache):
        """Create QuestionService instance with mocked dependencies."""
        service = QuestionService(db=mock_db, cache=mock_cache)
        return service
    
    @pytest.fixture
    def sample_test_id(self):
        """Sample test ID."""
        return ObjectId()
    
    @pytest.fixture
    def sample_question_distribution(self):
        """Sample question distribution."""
        return {
            QuestionType.MCQ: 4,
            QuestionType.STATEMENT_SET: 3,
            QuestionType.SCENARIO_MCQ: 2,
            QuestionType.THIS_OR_THAT: 2,
            QuestionType.PLOT_DAY: 1
        }
    
    @pytest.fixture
    def mock_llm_question_data(self):
        """Mock LLM-generated question data."""
        return {
            "question_text": "Which work environment appeals to you?",
            "options": [
                {
                    "id": "A",
                    "text": "Workshop with tools",
                    "scoring_guide": {"A": {"dimension": "R", "score": 3}}
                },
                {
                    "id": "B",
                    "text": "Creative studio",
                    "scoring_guide": {"B": {"dimension": "A", "score": 3}}
                }
            ],
            "dimensions_evaluated": ["R", "A"],
            "generation_metadata": {
                "generator": "langchain",
                "prompt_version": "1.0",
                "generated_at": "2024-01-15T10:00:00Z"
            }
        }


class TestQuestionGeneration:
    """Test question generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_test_questions_success(
        self, 
        question_service, 
        sample_test_id, 
        sample_question_distribution,
        mock_db
    ):
        """Test successful test question generation."""
        with patch('src.core.settings.feature_flags.ENABLE_DYNAMIC_QUESTIONS', False):
            questions = await question_service.generate_test_questions(
                test_id=sample_test_id,
                age_group=AgeGroup.YOUNG_ADULT,
                distribution=sample_question_distribution
            )
            
            assert len(questions) == 12  # Sum of distribution values
            assert all(isinstance(q, Question) for q in questions)
            assert all(q.test_id == sample_test_id for q in questions)
            
            # Verify questions are shuffled (numbers should be sequential after shuffle)
            question_numbers = [q.question_number for q in questions]
            assert sorted(question_numbers) == list(range(1, 13))
            
            # Verify database save was called
            mock_db.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_llm_enabled(
        self,
        question_service,
        sample_test_id,
        mock_llm_question_data
    ):
        """Test question generation with LLM enabled."""
        with patch('src.core.settings.feature_flags.ENABLE_DYNAMIC_QUESTIONS', True):
            with patch.object(question_service, '_generate_with_llm') as mock_llm_gen:
                mock_question = Question(
                    test_id=sample_test_id,
                    question_number=1,
                    question_type=QuestionType.MCQ,
                    question_text="LLM generated question",
                    age_group="18-25",
                    dimensions_evaluated=[RaisecDimension.REALISTIC],
                    is_static=False,
                    scoring_rule=ScoringRule()
                )
                mock_llm_gen.return_value = mock_question
                
                question = await question_service._generate_single_question(
                    test_id=sample_test_id,
                    question_number=1,
                    question_type=QuestionType.MCQ,
                    age_group=AgeGroup.YOUNG_ADULT
                )
                
                assert question is not None
                assert not question.is_static
                assert question.question_text == "LLM generated question"
    
    @pytest.mark.asyncio
    async def test_llm_generation_with_enhanced_parameters(
        self,
        question_service,
        sample_test_id,
        mock_llm_question_data
    ):
        """Test LLM generation with enhanced parameters."""
        with patch('src.langchain_handlers.question_generator.QuestionGenerator') as MockGenerator:
            mock_generator = MockGenerator.return_value
            mock_generator.generate_question = AsyncMock(return_value=mock_llm_question_data)
            
            with patch('src.core.settings.cache_settings.ENABLE_QUESTION_CACHE', True):
                result = await question_service._generate_with_llm(
                    test_id=sample_test_id,
                    question_number=1,
                    question_type=QuestionType.MCQ,
                    age_group=AgeGroup.YOUNG_ADULT
                )
                
                assert result is not None
                assert not result.is_static
                
                # Verify generator was called with enhanced parameters
                mock_generator.generate_question.assert_called_once()
                call_kwargs = mock_generator.generate_question.call_args[1]
                assert "dimensions_focus" in call_kwargs
                assert "context" in call_kwargs
                assert "constraints" in call_kwargs
    
    @pytest.mark.asyncio
    async def test_llm_fallback_to_static(
        self,
        question_service,
        sample_test_id
    ):
        """Test fallback to static questions when LLM fails."""
        with patch('src.core.settings.feature_flags.ENABLE_DYNAMIC_QUESTIONS', True):
            with patch.object(question_service, '_generate_with_llm') as mock_llm_gen:
                with patch.object(question_service, '_generate_static_question') as mock_static_gen:
                    # LLM generation fails
                    mock_llm_gen.return_value = None
                    
                    # Static generation succeeds
                    mock_static_question = Question(
                        test_id=sample_test_id,
                        question_number=1,
                        question_type=QuestionType.MCQ,
                        question_text="Static question",
                        age_group="18-25",
                        dimensions_evaluated=[RaisecDimension.REALISTIC],
                        is_static=True,
                        scoring_rule=ScoringRule()
                    )
                    mock_static_gen.return_value = mock_static_question
                    
                    question = await question_service._generate_single_question(
                        test_id=sample_test_id,
                        question_number=1,
                        question_type=QuestionType.MCQ,
                        age_group=AgeGroup.YOUNG_ADULT
                    )
                    
                    assert question is not None
                    assert question.is_static
                    assert question.question_text == "Static question"
                    
                    # Verify both methods were called
                    mock_llm_gen.assert_called_once()
                    mock_static_gen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_question_from_llm_data(
        self,
        question_service,
        sample_test_id,
        mock_llm_question_data
    ):
        """Test creating Question object from LLM data."""
        with patch('src.utils.formatters.QuestionFormatter') as MockFormatter:
            mock_formatter = MockFormatter.return_value
            mock_formatter.format_llm_response.return_value = {
                "question_text": "Formatted question",
                "options": [
                    {"id": "A", "text": "Option A", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                    {"id": "B", "text": "Option B", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
                ],
                "dimensions_evaluated": ["R", "A"],
                "generation_metadata": {"prompt_version": "1.0"}
            }
            
            question = question_service._create_question_from_llm_data(
                test_id=sample_test_id,
                question_number=1,
                question_type=QuestionType.MCQ,
                age_group=AgeGroup.YOUNG_ADULT,
                llm_data=mock_llm_question_data
            )
            
            assert isinstance(question, Question)
            assert question.question_text == "Formatted question"
            assert len(question.options) == 2
            assert not question.is_static
            assert question.llm_metadata is not None
            assert question.llm_metadata["prompt_version"] == "1.0"


class TestStaticQuestionGeneration:
    """Test static question generation functionality."""
    
    @pytest.mark.asyncio
    async def test_load_static_questions_success(self, question_service):
        """Test successful loading of static questions."""
        mock_questions = [
            {
                "question_text": "Static question 1",
                "options": [
                    {"id": "a", "text": "Option A", "dimensions": {"R": 1.0}},
                    {"id": "b", "text": "Option B", "dimensions": {"A": 1.0}}
                ],
                "dimensions": ["R", "A"]
            }
        ]
        
        with patch('builtins.open', mock_open_json(mock_questions)):
            with patch('pathlib.Path.exists', return_value=True):
                questions = await question_service._load_static_questions(
                    AgeGroup.YOUNG_ADULT, QuestionType.MCQ
                )
                
                assert len(questions) == 1
                assert questions[0]["question_text"] == "Static question 1"
    
    @pytest.mark.asyncio
    async def test_load_static_questions_file_not_found(self, question_service):
        """Test loading static questions when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            questions = await question_service._load_static_questions(
                AgeGroup.YOUNG_ADULT, QuestionType.MCQ
            )
            
            # Should return fallback questions
            assert len(questions) > 0
            assert "question_text" in questions[0]
    
    @pytest.mark.asyncio
    async def test_create_fallback_questions(self, question_service):
        """Test creation of fallback questions."""
        # Test MCQ fallback
        mcq_questions = question_service._create_fallback_questions(QuestionType.MCQ)
        assert len(mcq_questions) > 0
        assert "options" in mcq_questions[0]
        
        # Test statement set fallback
        stmt_questions = question_service._create_fallback_questions(QuestionType.STATEMENT_SET)
        assert len(stmt_questions) > 0
        assert "statements" in stmt_questions[0]
        
        # Test plot day fallback
        plot_questions = question_service._create_fallback_questions(QuestionType.PLOT_DAY)
        assert len(plot_questions) > 0
        assert "tasks" in plot_questions[0]


class TestQuestionValidation:
    """Test question validation functionality."""
    
    def test_validate_question_distribution_valid(self, question_service):
        """Test valid question distribution."""
        valid_distribution = {
            QuestionType.MCQ: 4,
            QuestionType.STATEMENT_SET: 3,
            QuestionType.SCENARIO_MCQ: 2,
            QuestionType.THIS_OR_THAT: 2,
            QuestionType.PLOT_DAY: 1
        }
        
        result = asyncio.run(question_service.validate_question_distribution(valid_distribution))
        assert result is True
    
    def test_validate_question_distribution_invalid_total(self, question_service):
        """Test invalid question distribution - wrong total."""
        invalid_distribution = {
            QuestionType.MCQ: 6,  # Total = 8, should be 12
            QuestionType.STATEMENT_SET: 2
        }
        
        result = asyncio.run(question_service.validate_question_distribution(invalid_distribution))
        assert result is False
    
    def test_validate_question_distribution_too_many_plot_day(self, question_service):
        """Test invalid question distribution - too many plot day questions."""
        invalid_distribution = {
            QuestionType.MCQ: 4,
            QuestionType.STATEMENT_SET: 3,
            QuestionType.SCENARIO_MCQ: 2,
            QuestionType.THIS_OR_THAT: 1,
            QuestionType.PLOT_DAY: 2  # Should be max 1
        }
        
        result = asyncio.run(question_service.validate_question_distribution(invalid_distribution))
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_question_quality(self, question_service, mock_db):
        """Test question quality validation."""
        question_id = str(ObjectId())
        user_id = "user123"
        
        # Mock question data
        question_data = {
            "_id": ObjectId(question_id),
            "test_id": ObjectId(),
            "question_text": "This is a well-formed question that should pass validation checks?",
            "question_type": "mcq",
            "age_group": "18-25",
            "dimensions_evaluated": ["R", "A"],
            "options": [
                {"id": "A", "text": "Option A"},
                {"id": "B", "text": "Option B"},
                {"id": "C", "text": "Option C"}
            ]
        }
        
        # Mock test data for ownership validation
        test_data = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id)
        }
        
        mock_db.find_one.side_effect = [question_data, test_data]
        
        result = await question_service.validate_question(question_id, user_id)
        
        assert isinstance(result, QuestionValidationResponse)
        assert result.question_id == question_id
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestQuestionRetrieval:
    """Test question retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_test_questions_from_cache(self, question_service, mock_cache):
        """Test retrieving questions from cache."""
        test_id = str(ObjectId())
        cached_questions = [
            {
                "id": str(ObjectId()),
                "question_text": "Cached question",
                "question_type": "mcq"
            }
        ]
        
        # Mock cache hit
        question_service.cache_manager.get = AsyncMock(return_value=cached_questions)
        
        result = await question_service.get_test_questions(test_id)
        
        assert len(result) == 1
        assert isinstance(result[0], QuestionResponse)
    
    @pytest.mark.asyncio
    async def test_get_test_questions_from_database(self, question_service, mock_db):
        """Test retrieving questions from database when cache misses."""
        test_id = str(ObjectId())
        
        # Mock cache miss
        question_service.cache_manager.get = AsyncMock(return_value=None)
        
        # Mock database response
        question_docs = [
            {
                "_id": ObjectId(),
                "test_id": ObjectId(test_id),
                "question_number": 1,
                "question_type": "mcq",
                "question_text": "Database question",
                "age_group": "18-25",
                "dimensions_evaluated": ["R"],
                "is_static": True,
                "scoring_rule": {},
                "created_at": "2024-01-15T10:00:00Z"
            }
        ]
        
        mock_db.find.return_value = question_docs
        
        result = await question_service.get_test_questions(test_id)
        
        assert len(result) == 1
        assert isinstance(result[0], QuestionResponse)
        assert result[0].question_text == "Database question"
    
    @pytest.mark.asyncio
    async def test_get_question_with_ownership_validation(self, question_service, mock_db):
        """Test getting single question with ownership validation."""
        question_id = str(ObjectId())
        user_id = "user123"
        
        # Mock question data
        question_data = {
            "_id": ObjectId(question_id),
            "test_id": ObjectId(),
            "question_text": "Test question",
            "question_type": "mcq",
            "age_group": "18-25",
            "dimensions_evaluated": ["R"],
            "is_static": True,
            "scoring_rule": {},
            "created_at": "2024-01-15T10:00:00Z"
        }
        
        # Mock test data for ownership validation
        test_data = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id)
        }
        
        mock_db.find_one.side_effect = [question_data, test_data]
        
        result = await question_service.get_question(question_id, user_id)
        
        assert isinstance(result, QuestionResponse)
        assert result.question_text == "Test question"


class TestQuestionRegeneration:
    """Test question regeneration functionality."""
    
    @pytest.mark.asyncio
    async def test_regenerate_question_success(self, question_service, mock_db):
        """Test successful question regeneration."""
        test_id = str(ObjectId())
        question_number = 1
        user_id = "user123"
        
        # Mock existing question
        existing_question_data = {
            "_id": ObjectId(),
            "test_id": ObjectId(test_id),
            "question_number": 1,
            "question_type": "mcq",
            "question_text": "Old question",
            "age_group": "18-25",
            "dimensions_evaluated": ["R"],
            "is_static": True,
            "scoring_rule": {}
        }
        
        # Mock test data for ownership validation
        test_data = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id)
        }
        
        mock_db.find_one.side_effect = [test_data, existing_question_data]
        
        # Mock question generation
        with patch.object(question_service, '_generate_single_question') as mock_gen:
            new_question = Question(
                test_id=ObjectId(test_id),
                question_number=1,
                question_type=QuestionType.MCQ,
                question_text="New regenerated question",
                age_group="18-25",
                dimensions_evaluated=[RaisecDimension.REALISTIC],
                is_static=False,
                scoring_rule=ScoringRule()
            )
            mock_gen.return_value = new_question
            
            result = await question_service.regenerate_question(
                test_id, question_number, user_id, reason="User requested"
            )
            
            assert isinstance(result, QuestionResponse)
            assert result.question_text == "New regenerated question"
            
            # Verify database update was called
            mock_db.update_one.assert_called_once()


class TestHelperMethods:
    """Test helper methods."""
    
    def test_get_balanced_dimensions_for_test(self, question_service):
        """Test getting balanced dimensions for test."""
        test_id = ObjectId()
        
        # Test different question types
        mcq_dims = question_service._get_balanced_dimensions_for_test(test_id, QuestionType.MCQ)
        assert len(mcq_dims) == 3
        assert all(isinstance(d, RaisecDimension) for d in mcq_dims)
        
        plot_day_dims = question_service._get_balanced_dimensions_for_test(test_id, QuestionType.PLOT_DAY)
        assert len(plot_day_dims) == 6  # All dimensions
    
    def test_get_test_context(self, question_service):
        """Test getting test context."""
        test_id = ObjectId()
        
        teen_context = question_service._get_test_context(test_id, AgeGroup.TEEN)
        assert "school activities" in teen_context.lower()
        
        adult_context = question_service._get_test_context(test_id, AgeGroup.ADULT)
        assert "career advancement" in adult_context.lower()
    
    def test_get_generation_constraints(self, question_service):
        """Test getting generation constraints."""
        teen_constraints = question_service._get_generation_constraints(
            AgeGroup.TEEN, QuestionType.MCQ
        )
        assert teen_constraints["language_complexity"] == "simple"
        assert teen_constraints["avoid_career_jargon"] is True
        
        adult_constraints = question_service._get_generation_constraints(
            AgeGroup.ADULT, QuestionType.MCQ
        )
        assert adult_constraints["include_leadership"] is True
    
    def test_parse_dimension_weights(self, question_service):
        """Test parsing dimension weights from scoring guide."""
        scoring_guide = {
            "A": {"dimension": "R", "score": 3},
            "B": {"dimension": "A", "score": 2}
        }
        
        weights = question_service._parse_dimension_weights(scoring_guide)
        
        assert RaisecDimension.REALISTIC in weights
        assert weights[RaisecDimension.REALISTIC] == 3.0
        assert RaisecDimension.ARTISTIC in weights
        assert weights[RaisecDimension.ARTISTIC] == 2.0
    
    def test_estimate_question_time(self, question_service):
        """Test question time estimation."""
        mcq_time = question_service._estimate_question_time(QuestionType.MCQ)
        assert mcq_time == 30
        
        plot_day_time = question_service._estimate_question_time(QuestionType.PLOT_DAY)
        assert plot_day_time == 180  # 3 minutes
        
        # Test unknown type fallback
        unknown_time = question_service._estimate_question_time("unknown_type")
        assert unknown_time == 45  # Default
    
    def test_get_age_group_from_string(self, question_service):
        """Test converting age group string to enum."""
        # Test valid age group strings
        teen_group = question_service._get_age_group_from_string("13-17")
        assert teen_group == AgeGroup.TEEN
        
        young_adult_group = question_service._get_age_group_from_string("18-25")
        assert young_adult_group == AgeGroup.YOUNG_ADULT
        
        adult_group = question_service._get_age_group_from_string("26-35")
        assert adult_group == AgeGroup.ADULT
        
        # Test invalid age group
        with pytest.raises(ValueError):
            question_service._get_age_group_from_string("invalid")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_database_error(self, question_service, mock_db):
        """Test handling database errors during question generation."""
        mock_db.insert_many.side_effect = Exception("Database connection failed")
        
        with pytest.raises(TruScholarError) as exc_info:
            await question_service.generate_test_questions(
                test_id=ObjectId(),
                age_group=AgeGroup.YOUNG_ADULT,
                distribution={QuestionType.MCQ: 1}
            )
        
        assert "Question generation failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_test_ownership_not_found(self, question_service, mock_db):
        """Test validation when test is not found."""
        mock_db.find_one.return_value = None
        
        with pytest.raises(ResourceNotFoundError):
            await question_service.validate_test_ownership("nonexistent_id", "user123")
    
    @pytest.mark.asyncio
    async def test_validate_test_ownership_permission_denied(self, question_service, mock_db):
        """Test validation when user doesn't own test."""
        test_data = {
            "_id": ObjectId(),
            "user_id": ObjectId("different_user")
        }
        mock_db.find_one.return_value = test_data
        
        with pytest.raises(PermissionError):
            await question_service.validate_test_ownership(str(ObjectId()), "user123")


class TestAnalyticsAndMetrics:
    """Test analytics and metrics functionality."""
    
    @pytest.mark.asyncio
    async def test_record_generation_success(self, question_service):
        """Test recording successful generation."""
        metadata = {
            "generation_time": 2.5,
            "generation_attempt": 1,
            "prompt_version": "1.0"
        }
        
        # Should not raise any errors
        await question_service._record_generation_success(
            QuestionType.MCQ, AgeGroup.YOUNG_ADULT, metadata
        )
    
    @pytest.mark.asyncio
    async def test_record_generation_failure(self, question_service):
        """Test recording generation failure."""
        # Should not raise any errors
        await question_service._record_generation_failure(
            QuestionType.MCQ, AgeGroup.YOUNG_ADULT, "LLM timeout"
        )
    
    @pytest.mark.asyncio
    async def test_get_question_distribution(self, question_service):
        """Test getting question distribution information."""
        distribution = await question_service.get_question_distribution()
        
        assert "standard" in distribution
        assert "total_questions" in distribution
        assert "weights" in distribution
        assert distribution["total_questions"] == 12
        
        # Test with age group filter
        teen_distribution = await question_service.get_question_distribution("13-17")
        assert "age_group" in teen_distribution
        assert "notes" in teen_distribution


# Helper function for mocking file operations
def mock_open_json(data):
    """Mock open function that returns JSON data."""
    import json
    from unittest.mock import mock_open
    
    json_data = json.dumps(data)
    return mock_open(read_data=json_data)


@pytest.mark.integration
class TestQuestionServiceIntegration:
    """Integration tests for QuestionService with real-like scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_question_generation_flow(self):
        """Test complete question generation flow."""
        # This would be a more comprehensive integration test
        # in a real test environment with test database
        pass
    
    @pytest.mark.asyncio
    async def test_performance_with_large_batches(self):
        """Test performance with large question batches."""
        # Performance testing would go here
        pass