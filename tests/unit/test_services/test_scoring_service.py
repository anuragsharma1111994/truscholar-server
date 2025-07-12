"""Unit tests for ScoringService.

Tests the comprehensive RAISEC scoring algorithm including individual answer scoring,
test-level aggregation, consistency analysis, and analytics generation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.services.scoring_service import ScoringService
from src.models.answer import Answer, AnswerMetrics, AnswerValidation, DimensionScore
from src.models.question import Question, ScoringRule
from src.models.test import Test, TestScores, RaisecProfile, ScoreAnalysis
from src.utils.constants import QuestionType, AgeGroup, RaisecDimension, TestStatus
from src.utils.exceptions import TruScholarError, ValidationError, ResourceNotFoundError
from src.schemas.test_schemas import (
    TestScoresResponse, 
    ScoringAnalyticsResponse,
    ScoringExplanationResponse
)


class TestScoringService:
    """Test suite for ScoringService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        db = Mock()
        db.find_one = AsyncMock(return_value=None)
        db.find = AsyncMock(return_value=[])
        db.count_documents = AsyncMock(return_value=0)
        db.update_one = AsyncMock()
        db.aggregate = AsyncMock(return_value=[])
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
    def scoring_service(self, mock_db, mock_cache):
        """Create ScoringService instance with mocked dependencies."""
        service = ScoringService(db=mock_db, cache=mock_cache)
        return service
    
    @pytest.fixture
    def sample_test_id(self):
        """Sample test ID."""
        return ObjectId()
    
    @pytest.fixture
    def sample_test_data(self, sample_test_id):
        """Sample test data."""
        return {
            "_id": sample_test_id,
            "user_id": ObjectId(),
            "status": "completed",
            "age_group": "18-25",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=2),
            "questions_answered": 12,
            "is_practice": False
        }
    
    @pytest.fixture
    def sample_answers(self, sample_test_id):
        """Sample answer data."""
        answers = []
        for i in range(12):
            answer = {
                "_id": ObjectId(),
                "test_id": sample_test_id,
                "question_id": ObjectId(),
                "question_number": i + 1,
                "question_type": "mcq",
                "answer_data": {"selected_option": "A"},
                "validation": {"is_valid": True, "errors": []},
                "metrics": {
                    "total_time_seconds": 30 + i * 5,
                    "revision_count": 0,
                    "confidence_level": 4
                },
                "is_final": True,
                "is_scored": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            answers.append(answer)
        return answers
    
    @pytest.fixture
    def sample_questions(self, sample_test_id):
        """Sample question data."""
        questions = []
        dimensions = list(RaisecDimension)
        for i in range(12):
            question = {
                "_id": ObjectId(),
                "test_id": sample_test_id,
                "question_number": i + 1,
                "question_type": "mcq",
                "dimensions_evaluated": [dimensions[i % 6].value],
                "scoring_rule": {
                    "mcq_option_scores": {"A": {"R": 3}, "B": {"A": 2}, "C": {"I": 1}},
                    "max_points": 3.0
                },
                "options": [
                    {"id": "A", "text": "Option A"},
                    {"id": "B", "text": "Option B"},
                    {"id": "C", "text": "Option C"}
                ]
            }
            questions.append(question)
        return questions


class TestScoreTestMethod:
    """Test the main score_test method."""
    
    @pytest.mark.asyncio
    async def test_score_test_success(
        self, 
        scoring_service, 
        sample_test_id, 
        sample_test_data,
        sample_answers,
        sample_questions,
        mock_db
    ):
        """Test successful test scoring."""
        # Mock database responses
        mock_db.find_one.side_effect = [sample_test_data]  # Test lookup
        mock_db.find.side_effect = [sample_answers, sample_questions]  # Answers and questions
        mock_db.update_one.return_value = Mock()
        
        # Mock cache miss
        scoring_service.cache_manager.get = AsyncMock(return_value=None)
        
        result = await scoring_service.score_test(sample_test_id)
        
        assert isinstance(result, TestScores)
        assert result.test_id == sample_test_id
        assert result.raisec_code is not None
        assert len(result.raisec_code) == 3  # Should be 3-letter code like "RIA"
        assert result.total_score > 0
        assert 0 <= result.consistency_score <= 100
        assert result.scored_at is not None
    
    @pytest.mark.asyncio
    async def test_score_test_from_cache(
        self,
        scoring_service,
        sample_test_id
    ):
        """Test retrieving scores from cache."""
        cached_scores = TestScores(
            test_id=sample_test_id,
            user_id=ObjectId(),
            raisec_code="RIA",
            raisec_profile=RaisecProfile(
                primary_code="RIA",
                dimension_scores={"R": 85, "I": 75, "A": 70, "S": 45, "E": 35, "C": 25}
            ),
            total_score=85.5,
            consistency_score=78.2,
            scored_at=datetime.utcnow()
        )
        
        # Mock cache hit
        scoring_service.cache_manager.get = AsyncMock(return_value=cached_scores.model_dump())
        
        result = await scoring_service.score_test(sample_test_id)
        
        assert isinstance(result, TestScores)
        assert result.test_id == sample_test_id
        assert result.raisec_code == "RIA"
        
        # Verify cache was checked
        scoring_service.cache_manager.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_score_test_force_rescore(
        self,
        scoring_service,
        sample_test_id,
        sample_test_data,
        sample_answers,
        sample_questions,
        mock_db
    ):
        """Test force rescoring bypasses cache."""
        # Mock database responses
        mock_db.find_one.side_effect = [sample_test_data]
        mock_db.find.side_effect = [sample_answers, sample_questions]
        
        # Mock cache with existing scores
        cached_scores = {"raisec_code": "OLD", "total_score": 50}
        scoring_service.cache_manager.get = AsyncMock(return_value=cached_scores)
        
        result = await scoring_service.score_test(sample_test_id, force_rescore=True)
        
        # Should have rescored, not used cache
        assert result.raisec_code != "OLD"
        assert result.total_score != 50
    
    @pytest.mark.asyncio
    async def test_score_test_invalid_test(
        self,
        scoring_service,
        sample_test_id,
        mock_db
    ):
        """Test scoring with invalid test ID."""
        # Mock test not found
        mock_db.find_one.return_value = None
        
        with pytest.raises(ResourceNotFoundError):
            await scoring_service.score_test(sample_test_id)
    
    @pytest.mark.asyncio
    async def test_score_test_incomplete_test(
        self,
        scoring_service,
        sample_test_id,
        mock_db
    ):
        """Test scoring incomplete test."""
        incomplete_test = {
            "_id": sample_test_id,
            "status": "in_progress",
            "questions_answered": 5
        }
        
        mock_db.find_one.return_value = incomplete_test
        
        with pytest.raises(ValidationError) as exc_info:
            await scoring_service.score_test(sample_test_id)
        
        assert "not completed" in str(exc_info.value).lower()


class TestScoreSingleAnswer:
    """Test the score_single_answer method."""
    
    @pytest.mark.asyncio
    async def test_score_mcq_answer(self, scoring_service):
        """Test scoring MCQ answer."""
        answer_data = {
            "question_id": ObjectId(),
            "question_type": QuestionType.MCQ,
            "answer_data": {"selected_option": "A"},
            "metrics": {"total_time_seconds": 30, "revision_count": 0}
        }
        
        question_data = {
            "dimensions_evaluated": ["R", "A"],
            "scoring_rule": {
                "mcq_option_scores": {"A": {"R": 3}, "B": {"A": 2}},
                "max_points": 3.0
            }
        }
        
        scores = await scoring_service.score_single_answer(answer_data, question_data)
        
        assert len(scores) > 0
        assert all(isinstance(score, DimensionScore) for score in scores)
        realistic_score = next((s for s in scores if s.dimension == RaisecDimension.REALISTIC), None)
        assert realistic_score is not None
        assert realistic_score.raw_score == 3.0
    
    @pytest.mark.asyncio
    async def test_score_statement_set_answer(self, scoring_service):
        """Test scoring statement set answer."""
        answer_data = {
            "question_id": ObjectId(),
            "question_type": QuestionType.STATEMENT_SET,
            "answer_data": {"ratings": {"stmt1": 5, "stmt2": 3, "stmt3": 4}},
            "metrics": {"total_time_seconds": 45, "revision_count": 1}
        }
        
        question_data = {
            "dimensions_evaluated": ["R", "A", "I"],
            "scoring_rule": {
                "statement_dimensions": {
                    "stmt1": {"R": 1.0},
                    "stmt2": {"A": 1.0}, 
                    "stmt3": {"I": 1.0}
                },
                "likert_scale_map": {1: 0, 2: 2.5, 3: 5, 4: 7.5, 5: 10}
            }
        }
        
        scores = await scoring_service.score_single_answer(answer_data, question_data)
        
        assert len(scores) == 3
        realistic_score = next((s for s in scores if s.dimension == RaisecDimension.REALISTIC), None)
        assert realistic_score is not None
        assert realistic_score.raw_score == 10.0  # Rating 5 -> 10 points
    
    @pytest.mark.asyncio 
    async def test_score_plot_day_answer(self, scoring_service):
        """Test scoring plot day answer."""
        answer_data = {
            "question_id": ObjectId(),
            "question_type": QuestionType.PLOT_DAY,
            "answer_data": {
                "placements": {
                    "9:00-12:00": ["task1", "task2"],
                    "12:00-15:00": ["task3"],
                    "not_interested": ["task4"]
                }
            },
            "metrics": {"total_time_seconds": 120, "revision_count": 2}
        }
        
        question_data = {
            "dimensions_evaluated": ["R", "A", "I", "S"],
            "scoring_rule": {
                "plot_day_task_dimensions": {
                    "task1": {"R": 1.0},
                    "task2": {"A": 1.0},
                    "task3": {"I": 1.0},
                    "task4": {"S": 1.0}
                },
                "plot_day_time_slot_weights": {
                    "9:00-12:00": 1.2,
                    "12:00-15:00": 1.0,
                    "not_interested": 0.0
                },
                "plot_day_task_points": 5.0
            }
        }
        
        scores = await scoring_service.score_single_answer(answer_data, question_data)
        
        assert len(scores) >= 3  # Should have scores for R, A, I (not S as it was "not_interested")
        realistic_score = next((s for s in scores if s.dimension == RaisecDimension.REALISTIC), None)
        assert realistic_score is not None
        assert realistic_score.raw_score == 6.0  # 5 points * 1.2 weight
    
    @pytest.mark.asyncio
    async def test_score_skipped_answer(self, scoring_service):
        """Test scoring skipped answer."""
        answer_data = {
            "question_id": ObjectId(),
            "question_type": QuestionType.MCQ,
            "is_skipped": True,
            "skip_reason": "User choice"
        }
        
        question_data = {
            "dimensions_evaluated": ["R"],
            "scoring_rule": {"max_points": 3.0}
        }
        
        scores = await scoring_service.score_single_answer(answer_data, question_data)
        
        assert len(scores) == 0  # Skipped answers don't contribute to scoring


class TestConsistencyAnalysis:
    """Test consistency analysis methods."""
    
    def test_analyze_answer_consistency(self, scoring_service):
        """Test answer consistency analysis."""
        answers = [
            {"metrics": {"total_time_seconds": 30}, "dimension_scores": [{"dimension": "R", "raw_score": 8}]},
            {"metrics": {"total_time_seconds": 35}, "dimension_scores": [{"dimension": "R", "raw_score": 7}]},
            {"metrics": {"total_time_seconds": 28}, "dimension_scores": [{"dimension": "R", "raw_score": 9}]},
            {"metrics": {"total_time_seconds": 32}, "dimension_scores": [{"dimension": "A", "raw_score": 3}]},
        ]
        
        consistency = scoring_service._analyze_answer_consistency(answers)
        
        assert isinstance(consistency, dict)
        assert "time_consistency" in consistency
        assert "score_consistency" in consistency
        assert "dimension_consistency" in consistency
        assert 0 <= consistency["overall_consistency"] <= 100
    
    def test_analyze_dimension_consistency(self, scoring_service):
        """Test dimension-specific consistency analysis."""
        dimension_scores = {
            RaisecDimension.REALISTIC: [8, 7, 9, 8, 7],
            RaisecDimension.ARTISTIC: [3, 2, 4, 3, 2],
            RaisecDimension.INVESTIGATIVE: [6, 7, 6, 5, 6]
        }
        
        consistency = scoring_service._analyze_dimension_consistency(dimension_scores)
        
        assert isinstance(consistency, dict)
        assert RaisecDimension.REALISTIC in consistency
        assert RaisecDimension.ARTISTIC in consistency
        assert RaisecDimension.INVESTIGATIVE in consistency
        
        for dim_consistency in consistency.values():
            assert "coefficient_of_variation" in dim_consistency
            assert "consistency_rating" in dim_consistency
            assert 0 <= dim_consistency["score"] <= 100


class TestRaisecProfileGeneration:
    """Test RAISEC profile generation."""
    
    def test_generate_raisec_profile(self, scoring_service):
        """Test RAISEC profile generation."""
        dimension_scores = {
            RaisecDimension.REALISTIC: 85,
            RaisecDimension.INVESTIGATIVE: 75,
            RaisecDimension.ARTISTIC: 70,
            RaisecDimension.SOCIAL: 45,
            RaisecDimension.ENTERPRISING: 35,
            RaisecDimension.CONVENTIONAL: 25
        }
        
        profile = scoring_service._generate_raisec_profile(dimension_scores)
        
        assert isinstance(profile, RaisecProfile)
        assert profile.primary_code == "RIA"
        assert len(profile.primary_code) == 3
        assert profile.dimension_scores == dimension_scores
        assert profile.profile_description is not None
    
    def test_get_raisec_code_from_scores(self, scoring_service):
        """Test RAISEC code generation from scores."""
        dimension_scores = {
            RaisecDimension.SOCIAL: 90,
            RaisecDimension.ARTISTIC: 85,
            RaisecDimension.ENTERPRISING: 80,
            RaisecDimension.REALISTIC: 40,
            RaisecDimension.INVESTIGATIVE: 35,
            RaisecDimension.CONVENTIONAL: 30
        }
        
        code = scoring_service._get_raisec_code_from_scores(dimension_scores)
        
        assert code == "SAE"
        assert len(code) == 3


class TestAnalyticsGeneration:
    """Test analytics generation methods."""
    
    @pytest.mark.asyncio
    async def test_generate_scoring_analytics(
        self,
        scoring_service,
        sample_test_id,
        sample_answers,
        sample_questions
    ):
        """Test comprehensive analytics generation."""
        test_scores = TestScores(
            test_id=sample_test_id,
            user_id=ObjectId(),
            raisec_code="RIA",
            total_score=75.5,
            consistency_score=82.3
        )
        
        analytics = await scoring_service._generate_scoring_analytics(
            sample_test_id, sample_answers, sample_questions, test_scores
        )
        
        assert isinstance(analytics, ScoreAnalysis)
        assert analytics.test_id == sample_test_id
        assert "timing_patterns" in analytics.detailed_analysis
        assert "answer_patterns" in analytics.detailed_analysis
        assert "consistency_analysis" in analytics.detailed_analysis
        assert len(analytics.insights) > 0
    
    def test_analyze_timing_patterns(self, scoring_service):
        """Test timing pattern analysis."""
        answers = [
            {"metrics": {"total_time_seconds": t}} 
            for t in [30, 35, 28, 45, 32, 38, 29, 33, 31, 36, 27, 34]
        ]
        
        patterns = scoring_service._analyze_timing_patterns(answers)
        
        assert isinstance(patterns, dict)
        assert "pattern" in patterns
        assert "speed_category" in patterns
        assert "consistency" in patterns
        assert "mean_time_seconds" in patterns
        assert patterns["pattern"] in ["consistent", "speeding_up", "slowing_down"]
    
    def test_analyze_answer_patterns(self, scoring_service):
        """Test answer pattern analysis."""
        answers = [
            {
                "question_type": "mcq",
                "answer_data": {"selected_option": "A"},
                "metrics": {"revision_count": 0}
            },
            {
                "question_type": "mcq", 
                "answer_data": {"selected_option": "B"},
                "metrics": {"revision_count": 1}
            },
            {
                "question_type": "statement_set",
                "answer_data": {"ratings": {"s1": 4, "s2": 5}},
                "metrics": {"revision_count": 0}
            }
        ]
        
        patterns = scoring_service._analyze_answer_patterns(answers)
        
        assert isinstance(patterns, dict)
        assert "revision_patterns" in patterns
        assert "response_patterns" in patterns
        assert "confidence_indicators" in patterns


class TestScoringExplanation:
    """Test scoring explanation generation."""
    
    @pytest.mark.asyncio
    async def test_get_scoring_explanation_summary(
        self,
        scoring_service,
        sample_test_id,
        mock_db
    ):
        """Test summary scoring explanation."""
        # Mock test scores
        test_scores = TestScores(
            test_id=sample_test_id,
            user_id=ObjectId(),
            raisec_code="RIA",
            raisec_profile=RaisecProfile(
                primary_code="RIA",
                dimension_scores={
                    "R": 85, "I": 75, "A": 70, 
                    "S": 45, "E": 35, "C": 25
                }
            ),
            total_score=75.5,
            consistency_score=82.3
        )
        
        scoring_service.cache_manager.get = AsyncMock(return_value=test_scores.model_dump())
        
        explanation = await scoring_service.get_scoring_explanation(
            sample_test_id, explanation_type="summary"
        )
        
        assert isinstance(explanation, dict)
        assert "overall_summary" in explanation
        assert "raisec_code_explanation" in explanation
        assert "key_strengths" in explanation
        assert "development_areas" in explanation
        assert explanation["raisec_code"] == "RIA"
    
    @pytest.mark.asyncio
    async def test_get_scoring_explanation_detailed(
        self,
        scoring_service,
        sample_test_id,
        mock_db
    ):
        """Test detailed scoring explanation."""
        # Mock test scores and analytics
        test_scores = TestScores(
            test_id=sample_test_id,
            user_id=ObjectId(),
            raisec_code="SAE",
            total_score=68.2,
            consistency_score=75.8
        )
        
        scoring_service.cache_manager.get = AsyncMock(return_value=test_scores.model_dump())
        
        explanation = await scoring_service.get_scoring_explanation(
            sample_test_id, explanation_type="detailed"
        )
        
        assert isinstance(explanation, dict)
        assert "dimension_explanations" in explanation
        assert "methodology_notes" in explanation
        assert "confidence_indicators" in explanation
        assert len(explanation["dimension_explanations"]) == 6  # All RAISEC dimensions


class TestHelperMethods:
    """Test helper and utility methods."""
    
    def test_normalize_scores(self, scoring_service):
        """Test score normalization."""
        raw_scores = {
            RaisecDimension.REALISTIC: 45.5,
            RaisecDimension.ARTISTIC: 32.0,
            RaisecDimension.INVESTIGATIVE: 38.5
        }
        
        normalized = scoring_service._normalize_scores(raw_scores)
        
        assert all(0 <= score <= 100 for score in normalized.values())
        assert max(normalized.values()) == 100  # Highest score should be normalized to 100
    
    def test_apply_time_adjustments(self, scoring_service):
        """Test time-based score adjustments."""
        base_score = 85.0
        time_seconds = 45  # Fast completion
        expected_time = 60
        
        adjusted = scoring_service._apply_time_adjustments(base_score, time_seconds, expected_time)
        
        # Fast completion should slightly reduce score due to potential rushing
        assert adjusted < base_score
        assert adjusted > 0
    
    def test_calculate_confidence_score(self, scoring_service):
        """Test confidence score calculation."""
        answer_metrics = {
            "revision_count": 1,
            "total_time_seconds": 35,
            "hesitation_score": 20
        }
        question_difficulty = 0.6
        
        confidence = scoring_service._calculate_confidence_score(answer_metrics, question_difficulty)
        
        assert 0 <= confidence <= 100
        assert isinstance(confidence, (int, float))
    
    def test_get_expected_time_for_question(self, scoring_service):
        """Test expected time calculation for different question types."""
        mcq_time = scoring_service._get_expected_time_for_question(QuestionType.MCQ, AgeGroup.YOUNG_ADULT)
        plot_day_time = scoring_service._get_expected_time_for_question(QuestionType.PLOT_DAY, AgeGroup.TEEN)
        
        assert mcq_time > 0
        assert plot_day_time > mcq_time  # Plot day should take longer
        assert isinstance(mcq_time, (int, float))
        assert isinstance(plot_day_time, (int, float))


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_score_test_database_error(
        self,
        scoring_service,
        sample_test_id,
        mock_db
    ):
        """Test handling database errors during scoring."""
        mock_db.find_one.side_effect = Exception("Database connection failed")
        
        with pytest.raises(TruScholarError) as exc_info:
            await scoring_service.score_test(sample_test_id)
        
        assert "scoring failed" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_score_invalid_answer_data(self, scoring_service):
        """Test handling invalid answer data."""
        invalid_answer = {
            "question_type": QuestionType.MCQ,
            "answer_data": None,  # Invalid
            "metrics": {}
        }
        
        question_data = {
            "dimensions_evaluated": ["R"],
            "scoring_rule": {"max_points": 3.0}
        }
        
        scores = await scoring_service.score_single_answer(invalid_answer, question_data)
        
        # Should handle gracefully and return empty scores
        assert len(scores) == 0
    
    @pytest.mark.asyncio
    async def test_get_scoring_explanation_not_scored(
        self,
        scoring_service,
        sample_test_id,
        mock_db
    ):
        """Test explanation request for unscored test."""
        scoring_service.cache_manager.get = AsyncMock(return_value=None)
        mock_db.find_one.return_value = None
        
        with pytest.raises(ResourceNotFoundError):
            await scoring_service.get_scoring_explanation(sample_test_id)


class TestCacheIntegration:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_test_scores(
        self,
        scoring_service,
        sample_test_id,
        sample_test_data,
        sample_answers,
        sample_questions,
        mock_db
    ):
        """Test caching of test scores."""
        # Mock database responses
        mock_db.find_one.side_effect = [sample_test_data]
        mock_db.find.side_effect = [sample_answers, sample_questions]
        
        # Mock cache miss initially
        scoring_service.cache_manager.get = AsyncMock(return_value=None)
        scoring_service.cache_manager.set = AsyncMock()
        
        result = await scoring_service.score_test(sample_test_id)
        
        # Verify scores were cached
        scoring_service.cache_manager.set.assert_called()
        call_args = scoring_service.cache_manager.set.call_args
        assert sample_test_id in str(call_args[0][0])  # Cache key contains test ID
    
    @pytest.mark.asyncio
    async def test_invalidate_test_cache(self, scoring_service, sample_test_id):
        """Test cache invalidation."""
        scoring_service.cache_manager.delete = AsyncMock()
        
        await scoring_service._invalidate_test_cache(sample_test_id)
        
        scoring_service.cache_manager.delete.assert_called()


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_batch_score_multiple_tests(self, scoring_service):
        """Test batch scoring multiple tests."""
        test_ids = [ObjectId() for _ in range(3)]
        
        # Mock individual scoring
        with patch.object(scoring_service, 'score_test') as mock_score:
            mock_score.return_value = TestScores(
                test_id=ObjectId(),
                user_id=ObjectId(),
                raisec_code="RIA",
                total_score=75.0,
                consistency_score=80.0
            )
            
            results = await scoring_service.batch_score_tests(test_ids)
            
            assert len(results) == 3
            assert mock_score.call_count == 3
    
    def test_scoring_configuration_validation(self, scoring_service):
        """Test scoring configuration validation."""
        valid_config = {
            "question_weights": {
                "mcq": 1.0,
                "statement_set": 0.8,
                "plot_day": 1.5
            },
            "time_adjustment_factors": {
                "very_fast": 0.85,
                "normal": 1.0,
                "slow": 1.05
            }
        }
        
        # Should not raise any errors
        is_valid = scoring_service._validate_scoring_configuration(valid_config)
        assert is_valid is True
        
        invalid_config = {
            "question_weights": {
                "mcq": -1.0,  # Invalid negative weight
            }
        }
        
        is_valid = scoring_service._validate_scoring_configuration(invalid_config)
        assert is_valid is False


@pytest.mark.integration
class TestScoringServiceIntegration:
    """Integration tests for ScoringService with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_scoring_pipeline(self):
        """Test complete scoring pipeline end-to-end."""
        # This would be a comprehensive integration test
        # in a real test environment with test database
        pass
    
    @pytest.mark.asyncio
    async def test_scoring_with_different_age_groups(self):
        """Test scoring behavior with different age groups."""
        # Test age-specific scoring variations
        pass
    
    @pytest.mark.asyncio
    async def test_scoring_performance_with_large_datasets(self):
        """Test scoring performance with large number of answers."""
        # Performance testing would go here
        pass


# Helper functions for tests

def create_mock_answer(
    question_type: QuestionType = QuestionType.MCQ,
    answer_data: Dict[str, Any] = None,
    time_seconds: int = 30,
    revision_count: int = 0
) -> Dict[str, Any]:
    """Create mock answer data for testing."""
    if answer_data is None:
        if question_type == QuestionType.MCQ:
            answer_data = {"selected_option": "A"}
        elif question_type == QuestionType.STATEMENT_SET:
            answer_data = {"ratings": {"stmt1": 4, "stmt2": 3}}
        elif question_type == QuestionType.PLOT_DAY:
            answer_data = {"placements": {"9:00-12:00": ["task1"]}}
    
    return {
        "_id": ObjectId(),
        "test_id": ObjectId(),
        "question_id": ObjectId(),
        "question_number": 1,
        "question_type": question_type.value,
        "answer_data": answer_data,
        "validation": {"is_valid": True, "errors": []},
        "metrics": {
            "total_time_seconds": time_seconds,
            "revision_count": revision_count,
            "confidence_level": 4
        },
        "is_final": True,
        "is_scored": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }


def create_mock_question(
    question_type: QuestionType = QuestionType.MCQ,
    dimensions: List[str] = None,
    scoring_rule: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create mock question data for testing."""
    if dimensions is None:
        dimensions = ["R", "A"]
    
    if scoring_rule is None:
        if question_type == QuestionType.MCQ:
            scoring_rule = {
                "mcq_option_scores": {"A": {"R": 3}, "B": {"A": 2}},
                "max_points": 3.0
            }
        elif question_type == QuestionType.STATEMENT_SET:
            scoring_rule = {
                "statement_dimensions": {"stmt1": {"R": 1.0}, "stmt2": {"A": 1.0}},
                "likert_scale_map": {1: 0, 2: 2.5, 3: 5, 4: 7.5, 5: 10}
            }
    
    return {
        "_id": ObjectId(),
        "test_id": ObjectId(),
        "question_number": 1,
        "question_type": question_type.value,
        "dimensions_evaluated": dimensions,
        "scoring_rule": scoring_rule,
        "created_at": datetime.utcnow()
    }