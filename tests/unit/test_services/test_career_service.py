"""Comprehensive unit tests for career service functionality.

This module provides thorough testing coverage for the CareerService class,
including career recommendation generation, matching algorithms, market analysis,
and all core career-related business logic.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.services.career_service import CareerService, CareerMatchingResult, CareerAnalysisContext
from src.schemas.career_schemas import (
    CareerRecommendationRequest,
    CareerRecommendationResponse,
    CareerMatch,
    CareerDetailResponse,
    CareerSearchRequest,
    CareerSearchResponse,
    CareerPathRequest,
    CareerPathResponse,
    CareerTrendsResponse,
    EducationLevel,
    ExperienceLevel,
    WorkEnvironment,
    CareerFieldCategory
)
from src.utils.constants import RaisecDimension
from src.utils.exceptions import ValidationError, ResourceNotFoundError, TruScholarError
from src.langchain_handlers.career_recommender import RecommendationContext


class TestCareerService:
    """Test suite for CareerService class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for CareerService."""
        mock_career_recommender = Mock()
        mock_career_recommender.generate_career_recommendations = AsyncMock()
        mock_career_recommender.generate_career_insights = AsyncMock()
        mock_career_recommender.generate_career_comparison = AsyncMock()
        
        mock_test_service = Mock()
        mock_test_service.get_test_results = AsyncMock()
        
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        
        return {
            'career_recommender': mock_career_recommender,
            'test_service': mock_test_service,
            'cache': mock_cache
        }

    @pytest.fixture
    def career_service(self, mock_dependencies):
        """Create CareerService instance with mocked dependencies."""
        return CareerService(
            career_recommender=mock_dependencies['career_recommender'],
            test_service=mock_dependencies['test_service'],
            cache_client=mock_dependencies['cache']
        )

    @pytest.fixture
    def sample_raisec_scores(self):
        """Sample RAISEC scores for testing."""
        return {
            "R": 45.5,
            "A": 30.2,
            "I": 85.7,
            "S": 25.8,
            "E": 70.3,
            "C": 65.1
        }

    @pytest.fixture
    def sample_user_profile(self):
        """Sample user profile for testing."""
        return {
            "age": 22,
            "education_level": "bachelor",
            "experience_level": "entry_level",
            "location": "Bangalore",
            "interests": ["technology", "problem_solving", "innovation"],
            "work_preferences": {
                "environment": "hybrid",
                "team_size": "medium",
                "autonomy_level": "high"
            }
        }

    @pytest.fixture
    def sample_career_database(self):
        """Sample career database for testing."""
        return {
            "software_engineer": {
                "id": "software_engineer",
                "title": "Software Engineer",
                "category": "information_technology",
                "raisec_profile": {"R": 25, "A": 35, "I": 85, "S": 20, "E": 30, "C": 70},
                "salary_data": {"entry_level": 500000, "median": 800000, "experienced": 1200000},
                "job_market": {"growth_rate": 15.5, "demand": "high"}
            },
            "data_scientist": {
                "id": "data_scientist", 
                "title": "Data Scientist",
                "category": "information_technology",
                "raisec_profile": {"R": 20, "A": 25, "I": 95, "S": 15, "E": 40, "C": 75},
                "salary_data": {"entry_level": 600000, "median": 1000000, "experienced": 1800000},
                "job_market": {"growth_rate": 22.3, "demand": "very_high"}
            },
            "product_manager": {
                "id": "product_manager",
                "title": "Product Manager", 
                "category": "business",
                "raisec_profile": {"R": 15, "A": 45, "I": 70, "S": 60, "E": 85, "C": 50},
                "salary_data": {"entry_level": 800000, "median": 1500000, "experienced": 2500000},
                "job_market": {"growth_rate": 18.7, "demand": "high"}
            }
        }

    @pytest.fixture
    def sample_recommendation_request(self):
        """Sample career recommendation request."""
        return CareerRecommendationRequest(
            test_id="test_12345",
            max_recommendations=10,
            include_market_data=True,
            recommendation_type="comprehensive",
            focus_areas=["technology", "innovation"],
            filters={
                "education_levels": ["bachelor", "master"],
                "experience_levels": ["entry_level", "early_career"],
                "work_environments": ["office", "remote", "hybrid"]
            }
        )


class TestCareerRecommendationGeneration:
    """Test career recommendation generation functionality."""

    @pytest.mark.asyncio
    async def test_get_career_recommendations_success(self, career_service, mock_dependencies, 
                                                     sample_recommendation_request, sample_raisec_scores,
                                                     sample_user_profile, sample_career_database):
        """Test successful career recommendation generation."""
        # Mock test service response
        mock_test_result = {
            "raisec_scores": sample_raisec_scores,
            "raisec_code": "IEC",
            "user_profile": sample_user_profile
        }
        mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
        
        # Mock career recommender response
        mock_recommendations = {
            "recommendations": [
                {
                    "career_id": "software_engineer",
                    "match_score": 92.5,
                    "confidence": 0.89,
                    "reasons": ["Strong I-E alignment", "Technical skills match"]
                },
                {
                    "career_id": "data_scientist", 
                    "match_score": 88.3,
                    "confidence": 0.85,
                    "reasons": ["Excellent I dimension fit", "Analytics focus"]
                }
            ],
            "analysis_summary": "Strong investigative-enterprising profile",
            "confidence_level": 0.87
        }
        mock_dependencies['career_recommender'].generate_career_recommendations.return_value = mock_recommendations
        
        # Patch career database loading
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            result = await career_service.get_career_recommendations(sample_recommendation_request)
        
        # Assertions
        assert isinstance(result, CareerRecommendationResponse)
        assert len(result.recommendations) == 2
        assert result.recommendations[0].match_score == 92.5
        assert result.analysis_summary == "Strong investigative-enterprising profile"
        assert result.confidence_level == 0.87
        
        # Verify dependencies were called correctly
        mock_dependencies['test_service'].get_test_results.assert_called_once_with(sample_recommendation_request.test_id)
        mock_dependencies['career_recommender'].generate_career_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_career_recommendations_test_not_found(self, career_service, mock_dependencies,
                                                           sample_recommendation_request):
        """Test career recommendation generation when test is not found."""
        # Mock test service to raise ResourceNotFoundError
        mock_dependencies['test_service'].get_test_results.side_effect = ResourceNotFoundError("Test not found")
        
        with pytest.raises(ResourceNotFoundError) as exc_info:
            await career_service.get_career_recommendations(sample_recommendation_request)
        
        assert "Test not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_career_recommendations_with_caching(self, career_service, mock_dependencies,
                                                         sample_recommendation_request, sample_raisec_scores):
        """Test career recommendation generation with caching."""
        # Mock cached result
        cached_result = {
            "recommendations": [{"career_id": "cached_career", "match_score": 85.0}],
            "cached": True,
            "cache_timestamp": datetime.utcnow().isoformat()
        }
        mock_dependencies['cache'].get.return_value = json.dumps(cached_result)
        
        result = await career_service.get_career_recommendations(sample_recommendation_request)
        
        # Should return cached result
        assert result.recommendations[0].career_id == "cached_career"
        assert result.recommendations[0].match_score == 85.0
        
        # Should not call test service or recommender
        mock_dependencies['test_service'].get_test_results.assert_not_called()
        mock_dependencies['career_recommender'].generate_career_recommendations.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_career_recommendations_validation_error(self, career_service, mock_dependencies):
        """Test career recommendation generation with validation error."""
        # Create invalid request
        invalid_request = CareerRecommendationRequest(
            test_id="",  # Invalid empty test ID
            max_recommendations=-1,  # Invalid negative number
            include_market_data=True
        )
        
        with pytest.raises(ValidationError) as exc_info:
            await career_service.get_career_recommendations(invalid_request)
        
        assert "validation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_career_recommendations_algorithm_failure(self, career_service, mock_dependencies,
                                                              sample_recommendation_request, sample_raisec_scores,
                                                              sample_user_profile):
        """Test handling of recommendation algorithm failure."""
        # Mock test service response
        mock_test_result = {
            "raisec_scores": sample_raisec_scores,
            "raisec_code": "IEC", 
            "user_profile": sample_user_profile
        }
        mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
        
        # Mock career recommender to raise exception
        mock_dependencies['career_recommender'].generate_career_recommendations.side_effect = Exception("Algorithm failed")
        
        with patch.object(career_service, '_load_career_database', return_value={}):
            with pytest.raises(TruScholarError) as exc_info:
                await career_service.get_career_recommendations(sample_recommendation_request)
        
        assert "recommendation generation failed" in str(exc_info.value).lower()


class TestCareerMatching:
    """Test career matching algorithms."""

    def test_calculate_career_match_success(self, career_service, sample_raisec_scores, 
                                          sample_user_profile, sample_career_database):
        """Test successful career match calculation."""
        context = CareerAnalysisContext(
            user_raisec_scores=sample_raisec_scores,
            user_raisec_code="IEC",
            user_preferences=sample_user_profile.get("work_preferences", {}),
            include_market_data=True
        )
        
        career_data = sample_career_database["software_engineer"]
        
        result = career_service._calculate_career_match(career_data, context)
        
        assert isinstance(result, CareerMatchingResult)
        assert 0 <= result.overall_score <= 100
        assert 0 <= result.raisec_similarity <= 100
        assert 0 <= result.preference_alignment <= 100
        assert 0 <= result.requirement_feasibility <= 100
        assert isinstance(result.confidence_level, float)
        assert 0 <= result.confidence_level <= 1

    def test_calculate_raisec_similarity(self, career_service, sample_raisec_scores):
        """Test RAISEC similarity calculation."""
        # Test with identical profiles (should be high similarity)
        career_raisec = sample_raisec_scores.copy()
        similarity = career_service._calculate_raisec_similarity(sample_raisec_scores, career_raisec)
        assert similarity > 95  # Nearly perfect match
        
        # Test with opposite profiles (should be low similarity)
        opposite_profile = {dim: 100 - score for dim, score in sample_raisec_scores.items()}
        similarity = career_service._calculate_raisec_similarity(sample_raisec_scores, opposite_profile)
        assert similarity < 30  # Poor match
        
        # Test with partially matching profile
        partial_match = sample_raisec_scores.copy()
        partial_match["I"] = 80  # Keep strong investigative
        partial_match["E"] = 65  # Keep moderate enterprising
        partial_match["R"] = 10  # Low realistic (different from user)
        
        similarity = career_service._calculate_raisec_similarity(sample_raisec_scores, partial_match)
        assert 60 <= similarity <= 90  # Moderate to good match

    def test_calculate_preference_alignment(self, career_service, sample_user_profile):
        """Test preference alignment calculation."""
        user_preferences = sample_user_profile.get("work_preferences", {})
        
        # Test with matching career characteristics
        matching_career = {
            "work_environment": ["hybrid"],  # Matches user preference
            "team_collaboration": "moderate",
            "autonomy_level": "high",  # Matches user preference
            "travel_requirements": "minimal"
        }
        
        alignment = career_service._calculate_preference_alignment(matching_career, 
                                                                 CareerAnalysisContext(
                                                                     user_raisec_scores={},
                                                                     user_raisec_code="",
                                                                     user_preferences=user_preferences
                                                                 ))
        assert alignment > 70  # Good alignment
        
        # Test with conflicting career characteristics
        conflicting_career = {
            "work_environment": ["office_only"],  # Conflicts with hybrid preference
            "team_collaboration": "intensive",
            "autonomy_level": "low",  # Conflicts with high autonomy preference
            "travel_requirements": "extensive"
        }
        
        alignment = career_service._calculate_preference_alignment(conflicting_career,
                                                                 CareerAnalysisContext(
                                                                     user_raisec_scores={},
                                                                     user_raisec_code="",
                                                                     user_preferences=user_preferences
                                                                 ))
        assert alignment < 50  # Poor alignment

    def test_calculate_requirement_feasibility(self, career_service, sample_user_profile):
        """Test requirement feasibility calculation."""
        # Test with achievable requirements
        achievable_career = {
            "education_requirements": ["bachelor"],  # User has bachelor level
            "experience_requirements": "entry_level",  # Matches user level
            "skill_requirements": ["programming", "analysis"],
            "certification_requirements": []
        }
        
        feasibility = career_service._calculate_requirement_feasibility(achievable_career,
                                                                       CareerAnalysisContext(
                                                                           user_raisec_scores={},
                                                                           user_raisec_code="",
                                                                           user_preferences={},
                                                                           user_background=sample_user_profile
                                                                       ))
        assert feasibility > 80  # High feasibility
        
        # Test with challenging requirements
        challenging_career = {
            "education_requirements": ["phd"],  # Higher than user's bachelor
            "experience_requirements": "senior_level",  # Higher than entry level
            "skill_requirements": ["advanced_research", "leadership", "domain_expertise"],
            "certification_requirements": ["professional_license"]
        }
        
        feasibility = career_service._calculate_requirement_feasibility(challenging_career,
                                                                       CareerAnalysisContext(
                                                                           user_raisec_scores={},
                                                                           user_raisec_code="",
                                                                           user_preferences={},
                                                                           user_background=sample_user_profile
                                                                       ))
        assert feasibility < 50  # Low feasibility


class TestCareerDetails:
    """Test career details functionality."""

    @pytest.mark.asyncio
    async def test_get_career_details_success(self, career_service, mock_dependencies, sample_career_database):
        """Test successful career details retrieval."""
        career_id = "software_engineer"
        test_id = "test_12345"
        
        # Mock career database
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            # Mock test results if test_id provided
            if test_id:
                mock_test_result = {
                    "raisec_scores": {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70, "C": 65},
                    "raisec_code": "IEC"
                }
                mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
            
            # Mock career insights generation
            mock_insights = {
                "match_analysis": "Strong alignment with investigative dimension",
                "growth_potential": "Excellent career growth prospects",
                "challenges": "May require continuous learning",
                "personalized_advice": "Focus on developing technical leadership skills"
            }
            mock_dependencies['career_recommender'].generate_career_insights.return_value = mock_insights
            
            result = await career_service.get_career_details(
                career_id=career_id,
                test_id=test_id,
                include_similar=True,
                include_path=True
            )
        
        assert isinstance(result, CareerDetailResponse)
        assert result.career.id == career_id
        assert result.career.title == "Software Engineer"
        assert result.personalized_insights is not None
        assert result.similar_careers is not None
        assert result.career_path is not None

    @pytest.mark.asyncio
    async def test_get_career_details_not_found(self, career_service, sample_career_database):
        """Test career details retrieval for non-existent career."""
        career_id = "non_existent_career"
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            with pytest.raises(ResourceNotFoundError) as exc_info:
                await career_service.get_career_details(career_id=career_id)
        
        assert "Career not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_career_details_without_personalization(self, career_service, sample_career_database):
        """Test career details retrieval without personalization."""
        career_id = "software_engineer"
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            result = await career_service.get_career_details(
                career_id=career_id,
                test_id=None,  # No personalization
                include_similar=False,
                include_path=False
            )
        
        assert isinstance(result, CareerDetailResponse)
        assert result.career.id == career_id
        assert result.personalized_insights is None
        assert result.similar_careers is None
        assert result.career_path is None


class TestCareerSearch:
    """Test career search functionality."""

    @pytest.mark.asyncio
    async def test_search_careers_success(self, career_service, sample_career_database):
        """Test successful career search."""
        search_request = CareerSearchRequest(
            query="software engineer",
            categories=[CareerFieldCategory.INFORMATION_TECHNOLOGY],
            education_levels=[EducationLevel.BACHELOR],
            experience_levels=[ExperienceLevel.ENTRY_LEVEL],
            work_environments=[WorkEnvironment.HYBRID],
            salary_range={"min": 400000, "max": 1000000},
            page=1,
            page_size=10
        )
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            result = await career_service.search_careers(search_request)
        
        assert isinstance(result, CareerSearchResponse)
        assert len(result.results) > 0
        assert result.total_count > 0
        assert result.page == 1
        assert result.page_size == 10

    @pytest.mark.asyncio
    async def test_search_careers_with_filters(self, career_service, sample_career_database):
        """Test career search with various filters."""
        # Test with salary filter
        search_request = CareerSearchRequest(
            salary_range={"min": 800000, "max": 2000000},
            page=1,
            page_size=10
        )
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            result = await career_service.search_careers(search_request)
        
        # Should only return high-salary careers
        for career in result.results:
            career_data = sample_career_database[career.id]
            assert career_data["salary_data"]["median"] >= 800000

    @pytest.mark.asyncio
    async def test_search_careers_no_results(self, career_service, sample_career_database):
        """Test career search with no matching results."""
        search_request = CareerSearchRequest(
            query="non_existent_career_type",
            salary_range={"min": 10000000, "max": 20000000},  # Unrealistic salary range
            page=1,
            page_size=10
        )
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            result = await career_service.search_careers(search_request)
        
        assert isinstance(result, CareerSearchResponse)
        assert len(result.results) == 0
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_search_careers_pagination(self, career_service, sample_career_database):
        """Test career search pagination."""
        # Create larger career database for pagination testing
        extended_database = sample_career_database.copy()
        for i in range(15):
            extended_database[f"career_{i}"] = {
                "id": f"career_{i}",
                "title": f"Career {i}",
                "category": "information_technology"
            }
        
        search_request = CareerSearchRequest(
            page=2,
            page_size=5
        )
        
        with patch.object(career_service, '_load_career_database', return_value=extended_database):
            result = await career_service.search_careers(search_request)
        
        assert result.page == 2
        assert result.page_size == 5
        assert len(result.results) <= 5


class TestCareerPathAnalysis:
    """Test career path analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_career_path_success(self, career_service, mock_dependencies, sample_career_database):
        """Test successful career path analysis."""
        path_request = CareerPathRequest(
            current_career="software_engineer",
            target_career="product_manager",
            user_profile={
                "education": "bachelor",
                "experience_years": 3,
                "skills": ["programming", "problem_solving"]
            },
            timeline="2_years"
        )
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            # Mock career recommender response
            mock_path_analysis = {
                "feasibility_score": 78.5,
                "transition_steps": [
                    "Develop product management skills",
                    "Gain customer interaction experience",
                    "Learn business analysis"
                ],
                "skill_gaps": ["product strategy", "market analysis"],
                "timeline_assessment": "Realistic for 2-year timeline",
                "alternative_paths": ["Technical Lead", "Engineering Manager"]
            }
            mock_dependencies['career_recommender'].generate_career_insights.return_value = mock_path_analysis
            
            result = await career_service.analyze_career_path(path_request)
        
        assert isinstance(result, CareerPathResponse)
        assert result.feasibility_score == 78.5
        assert len(result.transition_steps) > 0
        assert len(result.skill_gaps) > 0

    @pytest.mark.asyncio
    async def test_analyze_career_path_invalid_careers(self, career_service, sample_career_database):
        """Test career path analysis with invalid career IDs."""
        path_request = CareerPathRequest(
            current_career="non_existent_career",
            target_career="another_non_existent_career",
            user_profile={},
            timeline="1_year"
        )
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            with pytest.raises(ResourceNotFoundError) as exc_info:
                await career_service.analyze_career_path(path_request)
        
        assert "Career not found" in str(exc_info.value)


class TestCareerTrends:
    """Test career trends functionality."""

    @pytest.mark.asyncio
    async def test_get_career_trends_success(self, career_service):
        """Test successful career trends retrieval."""
        location = "Bangalore"
        timeframe = "5_years"
        
        # Mock market data loading
        mock_trends_data = {
            "trending_careers": [
                {
                    "career_id": "ai_engineer",
                    "title": "AI Engineer",
                    "growth_rate": 35.2,
                    "demand_level": "very_high"
                },
                {
                    "career_id": "data_scientist",
                    "title": "Data Scientist", 
                    "growth_rate": 28.7,
                    "demand_level": "high"
                }
            ],
            "emerging_fields": ["quantum_computing", "blockchain_development"],
            "declining_fields": ["traditional_system_admin"],
            "market_insights": "Technology sector showing strongest growth"
        }
        
        with patch.object(career_service, '_load_market_trends', return_value=mock_trends_data):
            result = await career_service.get_career_trends(
                location=location,
                timeframe=timeframe
            )
        
        assert isinstance(result, CareerTrendsResponse)
        assert len(result.trending_careers) > 0
        assert len(result.emerging_fields) > 0
        assert result.market_insights is not None

    @pytest.mark.asyncio
    async def test_get_career_trends_no_location(self, career_service):
        """Test career trends retrieval without location filter."""
        timeframe = "3_years"
        
        mock_trends_data = {
            "trending_careers": [],
            "emerging_fields": [],
            "declining_fields": [],
            "market_insights": "National market trends"
        }
        
        with patch.object(career_service, '_load_market_trends', return_value=mock_trends_data):
            result = await career_service.get_career_trends(timeframe=timeframe)
        
        assert isinstance(result, CareerTrendsResponse)
        assert result.market_insights == "National market trends"


class TestCareerServiceUtilities:
    """Test career service utility methods."""

    def test_validate_raisec_scores(self, career_service):
        """Test RAISEC scores validation."""
        # Valid scores
        valid_scores = {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70, "C": 65}
        assert career_service._validate_raisec_scores(valid_scores) is True
        
        # Invalid scores - missing dimension
        invalid_scores_missing = {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70}
        assert career_service._validate_raisec_scores(invalid_scores_missing) is False
        
        # Invalid scores - out of range
        invalid_scores_range = {"R": 45, "A": 30, "I": 150, "S": -10, "E": 70, "C": 65}
        assert career_service._validate_raisec_scores(invalid_scores_range) is False
        
        # Invalid scores - wrong type
        invalid_scores_type = {"R": "high", "A": 30, "I": 85, "S": 25, "E": 70, "C": 65}
        assert career_service._validate_raisec_scores(invalid_scores_type) is False

    def test_generate_cache_key(self, career_service):
        """Test cache key generation."""
        request = CareerRecommendationRequest(
            test_id="test_12345",
            max_recommendations=10,
            include_market_data=True
        )
        
        cache_key = career_service._generate_cache_key(request)
        
        assert isinstance(cache_key, str)
        assert "test_12345" in cache_key
        assert "career_recommendations" in cache_key

    def test_filter_careers_by_criteria(self, career_service, sample_career_database):
        """Test career filtering by criteria."""
        criteria = {
            "categories": ["information_technology"],
            "min_salary": 600000,
            "max_salary": 1500000
        }
        
        filtered_careers = career_service._filter_careers_by_criteria(sample_career_database, criteria)
        
        # Should include data_scientist (salary 600k-1800k in IT) but exclude product_manager (business category)
        career_ids = [career["id"] for career in filtered_careers]
        assert "data_scientist" in career_ids
        assert "product_manager" not in career_ids

    def test_calculate_match_confidence(self, career_service):
        """Test match confidence calculation."""
        # High confidence scenario
        high_confidence_scores = {
            "raisec_similarity": 92.5,
            "preference_alignment": 88.3,
            "requirement_feasibility": 95.0
        }
        confidence = career_service._calculate_match_confidence(high_confidence_scores)
        assert confidence > 0.85
        
        # Low confidence scenario
        low_confidence_scores = {
            "raisec_similarity": 45.2,
            "preference_alignment": 38.7,
            "requirement_feasibility": 52.1
        }
        confidence = career_service._calculate_match_confidence(low_confidence_scores)
        assert confidence < 0.60

    def test_sort_recommendations_by_score(self, career_service):
        """Test recommendation sorting by match score."""
        recommendations = [
            {"career_id": "career_a", "match_score": 75.5},
            {"career_id": "career_b", "match_score": 92.3},
            {"career_id": "career_c", "match_score": 83.7}
        ]
        
        sorted_recommendations = career_service._sort_recommendations_by_score(recommendations)
        
        # Should be sorted in descending order by match_score
        assert sorted_recommendations[0]["career_id"] == "career_b"  # Highest score
        assert sorted_recommendations[1]["career_id"] == "career_c"  # Middle score
        assert sorted_recommendations[2]["career_id"] == "career_a"  # Lowest score


class TestCareerServiceEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_empty_career_database(self, career_service, mock_dependencies, sample_recommendation_request):
        """Test handling of empty career database."""
        mock_test_result = {
            "raisec_scores": {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70, "C": 65},
            "raisec_code": "IEC",
            "user_profile": {}
        }
        mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
        
        with patch.object(career_service, '_load_career_database', return_value={}):
            result = await career_service.get_career_recommendations(sample_recommendation_request)
        
        assert isinstance(result, CareerRecommendationResponse)
        assert len(result.recommendations) == 0

    @pytest.mark.asyncio
    async def test_malformed_test_results(self, career_service, mock_dependencies, sample_recommendation_request):
        """Test handling of malformed test results."""
        # Mock malformed test result
        malformed_result = {
            "raisec_scores": {"R": "invalid"},  # Invalid score format
            "raisec_code": None,  # Missing required field
            "user_profile": "not_a_dict"  # Wrong type
        }
        mock_dependencies['test_service'].get_test_results.return_value = malformed_result
        
        with pytest.raises(ValidationError) as exc_info:
            await career_service.get_career_recommendations(sample_recommendation_request)
        
        assert "validation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_network_timeout_scenario(self, career_service, mock_dependencies, sample_recommendation_request):
        """Test handling of network timeout scenarios."""
        # Mock network timeout
        mock_dependencies['career_recommender'].generate_career_recommendations.side_effect = asyncio.TimeoutError("Request timeout")
        
        mock_test_result = {
            "raisec_scores": {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70, "C": 65},
            "raisec_code": "IEC",
            "user_profile": {}
        }
        mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
        
        with patch.object(career_service, '_load_career_database', return_value={}):
            with pytest.raises(TruScholarError) as exc_info:
                await career_service.get_career_recommendations(sample_recommendation_request)
        
        assert "timeout" in str(exc_info.value).lower()

    def test_extreme_raisec_scores(self, career_service):
        """Test handling of extreme RAISEC scores."""
        # All zeros
        zero_scores = {dim: 0 for dim in ["R", "A", "I", "S", "E", "C"]}
        result = career_service._validate_raisec_scores(zero_scores)
        assert result is True  # Valid but unusual
        
        # All maximum
        max_scores = {dim: 100 for dim in ["R", "A", "I", "S", "E", "C"]}
        result = career_service._validate_raisec_scores(max_scores)
        assert result is True  # Valid but unusual

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, career_service, mock_dependencies, sample_recommendation_request):
        """Test handling of concurrent recommendation requests."""
        mock_test_result = {
            "raisec_scores": {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70, "C": 65},
            "raisec_code": "IEC",
            "user_profile": {}
        }
        mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
        mock_dependencies['career_recommender'].generate_career_recommendations.return_value = {
            "recommendations": [],
            "analysis_summary": "Test",
            "confidence_level": 0.8
        }
        
        with patch.object(career_service, '_load_career_database', return_value={}):
            # Execute multiple concurrent requests
            tasks = [
                career_service.get_career_recommendations(sample_recommendation_request)
                for _ in range(3)
            ]
            
            results = await asyncio.gather(*tasks)
        
        # All requests should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, CareerRecommendationResponse)


class TestCareerServicePerformance:
    """Test performance-related aspects of career service."""

    @pytest.mark.asyncio
    async def test_recommendation_generation_time(self, career_service, mock_dependencies, 
                                                sample_recommendation_request, sample_career_database):
        """Test that recommendation generation completes within reasonable time."""
        import time
        
        mock_test_result = {
            "raisec_scores": {"R": 45, "A": 30, "I": 85, "S": 25, "E": 70, "C": 65},
            "raisec_code": "IEC",
            "user_profile": {}
        }
        mock_dependencies['test_service'].get_test_results.return_value = mock_test_result
        mock_dependencies['career_recommender'].generate_career_recommendations.return_value = {
            "recommendations": [{"career_id": "test", "match_score": 85}],
            "analysis_summary": "Test",
            "confidence_level": 0.8
        }
        
        with patch.object(career_service, '_load_career_database', return_value=sample_career_database):
            start_time = time.time()
            result = await career_service.get_career_recommendations(sample_recommendation_request)
            end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert isinstance(result, CareerRecommendationResponse)

    def test_large_career_database_handling(self, career_service):
        """Test handling of large career databases."""
        # Create large career database
        large_database = {}
        for i in range(1000):
            large_database[f"career_{i}"] = {
                "id": f"career_{i}",
                "title": f"Career {i}",
                "category": "test_category",
                "raisec_profile": {"R": 50, "A": 50, "I": 50, "S": 50, "E": 50, "C": 50}
            }
        
        # Test filtering performance
        import time
        start_time = time.time()
        
        filtered_careers = career_service._filter_careers_by_criteria(
            large_database, 
            {"categories": ["test_category"]}
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert len(filtered_careers) == 1000
        assert execution_time < 1.0  # Should complete within 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])