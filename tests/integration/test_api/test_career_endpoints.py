"""Integration tests for career recommendation API endpoints.

This module provides comprehensive integration testing for all career-related
API endpoints including recommendations, search, details, and analytics.
Tests cover authentication, validation, business logic, and error handling.
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app
from src.core.dependencies import get_current_user, get_career_service
from src.services.career_service import CareerService
from src.schemas.career_schemas import (
    CareerRecommendationRequest,
    CareerRecommendationResponse,
    CareerMatch,
    CareerDetailResponse,
    CareerSearchRequest,
    CareerFieldCategory,
    EducationLevel,
    ExperienceLevel,
    WorkEnvironment
)
from src.utils.constants import RaisecDimension


class TestCareerEndpointsIntegration:
    """Integration tests for career API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    @pytest.fixture
    def mock_current_user(self):
        """Mock current user for authentication."""
        return {
            "id": "user_123",
            "email": "test@example.com",
            "name": "Test User",
            "role": "user"
        }

    @pytest.fixture
    def mock_career_service(self):
        """Mock career service."""
        service = Mock(spec=CareerService)
        
        # Mock career recommendations
        service.get_career_recommendations = AsyncMock(return_value=CareerRecommendationResponse(
            recommendations=[
                CareerMatch(
                    career_id="software_engineer",
                    career_title="Software Engineer",
                    category=CareerFieldCategory.INFORMATION_TECHNOLOGY,
                    raisec_match_score=92.5,
                    primary_raisec_dimensions=[RaisecDimension.INVESTIGATIVE, RaisecDimension.ENTERPRISING],
                    secondary_raisec_dimensions=[RaisecDimension.CONVENTIONAL],
                    match_explanation="Strong alignment with investigative and enterprising characteristics",
                    confidence_level=0.89,
                    education_requirements=[EducationLevel.BACHELOR],
                    experience_level=ExperienceLevel.ENTRY_LEVEL,
                    work_environments=[WorkEnvironment.OFFICE, WorkEnvironment.REMOTE],
                    salary_range={"min": 500000, "max": 1200000, "currency": "INR"},
                    growth_potential="High",
                    key_skills=["Programming", "Problem Solving", "System Design"],
                    development_recommendations=[
                        "Develop expertise in multiple programming languages",
                        "Gain experience with system architecture",
                        "Build portfolio of diverse projects"
                    ]
                )
            ],
            analysis_summary="Strong investigative-enterprising profile with excellent technology career fit",
            user_raisec_code="IEC",
            user_raisec_scores={"I": 85, "E": 70, "C": 65, "R": 45, "A": 35, "S": 25},
            confidence_level=0.87,
            recommendation_metadata={
                "algorithm_version": "1.0",
                "generated_at": datetime.utcnow().isoformat(),
                "market_data_included": True
            }
        ))
        
        # Mock career details
        service.get_career_details = AsyncMock(return_value=CareerDetailResponse(
            career=CareerMatch(
                career_id="software_engineer",
                career_title="Software Engineer",
                category=CareerFieldCategory.INFORMATION_TECHNOLOGY,
                raisec_match_score=92.5,
                primary_raisec_dimensions=[RaisecDimension.INVESTIGATIVE, RaisecDimension.ENTERPRISING],
                secondary_raisec_dimensions=[RaisecDimension.CONVENTIONAL],
                match_explanation="Detailed career analysis",
                confidence_level=0.89
            ),
            detailed_description="Comprehensive description of software engineering career",
            typical_tasks=[
                "Design and develop software applications",
                "Debug and troubleshoot code issues",
                "Collaborate with cross-functional teams"
            ],
            career_path=[
                {"level": "Junior Developer", "years": "0-2", "responsibilities": ["Code development", "Bug fixes"]},
                {"level": "Senior Developer", "years": "3-5", "responsibilities": ["Architecture design", "Mentoring"]},
                {"level": "Tech Lead", "years": "6-8", "responsibilities": ["Team leadership", "Technical strategy"]}
            ],
            similar_careers=["Data Scientist", "Product Manager", "DevOps Engineer"],
            market_data={
                "job_openings": 15000,
                "growth_rate": 12.5,
                "salary_trends": "Increasing",
                "demand_level": "High"
            },
            personalized_insights={
                "strengths_alignment": "Excellent match for analytical thinking",
                "development_areas": "Consider developing leadership skills",
                "success_factors": ["Technical expertise", "Problem-solving", "Continuous learning"]
            }
        ))
        
        # Mock career search
        service.search_careers = AsyncMock()
        
        # Mock career path analysis
        service.analyze_career_path = AsyncMock()
        
        # Mock career trends
        service.get_career_trends = AsyncMock()
        
        return service

    @pytest.fixture
    def override_dependencies(self, mock_current_user, mock_career_service):
        """Override app dependencies with mocks."""
        app.dependency_overrides[get_current_user] = lambda: mock_current_user
        app.dependency_overrides[get_career_service] = lambda: mock_career_service
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def sample_test_data(self):
        """Sample test data for requests."""
        return {
            "test_id": "test_123",
            "raisec_scores": {
                "R": 45.5,
                "A": 30.2,
                "I": 85.7,
                "S": 25.8,
                "E": 70.3,
                "C": 65.1
            },
            "user_profile": {
                "age": 22,
                "education": "bachelor",
                "experience": "entry_level",
                "location": "Bangalore"
            }
        }


class TestCareerRecommendationEndpoints:
    """Test career recommendation endpoints."""

    def test_get_career_recommendations_success(self, client, override_dependencies, sample_test_data):
        """Test successful career recommendations retrieval."""
        
        request_data = {
            "test_id": sample_test_data["test_id"],
            "max_recommendations": 10,
            "include_market_data": True,
            "recommendation_type": "comprehensive"
        }
        
        response = client.post("/careers/recommendations", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        assert "analysis_summary" in data
        assert "user_raisec_code" in data
        assert "confidence_level" in data
        
        # Verify recommendation structure
        recommendations = data["recommendations"]
        assert len(recommendations) > 0
        
        first_recommendation = recommendations[0]
        assert "career_id" in first_recommendation
        assert "career_title" in first_recommendation
        assert "raisec_match_score" in first_recommendation
        assert "confidence_level" in first_recommendation
        assert first_recommendation["raisec_match_score"] > 0

    def test_get_career_recommendations_validation_error(self, client, override_dependencies):
        """Test career recommendations with validation errors."""
        
        # Missing required field
        invalid_request = {
            "max_recommendations": 10,
            "include_market_data": True
            # Missing test_id
        }
        
        response = client.post("/careers/recommendations", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data

    def test_get_career_recommendations_unauthorized(self, client):
        """Test career recommendations without authentication."""
        
        # Remove dependency override to test authentication
        request_data = {
            "test_id": "test_123",
            "max_recommendations": 10
        }
        
        response = client.post("/careers/recommendations", json=request_data)
        
        # Should require authentication
        assert response.status_code in [401, 403]

    def test_get_cached_recommendations_success(self, client, override_dependencies):
        """Test retrieving cached career recommendations."""
        
        test_id = "test_123"
        
        response = client.get(f"/careers/recommendations/{test_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        assert "analysis_summary" in data

    def test_get_cached_recommendations_not_found(self, client, override_dependencies, mock_career_service):
        """Test retrieving non-existent cached recommendations."""
        
        from src.utils.exceptions import ResourceNotFoundError
        mock_career_service.get_career_recommendations.side_effect = ResourceNotFoundError("Test not found")
        
        test_id = "non_existent_test"
        
        response = client.get(f"/careers/recommendations/{test_id}")
        
        assert response.status_code == 404

    def test_get_recommendations_with_filters(self, client, override_dependencies):
        """Test career recommendations with filters."""
        
        request_data = {
            "test_id": "test_123",
            "max_recommendations": 5,
            "filters": {
                "education_levels": ["bachelor"],
                "experience_levels": ["entry_level"],
                "work_environments": ["remote", "hybrid"],
                "salary_range": {"min": 500000, "max": 1500000}
            }
        }
        
        response = client.post("/careers/recommendations", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        
        # Verify filters are applied (mock service should handle this)
        recommendations = data["recommendations"]
        for rec in recommendations:
            assert rec["raisec_match_score"] > 0


class TestCareerDetailsEndpoints:
    """Test career details endpoints."""

    def test_get_career_details_success(self, client, override_dependencies):
        """Test successful career details retrieval."""
        
        career_id = "software_engineer"
        
        response = client.get(f"/careers/{career_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "career" in data
        assert "detailed_description" in data
        assert "typical_tasks" in data
        assert "career_path" in data
        assert "market_data" in data
        
        # Verify career structure
        career = data["career"]
        assert career["career_id"] == career_id
        assert "career_title" in career
        assert "raisec_match_score" in career

    def test_get_career_details_with_personalization(self, client, override_dependencies):
        """Test career details with personalization."""
        
        career_id = "software_engineer"
        test_id = "test_123"
        
        response = client.get(f"/careers/{career_id}?test_id={test_id}&include_similar=true&include_path=true")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "personalized_insights" in data
        assert "similar_careers" in data
        assert data["personalized_insights"] is not None

    def test_get_career_details_not_found(self, client, override_dependencies, mock_career_service):
        """Test career details for non-existent career."""
        
        from src.utils.exceptions import ResourceNotFoundError
        mock_career_service.get_career_details.side_effect = ResourceNotFoundError("Career not found")
        
        career_id = "non_existent_career"
        
        response = client.get(f"/careers/{career_id}")
        
        assert response.status_code == 404

    def test_explore_career_success(self, client, override_dependencies):
        """Test career exploration endpoint."""
        
        request_data = {
            "career_id": "software_engineer",
            "test_id": "test_123",
            "include_similar_careers": True,
            "include_career_path": True
        }
        
        response = client.post("/careers/explore", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "career" in data
        assert "personalized_insights" in data


class TestCareerSearchEndpoints:
    """Test career search endpoints."""

    def test_search_careers_success(self, client, override_dependencies, mock_career_service):
        """Test successful career search."""
        
        # Mock search response
        from src.schemas.career_schemas import CareerSearchResponse
        mock_career_service.search_careers.return_value = CareerSearchResponse(
            results=[
                CareerMatch(
                    career_id="software_engineer",
                    career_title="Software Engineer",
                    category=CareerFieldCategory.INFORMATION_TECHNOLOGY,
                    raisec_match_score=85.0,
                    primary_raisec_dimensions=[RaisecDimension.INVESTIGATIVE],
                    secondary_raisec_dimensions=[],
                    match_explanation="Search result",
                    confidence_level=0.8
                )
            ],
            total_count=1,
            page=1,
            page_size=10,
            filters_applied={
                "query": "software engineer",
                "categories": ["information_technology"]
            }
        )
        
        request_data = {
            "query": "software engineer",
            "categories": ["information_technology"],
            "education_levels": ["bachelor"],
            "page": 1,
            "page_size": 10
        }
        
        response = client.post("/careers/search", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        
        # Verify search results
        results = data["results"]
        assert len(results) > 0
        assert results[0]["career_title"] == "Software Engineer"

    def test_search_careers_with_filters(self, client, override_dependencies, mock_career_service):
        """Test career search with multiple filters."""
        
        mock_career_service.search_careers.return_value = CareerSearchResponse(
            results=[],
            total_count=0,
            page=1,
            page_size=10,
            filters_applied={}
        )
        
        request_data = {
            "categories": ["information_technology", "healthcare"],
            "education_levels": ["bachelor", "master"],
            "experience_levels": ["entry_level"],
            "work_environments": ["remote"],
            "salary_range": {"min": 600000, "max": 1500000},
            "page": 1,
            "page_size": 5
        }
        
        response = client.post("/careers/search", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_search_suggestions_success(self, client, override_dependencies):
        """Test search suggestions endpoint."""
        
        response = client.get("/careers/search/suggestions?query=software&limit=5")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5

    def test_search_suggestions_validation(self, client, override_dependencies):
        """Test search suggestions with validation."""
        
        # Query too short
        response = client.get("/careers/search/suggestions?query=a")
        
        assert response.status_code == 422


class TestCareerPathEndpoints:
    """Test career path analysis endpoints."""

    def test_analyze_career_path_success(self, client, override_dependencies, mock_career_service):
        """Test successful career path analysis."""
        
        from src.schemas.career_schemas import CareerPathResponse
        mock_career_service.analyze_career_path.return_value = CareerPathResponse(
            current_career="software_engineer",
            target_career="product_manager",
            feasibility_score=78.5,
            transition_steps=[
                "Develop product management skills",
                "Gain customer interaction experience",
                "Learn business analysis"
            ],
            skill_gaps=["Product strategy", "Market analysis"],
            timeline_assessment="Realistic for 2-year timeline",
            development_plan={
                "phase_1": "Technical leadership development",
                "phase_2": "Business acumen building",
                "phase_3": "Product management transition"
            },
            alternative_paths=["Technical Lead", "Engineering Manager"],
            success_probability=0.75
        )
        
        request_data = {
            "current_career": "software_engineer",
            "target_career": "product_manager",
            "user_profile": {
                "education": "bachelor",
                "experience_years": 3,
                "skills": ["programming", "problem_solving"]
            },
            "timeline": "2_years"
        }
        
        response = client.post("/careers/path-analysis", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "feasibility_score" in data
        assert "transition_steps" in data
        assert "skill_gaps" in data
        assert "timeline_assessment" in data
        
        assert data["feasibility_score"] == 78.5
        assert len(data["transition_steps"]) > 0

    def test_analyze_career_path_invalid_careers(self, client, override_dependencies, mock_career_service):
        """Test career path analysis with invalid careers."""
        
        from src.utils.exceptions import ResourceNotFoundError
        mock_career_service.analyze_career_path.side_effect = ResourceNotFoundError("Career not found")
        
        request_data = {
            "current_career": "non_existent_career",
            "target_career": "another_non_existent_career",
            "user_profile": {},
            "timeline": "1_year"
        }
        
        response = client.post("/careers/path-analysis", json=request_data)
        
        assert response.status_code == 404


class TestCareerTrendsEndpoints:
    """Test career trends and market data endpoints."""

    def test_get_career_trends_success(self, client, override_dependencies, mock_career_service):
        """Test successful career trends retrieval."""
        
        from src.schemas.career_schemas import CareerTrendsResponse
        mock_career_service.get_career_trends.return_value = CareerTrendsResponse(
            trending_careers=[
                {
                    "career_id": "ai_engineer",
                    "title": "AI Engineer",
                    "growth_rate": 35.2,
                    "demand_level": "very_high",
                    "trend_direction": "increasing"
                }
            ],
            emerging_fields=["Quantum Computing", "Blockchain Development"],
            declining_fields=["Traditional System Administration"],
            market_insights="Technology sector showing strongest growth in AI and data science",
            regional_data={
                "bangalore": {"growth_rate": 25.5, "opportunities": 1500},
                "mumbai": {"growth_rate": 18.3, "opportunities": 1200}
            },
            timeframe="5_years",
            data_sources=["Government statistics", "Industry reports"]
        )
        
        response = client.get("/careers/trends?location=bangalore&timeframe=5_years")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "trending_careers" in data
        assert "emerging_fields" in data
        assert "market_insights" in data
        
        trending_careers = data["trending_careers"]
        assert len(trending_careers) > 0
        assert trending_careers[0]["title"] == "AI Engineer"

    def test_get_career_trends_no_location(self, client, override_dependencies, mock_career_service):
        """Test career trends without location filter."""
        
        from src.schemas.career_schemas import CareerTrendsResponse
        mock_career_service.get_career_trends.return_value = CareerTrendsResponse(
            trending_careers=[],
            emerging_fields=[],
            declining_fields=[],
            market_insights="National market trends",
            timeframe="3_years",
            data_sources=[]
        )
        
        response = client.get("/careers/trends?timeframe=3_years")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["market_insights"] == "National market trends"


class TestCareerMetadataEndpoints:
    """Test career metadata and utility endpoints."""

    def test_get_career_categories_success(self, client, override_dependencies):
        """Test career categories endpoint."""
        
        response = client.get("/careers/categories")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Verify category structure
        first_category = data[0]
        assert "id" in first_category
        assert "name" in first_category
        assert "description" in first_category

    def test_get_career_categories_with_counts(self, client, override_dependencies):
        """Test career categories with counts."""
        
        response = client.get("/careers/categories?include_counts=true")
        
        assert response.status_code == 200
        
        data = response.json()
        first_category = data[0]
        assert "career_count" in first_category

    def test_get_education_levels(self, client, override_dependencies):
        """Test education levels endpoint."""
        
        response = client.get("/careers/filters/education-levels")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        first_level = data[0]
        assert "id" in first_level
        assert "name" in first_level
        assert "description" in first_level

    def test_get_work_environments(self, client, override_dependencies):
        """Test work environments endpoint."""
        
        response = client.get("/careers/filters/work-environments")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestCareerComparisonEndpoints:
    """Test career comparison endpoints."""

    def test_compare_careers_success(self, client, override_dependencies):
        """Test successful career comparison."""
        
        career_ids = ["software_engineer", "data_scientist", "product_manager"]
        
        request_data = {
            "career_ids": career_ids,
            "test_id": "test_123",
            "comparison_aspects": ["salary", "growth", "raisec_fit"]
        }
        
        response = client.post("/careers/compare", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "careers_compared" in data
        assert "comparison_aspects" in data
        assert "recommendation" in data
        
        assert data["careers_compared"] == career_ids

    def test_compare_careers_insufficient_careers(self, client, override_dependencies):
        """Test career comparison with insufficient careers."""
        
        request_data = {
            "career_ids": ["software_engineer"],  # Only one career
            "comparison_aspects": ["salary", "growth"]
        }
        
        response = client.post("/careers/compare", json=request_data)
        
        assert response.status_code == 400

    def test_compare_careers_too_many_careers(self, client, override_dependencies):
        """Test career comparison with too many careers."""
        
        career_ids = [f"career_{i}" for i in range(6)]  # More than 5 careers
        
        request_data = {
            "career_ids": career_ids,
            "comparison_aspects": ["salary"]
        }
        
        response = client.post("/careers/compare", json=request_data)
        
        assert response.status_code == 400


class TestCareerFavoritesEndpoints:
    """Test career favorites/bookmarks endpoints."""

    def test_add_career_favorite_success(self, client, override_dependencies):
        """Test adding career to favorites."""
        
        career_id = "software_engineer"
        
        response = client.post(f"/careers/favorites/{career_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Career added to favorites"
        assert data["data"]["career_id"] == career_id

    def test_remove_career_favorite_success(self, client, override_dependencies):
        """Test removing career from favorites."""
        
        career_id = "software_engineer"
        
        response = client.delete(f"/careers/favorites/{career_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Career removed from favorites"

    def test_get_favorite_careers_success(self, client, override_dependencies):
        """Test retrieving favorite careers."""
        
        response = client.get("/careers/favorites")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)


class TestCareerAnalyticsEndpoints:
    """Test career analytics endpoints."""

    def test_get_popular_careers_success(self, client, override_dependencies):
        """Test popular careers endpoint."""
        
        response = client.get("/careers/analytics/popular?limit=5&timeframe=30_days")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
        
        if data:
            first_career = data[0]
            assert "career_id" in first_career
            assert "title" in first_career
            assert "popularity_score" in first_career

    def test_get_popular_careers_pagination(self, client, override_dependencies):
        """Test popular careers with pagination."""
        
        response = client.get("/careers/analytics/popular?limit=3")
        
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) <= 3


class TestCareerBackgroundTasks:
    """Test career background task endpoints."""

    def test_refresh_recommendations_success(self, client, override_dependencies):
        """Test background refresh of recommendations."""
        
        request_data = {
            "test_id": "test_123"
        }
        
        response = client.post("/careers/recommendations/refresh", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Career recommendations refresh started"
        assert data["data"]["test_id"] == "test_123"


class TestCareerEndpointErrorHandling:
    """Test error handling across career endpoints."""

    def test_validation_error_handling(self, client, override_dependencies):
        """Test validation error handling."""
        
        # Send request with invalid data
        invalid_request = {
            "test_id": "",  # Empty test ID
            "max_recommendations": -1,  # Negative number
        }
        
        response = client.post("/careers/recommendations", json=invalid_request)
        
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data

    def test_resource_not_found_handling(self, client, override_dependencies, mock_career_service):
        """Test resource not found error handling."""
        
        from src.utils.exceptions import ResourceNotFoundError
        mock_career_service.get_career_details.side_effect = ResourceNotFoundError("Career not found")
        
        response = client.get("/careers/non_existent_career")
        
        assert response.status_code == 404

    def test_internal_server_error_handling(self, client, override_dependencies, mock_career_service):
        """Test internal server error handling."""
        
        mock_career_service.get_career_recommendations.side_effect = Exception("Internal error")
        
        request_data = {
            "test_id": "test_123",
            "max_recommendations": 10
        }
        
        response = client.post("/careers/recommendations", json=request_data)
        
        assert response.status_code == 500


class TestCareerEndpointPerformance:
    """Test performance aspects of career endpoints."""

    @pytest.mark.asyncio
    async def test_recommendation_endpoint_performance(self, async_client, override_dependencies):
        """Test recommendation endpoint performance."""
        
        import time
        
        request_data = {
            "test_id": "test_123",
            "max_recommendations": 20
        }
        
        start_time = time.time()
        response = await async_client.post("/careers/recommendations", json=request_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert response.status_code == 200
        assert execution_time < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self, async_client, override_dependencies):
        """Test handling of concurrent requests."""
        
        request_data = {
            "test_id": "test_123",
            "max_recommendations": 10
        }
        
        # Send multiple concurrent requests
        tasks = [
            async_client.post("/careers/recommendations", json=request_data)
            for _ in range(5)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_large_response_handling(self, client, override_dependencies, mock_career_service):
        """Test handling of large responses."""
        
        # Mock large response
        large_recommendations = []
        for i in range(50):
            large_recommendations.append(CareerMatch(
                career_id=f"career_{i}",
                career_title=f"Career {i}",
                category=CareerFieldCategory.INFORMATION_TECHNOLOGY,
                raisec_match_score=80.0 + i,
                primary_raisec_dimensions=[RaisecDimension.INVESTIGATIVE],
                secondary_raisec_dimensions=[],
                match_explanation=f"Career {i} explanation",
                confidence_level=0.8
            ))
        
        mock_career_service.get_career_recommendations.return_value = CareerRecommendationResponse(
            recommendations=large_recommendations,
            analysis_summary="Large response test",
            user_raisec_code="IEC",
            user_raisec_scores={"I": 85, "E": 70, "C": 65},
            confidence_level=0.8
        )
        
        request_data = {
            "test_id": "test_123",
            "max_recommendations": 50
        }
        
        response = client.post("/careers/recommendations", json=request_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["recommendations"]) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])