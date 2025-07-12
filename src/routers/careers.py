"""Career recommendation API endpoints.

This module provides REST API endpoints for career recommendations, career exploration,
job market analysis, and career path planning based on RAISEC assessment results.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from bson import ObjectId
from datetime import datetime

from src.core.dependencies import get_current_user, get_career_service
from src.core.security import require_permissions
from src.services.career_service import CareerService
from src.utils.exceptions import TruScholarError, ValidationError, ResourceNotFoundError
from src.utils.logger import get_logger
from src.schemas.career_schemas import (
    # Request schemas
    CareerRecommendationRequest,
    CareerExplorationRequest,
    CareerSearchRequest,
    CareerPathRequest,
    
    # Response schemas
    CareerRecommendationResponse,
    CareerDetailResponse,
    CareerSearchResponse,
    CareerPathResponse,
    CareerTrendsResponse,
    CareerMatch,
    
    # Enums
    CareerFieldCategory,
    EducationLevel,
    ExperienceLevel,
    WorkEnvironment
)
from src.schemas.base import BaseResponse, ErrorResponse

router = APIRouter(prefix="/careers", tags=["careers"])
logger = get_logger(__name__)


# Career Recommendations

@router.post(
    "/recommendations",
    response_model=CareerRecommendationResponse,
    summary="Get personalized career recommendations",
    description="Generate personalized career recommendations based on RAISEC assessment results and user preferences."
)
async def get_career_recommendations(
    request: CareerRecommendationRequest,
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get personalized career recommendations based on RAISEC scores."""
    try:
        logger.info(f"Career recommendations requested for test: {request.test_id}")
        
        # Validate test ownership
        # This would typically check if the user owns the test
        # For now, we'll trust the request
        
        recommendations = await career_service.get_career_recommendations(request)
        
        logger.info(f"Generated {len(recommendations.recommendations)} career recommendations")
        return recommendations
        
    except ResourceNotFoundError as e:
        logger.warning(f"Test not found for recommendations: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        logger.warning(f"Invalid recommendation request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating career recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate career recommendations")


@router.get(
    "/recommendations/{test_id}",
    response_model=CareerRecommendationResponse,
    summary="Get cached career recommendations",
    description="Retrieve previously generated career recommendations for a test."
)
async def get_cached_recommendations(
    test_id: str = Path(..., description="Test ID"),
    include_analytics: bool = Query(True, description="Include detailed analytics"),
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get cached career recommendations for a specific test."""
    try:
        # Create a basic request to get cached recommendations
        request = CareerRecommendationRequest(
            test_id=test_id,
            include_market_data=include_analytics
        )
        
        recommendations = await career_service.get_career_recommendations(request)
        return recommendations
        
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving cached recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")


# Career Exploration

@router.get(
    "/{career_id}",
    response_model=CareerDetailResponse,
    summary="Get detailed career information",
    description="Get comprehensive information about a specific career including requirements, outlook, and personalized insights."
)
async def get_career_details(
    career_id: str = Path(..., description="Career identifier"),
    test_id: Optional[str] = Query(None, description="Test ID for personalized insights"),
    include_similar: bool = Query(True, description="Include similar careers"),
    include_path: bool = Query(True, description="Include career path information"),
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get detailed information about a specific career."""
    try:
        career_details = await career_service.get_career_details(
            career_id=career_id,
            test_id=test_id,
            include_similar=include_similar,
            include_path=include_path
        )
        
        return career_details
        
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Career not found: {career_id}")
    except Exception as e:
        logger.error(f"Error getting career details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve career details")


@router.post(
    "/explore",
    response_model=CareerDetailResponse,
    summary="Explore career with personalized insights",
    description="Explore a specific career with personalized insights based on user's assessment results."
)
async def explore_career(
    request: CareerExplorationRequest,
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Explore a career with personalized insights."""
    try:
        career_details = await career_service.get_career_details(
            career_id=request.career_id,
            test_id=request.test_id,
            include_similar=request.include_similar_careers,
            include_path=request.include_career_path
        )
        
        return career_details
        
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exploring career: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to explore career")


# Career Search

@router.post(
    "/search",
    response_model=CareerSearchResponse,
    summary="Search careers",
    description="Search for careers based on keywords, categories, and filters."
)
async def search_careers(
    request: CareerSearchRequest,
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Search for careers based on criteria."""
    try:
        search_results = await career_service.search_careers(request)
        return search_results
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching careers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search careers")


@router.get(
    "/search/suggestions",
    response_model=List[str],
    summary="Get search suggestions",
    description="Get search suggestions for career queries."
)
async def get_search_suggestions(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Maximum suggestions"),
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get search suggestions for career queries."""
    try:
        # This would typically use a search suggestion service
        # For now, return basic suggestions
        suggestions = [
            f"{query} engineer",
            f"{query} manager", 
            f"{query} specialist",
            f"{query} analyst",
            f"{query} coordinator"
        ]
        
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get suggestions")


# Career Path Analysis

@router.post(
    "/path-analysis",
    response_model=CareerPathResponse,
    summary="Analyze career path",
    description="Analyze career progression paths and requirements from current to target career."
)
async def analyze_career_path(
    request: CareerPathRequest,
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Analyze career progression path and requirements."""
    try:
        path_analysis = await career_service.analyze_career_path(request)
        return path_analysis
        
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing career path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze career path")


# Career Trends and Market Data

@router.get(
    "/trends",
    response_model=CareerTrendsResponse,
    summary="Get career trends",
    description="Get current career trends, emerging fields, and job market insights."
)
async def get_career_trends(
    location: Optional[str] = Query(None, description="Geographic location for trends"),
    timeframe: str = Query("5_years", description="Analysis timeframe"),
    include_emerging: bool = Query(True, description="Include emerging careers"),
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get career trends and market insights."""
    try:
        trends = await career_service.get_career_trends(
            location=location,
            timeframe=timeframe
        )
        
        return trends
        
    except Exception as e:
        logger.error(f"Error getting career trends: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get career trends")


# Career Categories and Filters

@router.get(
    "/categories",
    response_model=List[dict],
    summary="Get career categories",
    description="Get available career categories and their descriptions."
)
async def get_career_categories(
    include_counts: bool = Query(False, description="Include career counts per category"),
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get available career categories."""
    try:
        categories = []
        for category in CareerFieldCategory:
            category_info = {
                "id": category.value,
                "name": category.value.replace("_", " ").title(),
                "description": f"Careers in {category.value.replace('_', ' ')}"
            }
            
            if include_counts:
                # This would query the actual database for counts
                category_info["career_count"] = 5  # Placeholder
            
            categories.append(category_info)
        
        return categories
        
    except Exception as e:
        logger.error(f"Error getting career categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get career categories")


@router.get(
    "/filters/education-levels",
    response_model=List[dict],
    summary="Get education level options",
    description="Get available education level filter options."
)
async def get_education_levels(
    current_user = Depends(get_current_user)
):
    """Get available education level options."""
    try:
        education_levels = []
        for level in EducationLevel:
            education_levels.append({
                "id": level.value,
                "name": level.value.replace("_", " ").title(),
                "description": f"{level.value.replace('_', ' ').title()} level education"
            })
        
        return education_levels
        
    except Exception as e:
        logger.error(f"Error getting education levels: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get education levels")


@router.get(
    "/filters/work-environments",
    response_model=List[dict],
    summary="Get work environment options",
    description="Get available work environment filter options."
)
async def get_work_environments(
    current_user = Depends(get_current_user)
):
    """Get available work environment options."""
    try:
        environments = []
        for env in WorkEnvironment:
            environments.append({
                "id": env.value,
                "name": env.value.replace("_", " ").title(),
                "description": f"Work in {env.value.replace('_', ' ')} environment"
            })
        
        return environments
        
    except Exception as e:
        logger.error(f"Error getting work environments: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get work environments")


# Career Comparison

@router.post(
    "/compare",
    response_model=dict,
    summary="Compare careers",
    description="Compare multiple careers across various dimensions."
)
async def compare_careers(
    career_ids: List[str],
    test_id: Optional[str] = None,
    comparison_aspects: Optional[List[str]] = None,
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Compare multiple careers."""
    try:
        if len(career_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 careers required for comparison")
        
        if len(career_ids) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 careers can be compared")
        
        # This would use the LangChain career recommender for comparison
        # For now, return a basic comparison structure
        comparison_result = {
            "careers_compared": career_ids,
            "comparison_aspects": comparison_aspects or [
                "salary", "education_requirements", "job_outlook", 
                "work_environment", "raisec_fit"
            ],
            "comparison_matrix": {},
            "recommendation": "Based on your profile, consider career options that align with your strengths.",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return comparison_result
        
    except Exception as e:
        logger.error(f"Error comparing careers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to compare careers")


# Career Bookmarks/Favorites

@router.post(
    "/favorites/{career_id}",
    response_model=BaseResponse,
    summary="Add career to favorites",
    description="Add a career to user's favorites list."
)
async def add_career_favorite(
    career_id: str = Path(..., description="Career identifier"),
    current_user = Depends(get_current_user)
):
    """Add a career to user's favorites."""
    try:
        # This would typically save to user's profile in database
        logger.info(f"Career {career_id} added to favorites for user {current_user.get('id')}")
        
        return BaseResponse(
            success=True,
            message="Career added to favorites",
            data={"career_id": career_id}
        )
        
    except Exception as e:
        logger.error(f"Error adding career favorite: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add career to favorites")


@router.delete(
    "/favorites/{career_id}",
    response_model=BaseResponse,
    summary="Remove career from favorites",
    description="Remove a career from user's favorites list."
)
async def remove_career_favorite(
    career_id: str = Path(..., description="Career identifier"),
    current_user = Depends(get_current_user)
):
    """Remove a career from user's favorites."""
    try:
        # This would typically remove from user's profile in database
        logger.info(f"Career {career_id} removed from favorites for user {current_user.get('id')}")
        
        return BaseResponse(
            success=True,
            message="Career removed from favorites",
            data={"career_id": career_id}
        )
        
    except Exception as e:
        logger.error(f"Error removing career favorite: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to remove career from favorites")


@router.get(
    "/favorites",
    response_model=List[CareerMatch],
    summary="Get favorite careers",
    description="Get user's favorite careers list."
)
async def get_favorite_careers(
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get user's favorite careers."""
    try:
        # This would typically query user's favorites from database
        # For now, return empty list
        favorites = []
        
        return favorites
        
    except Exception as e:
        logger.error(f"Error getting favorite careers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get favorite careers")


# Analytics and Insights

@router.get(
    "/analytics/popular",
    response_model=List[dict],
    summary="Get popular careers",
    description="Get currently popular careers based on user interactions."
)
async def get_popular_careers(
    limit: int = Query(10, ge=1, le=50, description="Number of popular careers"),
    timeframe: str = Query("30_days", description="Timeframe for popularity"),
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Get popular careers based on user interactions."""
    try:
        # This would query analytics data
        # For now, return placeholder data
        popular_careers = [
            {
                "career_id": "software_engineer",
                "title": "Software Engineer",
                "popularity_score": 95,
                "view_count": 1250,
                "favorite_count": 340
            },
            {
                "career_id": "data_scientist", 
                "title": "Data Scientist",
                "popularity_score": 88,
                "view_count": 980,
                "favorite_count": 275
            }
        ]
        
        return popular_careers[:limit]
        
    except Exception as e:
        logger.error(f"Error getting popular careers: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get popular careers")


# Background Tasks

@router.post(
    "/recommendations/refresh",
    response_model=BaseResponse,
    summary="Refresh career recommendations",
    description="Trigger a background refresh of career recommendations for a test."
)
async def refresh_recommendations(
    test_id: str,
    background_tasks: BackgroundTasks,
    career_service: CareerService = Depends(get_career_service),
    current_user = Depends(get_current_user)
):
    """Refresh career recommendations in the background."""
    try:
        # Add background task to refresh recommendations
        background_tasks.add_task(
            refresh_recommendations_task,
            test_id,
            career_service
        )
        
        return BaseResponse(
            success=True,
            message="Career recommendations refresh started",
            data={"test_id": test_id}
        )
        
    except Exception as e:
        logger.error(f"Error starting recommendations refresh: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start recommendations refresh")


# Background task functions

async def refresh_recommendations_task(test_id: str, career_service: CareerService):
    """Background task to refresh career recommendations."""
    try:
        logger.info(f"Starting background refresh for test: {test_id}")
        
        # Create refresh request
        request = CareerRecommendationRequest(
            test_id=test_id,
            max_recommendations=25,
            include_market_data=True
        )
        
        # Generate fresh recommendations
        await career_service.get_career_recommendations(request)
        
        logger.info(f"Successfully refreshed recommendations for test: {test_id}")
        
    except Exception as e:
        logger.error(f"Error in background recommendations refresh: {str(e)}")


# Exception handlers should be added at the app level, not router level