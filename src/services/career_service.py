"""Career recommendation service for TruScholar.

This service handles career recommendations, job market analysis, and career path planning
based on RAISEC assessment results and user preferences.
"""

import json
import asyncio
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from bson import ObjectId

from src.core.config import get_settings
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.cache.cache_manager import CacheManager
from src.cache.cache_keys import CacheKeys
from src.services.scoring_service import ScoringService
from src.models.test import TestScores
from src.utils.constants import RaisecDimension
from src.utils.exceptions import TruScholarError, ValidationError, ResourceNotFoundError
from src.utils.logger import get_logger
from src.schemas.career_schemas import (
    CareerRecommendationRequest,
    CareerRecommendationResponse,
    CareerDetailResponse,
    CareerSearchRequest,
    CareerSearchResponse,
    CareerPathRequest,
    CareerPathResponse,
    CareerTrendsResponse,
    CareerMatch,
    CareerAnalysisContext,
    CareerMatchingResult,
    SkillRequirement,
    SalaryData,
    JobMarketData,
    CareerFieldCategory,
    EducationLevel,
    ExperienceLevel,
    JobOutlook,
    WorkEnvironment
)

settings = get_settings()
logger = get_logger(__name__)


class CareerService:
    """Service for career recommendations and analysis."""
    
    def __init__(self, db=None, cache=None, scoring_service=None):
        """Initialize career service.
        
        Args:
            db: Database instance (defaults to MongoDB)
            cache: Cache instance (defaults to RedisClient)
            scoring_service: Scoring service instance
        """
        self.db = db or MongoDB
        self.cache = cache or RedisClient
        self.cache_manager = CacheManager(self.cache)
        self.scoring_service = scoring_service or ScoringService()
        
        # Load career data
        self.career_database = self._load_career_database()
        self.noc_mapping = self._load_noc_mapping()
        self.onet_mapping = self._load_onet_mapping()
        
        # Recommendation configuration
        self.recommendation_config = self._load_recommendation_config()
        
        # RAISEC dimension weights for different career aspects
        self.dimension_weights = {
            "primary_tasks": 0.4,
            "work_environment": 0.25,
            "skills_required": 0.2,
            "career_values": 0.15
        }
    
    async def get_career_recommendations(
        self,
        request: CareerRecommendationRequest
    ) -> CareerRecommendationResponse:
        """Get personalized career recommendations based on RAISEC scores.
        
        Args:
            request: Career recommendation request
            
        Returns:
            CareerRecommendationResponse: Personalized career recommendations
            
        Raises:
            ResourceNotFoundError: If test not found or not scored
            ValidationError: If request data is invalid
        """
        logger.info(f"Generating career recommendations for test: {request.test_id}")
        
        try:
            # Get test scores
            test_scores = await self._get_test_scores(request.test_id)
            if not test_scores:
                raise ResourceNotFoundError(f"Scored test not found: {request.test_id}")
            
            # Check cache first
            cache_key = CacheKeys.career_recommendations(
                request.test_id,
                str(hash(str(request.model_dump())))
            )
            
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Returning cached career recommendations")
                return CareerRecommendationResponse(**cached_result)
            
            # Create analysis context
            context = self._create_analysis_context(test_scores, request)
            
            # Generate recommendations
            career_matches = await self._generate_career_matches(context)
            
            # Rank and filter recommendations
            top_matches = self._rank_and_filter_matches(
                career_matches, 
                context,
                request.max_recommendations
            )
            
            # Enhance with market data
            if request.include_market_data:
                top_matches = await self._enhance_with_market_data(
                    top_matches,
                    context.location_preference
                )
            
            # Generate insights and themes
            insights = self._generate_career_insights(top_matches, context)
            themes = self._extract_career_themes(top_matches)
            next_steps = self._generate_next_steps(top_matches, context)
            
            # Create response
            response = CareerRecommendationResponse(
                test_id=request.test_id,
                user_raisec_code=context.user_raisec_code,
                recommendations=top_matches,
                total_matches=len(career_matches),
                recommendation_confidence=self._calculate_recommendation_confidence(
                    top_matches, context
                ),
                filters_applied=self._get_applied_filters(request),
                key_insights=insights,
                career_themes=themes,
                recommended_next_steps=next_steps,
                generated_at=datetime.utcnow().isoformat(),
                recommendation_version="v1.0"
            )
            
            # Cache the result
            await self.cache_manager.set(
                cache_key, 
                response.model_dump(),
                ttl=settings.cache_settings.career_recommendations_ttl
            )
            
            logger.info(f"Generated {len(top_matches)} career recommendations")
            return response
            
        except Exception as e:
            logger.error(f"Error generating career recommendations: {str(e)}")
            if isinstance(e, (ResourceNotFoundError, ValidationError)):
                raise
            raise TruScholarError(f"Career recommendation generation failed: {str(e)}")
    
    async def get_career_details(
        self,
        career_id: str,
        test_id: Optional[str] = None,
        include_similar: bool = True,
        include_path: bool = True
    ) -> CareerDetailResponse:
        """Get detailed information about a specific career.
        
        Args:
            career_id: Career identifier
            test_id: Optional test ID for personalized insights
            include_similar: Include similar careers
            include_path: Include career path information
            
        Returns:
            CareerDetailResponse: Detailed career information
        """
        logger.info(f"Getting career details for: {career_id}")
        
        # Get career data
        career_data = self.career_database.get(career_id)
        if not career_data:
            raise ResourceNotFoundError(f"Career not found: {career_id}")
        
        # Convert to CareerMatch
        career_match = self._create_career_match(career_data)
        
        # Get personalized insights if test provided
        personalized_insights = None
        if test_id:
            test_scores = await self._get_test_scores(test_id)
            if test_scores:
                context = self._create_analysis_context_from_scores(test_scores)
                personalized_insights = self._generate_personalized_insights(
                    career_data, context
                )
        
        # Get similar careers
        similar_careers = []
        if include_similar:
            similar_careers = await self._find_similar_careers(career_id, limit=10)
        
        # Get career path information
        career_path = None
        if include_path:
            career_path = self._generate_career_path_info(career_data)
        
        # Get educational pathways
        education_pathways = self._get_education_pathways(career_data)
        
        # Get geographic opportunities
        geographic_opportunities = await self._get_geographic_opportunities(career_id)
        
        return CareerDetailResponse(
            career=career_match,
            personalized_insights=personalized_insights,
            career_path=career_path,
            similar_careers=similar_careers,
            education_pathways=education_pathways,
            geographic_opportunities=geographic_opportunities
        )
    
    async def search_careers(
        self,
        request: CareerSearchRequest
    ) -> CareerSearchResponse:
        """Search for careers based on query and filters.
        
        Args:
            request: Career search request
            
        Returns:
            CareerSearchResponse: Search results
        """
        start_time = datetime.utcnow()
        logger.info(f"Searching careers with query: '{request.query}'")
        
        # Perform search
        matching_careers = self._search_career_database(request)
        
        # Convert to CareerMatch objects
        career_matches = [
            self._create_career_match(career_data)
            for career_data in matching_careers[:request.limit]
        ]
        
        # Calculate search time
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Generate suggestions
        suggested_queries = self._generate_search_suggestions(request.query)
        related_categories = self._get_related_categories(matching_careers)
        
        return CareerSearchResponse(
            query=request.query,
            results=career_matches,
            total_results=len(matching_careers),
            search_time_ms=search_time,
            filters_applied=self._get_search_filters_applied(request),
            suggested_queries=suggested_queries,
            related_categories=related_categories
        )
    
    async def analyze_career_path(
        self,
        request: CareerPathRequest
    ) -> CareerPathResponse:
        """Analyze career progression path and requirements.
        
        Args:
            request: Career path analysis request
            
        Returns:
            CareerPathResponse: Career path analysis
        """
        logger.info(f"Analyzing career path from '{request.current_career}' to '{request.target_career}'")
        
        # Get career data
        current_career_data = self._find_career_by_title(request.current_career)
        target_career_data = None
        if request.target_career:
            target_career_data = self._find_career_by_title(request.target_career)
        
        if not current_career_data:
            raise ResourceNotFoundError(f"Career not found: {request.current_career}")
        
        # Generate career path steps
        career_path = self._generate_career_progression_steps(
            current_career_data,
            target_career_data,
            request.current_education,
            request.years_experience
        )
        
        # Analyze gaps
        education_gap = self._analyze_education_gap(
            request.current_education,
            target_career_data or current_career_data
        )
        skill_gap = self._analyze_skill_gap(
            current_career_data,
            target_career_data or current_career_data
        )
        
        # Generate recommendations
        immediate_actions = self._generate_immediate_actions(
            career_path, education_gap, skill_gap
        )
        long_term_strategy = self._generate_long_term_strategy(career_path)
        
        # Find alternative paths
        alternative_paths = self._find_alternative_career_paths(
            current_career_data,
            target_career_data
        )
        
        # Get resources
        recommended_courses = self._get_recommended_courses(education_gap, skill_gap)
        professional_orgs = self._get_professional_organizations(
            target_career_data or current_career_data
        )
        networking_opportunities = self._get_networking_opportunities(
            target_career_data or current_career_data,
            request.location
        )
        
        return CareerPathResponse(
            current_career=request.current_career,
            target_career=request.target_career,
            career_path=career_path,
            estimated_timeline=self._calculate_path_timeline(career_path),
            difficulty_level=self._assess_path_difficulty(career_path, education_gap, skill_gap),
            education_gap=education_gap,
            skill_gap=skill_gap,
            experience_requirements=self._get_experience_requirements(target_career_data),
            immediate_actions=immediate_actions,
            long_term_strategy=long_term_strategy,
            alternative_paths=alternative_paths,
            recommended_courses=recommended_courses,
            professional_organizations=professional_orgs,
            networking_opportunities=networking_opportunities
        )
    
    async def get_career_trends(
        self,
        location: Optional[str] = None,
        timeframe: str = "5_years"
    ) -> CareerTrendsResponse:
        """Get career trends and market insights.
        
        Args:
            location: Geographic location for trends
            timeframe: Analysis timeframe
            
        Returns:
            CareerTrendsResponse: Career trends and insights
        """
        logger.info(f"Getting career trends for location: {location}")
        
        # Cache key for trends
        cache_key = CacheKeys.career_trends(location, timeframe)
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result:
            return CareerTrendsResponse(**cached_result)
        
        # Analyze trending careers
        trending_careers = self._analyze_trending_careers(location)
        emerging_fields = self._identify_emerging_fields()
        declining_careers = self._identify_declining_careers()
        
        # Market insights
        industry_growth = self._analyze_industry_growth()
        skill_demand = self._analyze_skill_demand()
        hot_job_markets = self._identify_hot_job_markets()
        
        # Future predictions
        automation_risk = self._assess_automation_risk()
        future_skills = self._predict_future_skills()
        
        response = CareerTrendsResponse(
            trending_careers=trending_careers,
            emerging_fields=emerging_fields,
            declining_careers=declining_careers,
            industry_growth=industry_growth,
            skill_demand=skill_demand,
            hot_job_markets=hot_job_markets,
            automation_risk=automation_risk,
            future_skills=future_skills,
            analysis_date=datetime.utcnow().isoformat(),
            data_sources=["BLS", "O*NET", "NOC", "Industry Reports"]
        )
        
        # Cache for 24 hours
        await self.cache_manager.set(cache_key, response.model_dump(), ttl=86400)
        
        return response
    
    # Private helper methods
    
    async def _get_test_scores(self, test_id: str) -> Optional[TestScores]:
        """Get test scores from scoring service or database."""
        try:
            # Try to get from scoring service first
            test_scores = await self.scoring_service.get_cached_scores(ObjectId(test_id))
            if test_scores:
                return test_scores
            
            # Fall back to database
            test_data = await self.db.find_one(
                "tests",
                {"_id": ObjectId(test_id), "status": "completed"}
            )
            if test_data and test_data.get("scores"):
                return TestScores(**test_data["scores"])
            
            return None
        except Exception as e:
            logger.error(f"Error getting test scores: {e}")
            return None
    
    def _create_analysis_context(
        self,
        test_scores: TestScores,
        request: CareerRecommendationRequest
    ) -> CareerAnalysisContext:
        """Create analysis context from test scores and request."""
        return CareerAnalysisContext(
            user_raisec_scores=test_scores.raisec_profile.dimension_scores,
            user_raisec_code=test_scores.raisec_code,
            test_confidence=test_scores.consistency_score,
            education_level=request.education_level,
            experience_level=request.experience_level,
            location_preference=request.location_preference,
            salary_expectations=request.salary_range,
            work_environment_preferences=request.work_environment or [],
            recommendation_count=request.max_recommendations,
            include_emerging=request.include_emerging_careers
        )
    
    def _create_analysis_context_from_scores(self, test_scores: TestScores) -> CareerAnalysisContext:
        """Create basic analysis context from test scores only."""
        return CareerAnalysisContext(
            user_raisec_scores=test_scores.raisec_profile.dimension_scores,
            user_raisec_code=test_scores.raisec_code,
            test_confidence=test_scores.consistency_score
        )
    
    async def _generate_career_matches(
        self,
        context: CareerAnalysisContext
    ) -> List[CareerMatchingResult]:
        """Generate career matches based on analysis context."""
        career_matches = []
        
        for career_id, career_data in self.career_database.items():
            # Skip emerging careers if not requested
            if not context.include_emerging and career_data.get("is_emerging", False):
                continue
            
            # Calculate match score
            match_result = self._calculate_career_match(career_data, context)
            
            # Apply confidence threshold
            if match_result.match_score >= context.confidence_threshold:
                career_matches.append(match_result)
        
        return career_matches
    
    def _calculate_career_match(
        self,
        career_data: Dict[str, Any],
        context: CareerAnalysisContext
    ) -> CareerMatchingResult:
        """Calculate how well a career matches user's RAISEC profile."""
        
        # Get career's RAISEC profile
        career_raisec = career_data.get("raisec_profile", {})
        
        # Calculate RAISEC similarity
        raisec_similarity = self._calculate_raisec_similarity(
            context.user_raisec_scores,
            career_raisec
        )
        
        # Calculate preference alignment
        preference_alignment = self._calculate_preference_alignment(
            career_data,
            context
        )
        
        # Calculate requirement feasibility
        requirement_feasibility = self._calculate_requirement_feasibility(
            career_data,
            context
        )
        
        # Calculate overall match score
        match_score = (
            raisec_similarity * 0.5 +
            preference_alignment * 0.3 +
            requirement_feasibility * 0.2
        )
        
        # Identify matching dimensions
        matching_dimensions = self._identify_matching_dimensions(
            context.user_raisec_scores,
            career_raisec
        )
        
        # Generate match reasoning
        match_reasoning = self._generate_match_reasoning(
            career_data,
            context,
            raisec_similarity,
            matching_dimensions
        )
        
        # Identify potential concerns
        potential_concerns = self._identify_potential_concerns(
            career_data,
            context,
            requirement_feasibility
        )
        
        return CareerMatchingResult(
            career_id=career_data["id"],
            match_score=match_score,
            raisec_similarity=raisec_similarity,
            preference_alignment=preference_alignment,
            requirement_feasibility=requirement_feasibility,
            matching_dimensions=matching_dimensions,
            match_reasoning=match_reasoning,
            potential_concerns=potential_concerns
        )
    
    def _calculate_raisec_similarity(
        self,
        user_scores: Dict[RaisecDimension, float],
        career_raisec: Dict[str, float]
    ) -> float:
        """Calculate similarity between user and career RAISEC profiles."""
        if not career_raisec:
            return 0.5  # Neutral score if no career RAISEC data
        
        total_similarity = 0
        dimension_count = 0
        
        for dimension, user_score in user_scores.items():
            career_score = career_raisec.get(dimension.value, 0)
            
            # Normalize scores to 0-1 range
            user_norm = user_score / 100.0
            career_norm = career_score / 100.0
            
            # Calculate similarity (1 - absolute difference)
            similarity = 1 - abs(user_norm - career_norm)
            
            # Weight by user's score strength
            weight = user_norm
            total_similarity += similarity * weight
            dimension_count += weight
        
        return total_similarity / dimension_count if dimension_count > 0 else 0.5
    
    def _calculate_preference_alignment(
        self,
        career_data: Dict[str, Any],
        context: CareerAnalysisContext
    ) -> float:
        """Calculate how well career aligns with user preferences."""
        alignment_score = 1.0
        
        # Education level preference
        if context.education_level:
            career_education = career_data.get("education_requirements", [])
            if context.education_level.value not in career_education:
                alignment_score *= 0.8
        
        # Work environment preference
        if context.work_environment_preferences:
            career_environments = career_data.get("work_environments", [])
            matching_envs = set(context.work_environment_preferences) & set(career_environments)
            if matching_envs:
                alignment_score *= 1.0
            else:
                alignment_score *= 0.7
        
        # Salary expectations
        if context.salary_expectations:
            career_salary = career_data.get("salary_data", {})
            if self._salary_meets_expectations(career_salary, context.salary_expectations):
                alignment_score *= 1.0
            else:
                alignment_score *= 0.9
        
        return min(alignment_score, 1.0)
    
    def _calculate_requirement_feasibility(
        self,
        career_data: Dict[str, Any],
        context: CareerAnalysisContext
    ) -> float:
        """Calculate feasibility of meeting career requirements."""
        feasibility = 1.0
        
        # Education feasibility
        if context.education_level:
            required_education = career_data.get("min_education_level")
            if required_education:
                education_levels = [
                    "high_school", "certificate", "associate", 
                    "bachelor", "master", "doctoral", "professional"
                ]
                user_level = education_levels.index(context.education_level.value)
                required_level = education_levels.index(required_education)
                
                if user_level >= required_level:
                    feasibility *= 1.0
                elif user_level >= required_level - 1:
                    feasibility *= 0.9
                else:
                    feasibility *= 0.7
        
        # Experience feasibility
        if context.experience_level:
            required_experience = career_data.get("experience_level")
            if required_experience and required_experience == "senior_level" and context.experience_level.value == "entry_level":
                feasibility *= 0.6
        
        return feasibility
    
    def _identify_matching_dimensions(
        self,
        user_scores: Dict[RaisecDimension, float],
        career_raisec: Dict[str, float]
    ) -> List[RaisecDimension]:
        """Identify strongly matching RAISEC dimensions."""
        matching_dimensions = []
        
        for dimension, user_score in user_scores.items():
            career_score = career_raisec.get(dimension.value, 0)
            
            # Consider it a match if both are above threshold and similar
            if user_score >= 60 and career_score >= 60:
                similarity = 1 - abs(user_score - career_score) / 100
                if similarity >= 0.7:
                    matching_dimensions.append(dimension)
        
        return matching_dimensions
    
    def _generate_match_reasoning(
        self,
        career_data: Dict[str, Any],
        context: CareerAnalysisContext,
        raisec_similarity: float,
        matching_dimensions: List[RaisecDimension]
    ) -> List[str]:
        """Generate human-readable reasons for the career match."""
        reasons = []
        
        # RAISEC match reasons
        if raisec_similarity >= 0.8:
            reasons.append("Excellent alignment with your personality profile")
        elif raisec_similarity >= 0.6:
            reasons.append("Good compatibility with your interests and strengths")
        
        # Specific dimension matches
        if RaisecDimension.REALISTIC in matching_dimensions:
            reasons.append("Matches your preference for hands-on, practical work")
        if RaisecDimension.INVESTIGATIVE in matching_dimensions:
            reasons.append("Aligns with your analytical and research-oriented nature")
        if RaisecDimension.ARTISTIC in matching_dimensions:
            reasons.append("Suits your creative and expressive interests")
        if RaisecDimension.SOCIAL in matching_dimensions:
            reasons.append("Perfect for your people-oriented and helping nature")
        if RaisecDimension.ENTERPRISING in matching_dimensions:
            reasons.append("Matches your leadership and business-minded approach")
        if RaisecDimension.CONVENTIONAL in matching_dimensions:
            reasons.append("Aligns with your organized and detail-oriented style")
        
        # Work environment matches
        if context.work_environment_preferences:
            career_environments = career_data.get("work_environments", [])
            matching_envs = set(context.work_environment_preferences) & set(career_environments)
            if matching_envs:
                env_names = [env.replace("_", " ").title() for env in matching_envs]
                reasons.append(f"Offers your preferred work environment: {', '.join(env_names)}")
        
        return reasons[:5]  # Limit to top 5 reasons
    
    def _identify_potential_concerns(
        self,
        career_data: Dict[str, Any],
        context: CareerAnalysisContext,
        requirement_feasibility: float
    ) -> List[str]:
        """Identify potential concerns or challenges."""
        concerns = []
        
        # Education gap
        if context.education_level:
            required_education = career_data.get("min_education_level")
            if required_education:
                education_levels = [
                    "high_school", "certificate", "associate", 
                    "bachelor", "master", "doctoral", "professional"
                ]
                try:
                    user_level = education_levels.index(context.education_level.value)
                    required_level = education_levels.index(required_education)
                    
                    if user_level < required_level:
                        concerns.append(f"Requires additional education: {required_education.replace('_', ' ').title()}")
                except ValueError:
                    pass
        
        # Salary expectations
        if context.salary_expectations:
            career_salary = career_data.get("salary_data", {})
            if not self._salary_meets_expectations(career_salary, context.salary_expectations):
                concerns.append("Salary range may not meet your expectations")
        
        # Job market competitiveness
        job_market = career_data.get("job_market", {})
        if job_market.get("competitiveness", 3) >= 4:
            concerns.append("Highly competitive job market")
        
        # Automation risk
        automation_risk = career_data.get("automation_risk", 0)
        if automation_risk >= 0.7:
            concerns.append("Higher risk of automation in the future")
        
        return concerns
    
    def _rank_and_filter_matches(
        self,
        career_matches: List[CareerMatchingResult],
        context: CareerAnalysisContext,
        limit: int
    ) -> List[CareerMatch]:
        """Rank and filter career matches to return top recommendations."""
        
        # Sort by match score
        career_matches.sort(key=lambda x: x.match_score, reverse=True)
        
        # Convert to CareerMatch objects
        career_match_objects = []
        for match_result in career_matches[:limit]:
            career_data = self.career_database[match_result.career_id]
            career_match = self._create_career_match(career_data, match_result)
            career_match_objects.append(career_match)
        
        return career_match_objects
    
    def _create_career_match(
        self,
        career_data: Dict[str, Any],
        match_result: Optional[CareerMatchingResult] = None
    ) -> CareerMatch:
        """Create CareerMatch object from career data."""
        
        # Extract RAISEC data
        raisec_profile = career_data.get("raisec_profile", {})
        primary_dimensions = [
            RaisecDimension(dim) for dim, score in raisec_profile.items()
            if score >= 60
        ][:3]  # Top 3 dimensions
        
        # Create dimensional fit scores
        dimensional_fit = {}
        if match_result:
            # Use calculated fit from match result
            for dim in RaisecDimension:
                dimensional_fit[dim.value] = raisec_profile.get(dim.value, 0)
        else:
            dimensional_fit = raisec_profile
        
        # Create skill requirements
        skills = career_data.get("key_skills", [])
        skill_requirements = [
            SkillRequirement(
                skill_name=skill["name"],
                importance_level=skill.get("importance", 3),
                skill_category=skill.get("category", "technical"),
                description=skill.get("description")
            )
            for skill in skills
        ]
        
        # Create salary data
        salary_info = career_data.get("salary_data", {})
        salary_data = None
        if salary_info:
            salary_data = SalaryData(**salary_info)
        
        # Create job market data
        market_info = career_data.get("job_market", {})
        job_market = None
        if market_info:
            job_market = JobMarketData(**market_info)
        
        return CareerMatch(
            career_id=career_data["id"],
            career_title=career_data["title"],
            category=CareerFieldCategory(career_data.get("category", "business")),
            raisec_match_score=match_result.match_score * 100 if match_result else 75.0,
            primary_raisec_dimensions=primary_dimensions,
            dimensional_fit=dimensional_fit,
            description=career_data.get("description", ""),
            typical_tasks=career_data.get("typical_tasks", []),
            work_environment=[
                WorkEnvironment(env) for env in career_data.get("work_environments", [])
            ],
            education_requirements=[
                EducationLevel(req) for req in career_data.get("education_requirements", [])
            ],
            key_skills=skill_requirements,
            experience_needed=ExperienceLevel(
                career_data.get("experience_level", "entry_level")
            ),
            salary_data=salary_data,
            job_market=job_market,
            match_reasons=match_result.match_reasoning if match_result else [],
            potential_challenges=match_result.potential_concerns if match_result else [],
            similar_careers=career_data.get("similar_careers", [])
        )
    
    async def _enhance_with_market_data(
        self,
        career_matches: List[CareerMatch],
        location: Optional[str] = None
    ) -> List[CareerMatch]:
        """Enhance career matches with current market data."""
        # This would integrate with external APIs (BLS, O*NET, etc.)
        # For now, return as-is
        return career_matches
    
    def _generate_career_insights(
        self,
        career_matches: List[CareerMatch],
        context: CareerAnalysisContext
    ) -> List[str]:
        """Generate key insights from career recommendations."""
        insights = []
        
        # Analyze dominant categories
        categories = Counter([career.category for career in career_matches[:10]])
        if categories:
            top_category = categories.most_common(1)[0]
            insights.append(
                f"You show strong alignment with {top_category[0].value.replace('_', ' ').title()} careers"
            )
        
        # Analyze education patterns
        education_reqs = [
            req for career in career_matches[:10] 
            for req in career.education_requirements
        ]
        education_counter = Counter(education_reqs)
        if education_counter:
            common_education = education_counter.most_common(1)[0][0]
            insights.append(
                f"Most recommended careers require {common_education.value.replace('_', ' ').title()} education"
            )
        
        # Analyze RAISEC patterns
        user_top_dimensions = sorted(
            context.user_raisec_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        if user_top_dimensions:
            top_dim = user_top_dimensions[0][0]
            insights.append(
                f"Your strongest dimension ({top_dim.value}) appears frequently in your top matches"
            )
        
        return insights
    
    def _extract_career_themes(self, career_matches: List[CareerMatch]) -> List[str]:
        """Extract common themes from career recommendations."""
        themes = set()
        
        # Analyze job titles for common themes
        for career in career_matches[:15]:
            title_words = career.career_title.lower().split()
            common_themes = {
                "analyst", "manager", "engineer", "designer", "researcher",
                "consultant", "specialist", "coordinator", "developer", "technician"
            }
            themes.update(word for word in title_words if word in common_themes)
        
        # Add category-based themes
        categories = set(career.category.value for career in career_matches[:10])
        themes.update(cat.replace("_", " ").title() for cat in categories)
        
        return list(themes)[:8]  # Return top 8 themes
    
    def _generate_next_steps(
        self,
        career_matches: List[CareerMatch],
        context: CareerAnalysisContext
    ) -> List[str]:
        """Generate recommended next steps based on career matches."""
        next_steps = []
        
        # Research recommendations
        next_steps.append("Research your top 3-5 career matches in detail")
        next_steps.append("Conduct informational interviews with professionals in these fields")
        
        # Education/skills
        common_skills = Counter()
        for career in career_matches[:10]:
            for skill in career.key_skills:
                common_skills[skill.skill_name] += 1
        
        if common_skills:
            top_skill = common_skills.most_common(1)[0][0]
            next_steps.append(f"Develop skills in {top_skill}")
        
        # Experience building
        if context.experience_level in [None, ExperienceLevel.ENTRY_LEVEL]:
            next_steps.append("Look for internships, volunteer opportunities, or entry-level positions")
        
        # Networking
        next_steps.append("Join professional associations related to your target careers")
        
        return next_steps[:6]  # Limit to 6 recommendations
    
    def _calculate_recommendation_confidence(
        self,
        career_matches: List[CareerMatch],
        context: CareerAnalysisContext
    ) -> float:
        """Calculate confidence in the overall recommendations."""
        if not career_matches:
            return 0.0
        
        # Factor in test confidence
        test_confidence = context.test_confidence / 100.0
        
        # Factor in number of high-quality matches
        high_quality_matches = sum(1 for career in career_matches if career.raisec_match_score >= 80)
        match_quality = min(high_quality_matches / 5, 1.0)  # Normalize to 0-1
        
        # Factor in score distribution (prefer diverse but strong matches)
        scores = [career.raisec_match_score for career in career_matches[:10]]
        if len(scores) > 1:
            score_std = statistics.stdev(scores)
            score_diversity = min(score_std / 20, 1.0)  # Normalize
        else:
            score_diversity = 0.5
        
        # Combined confidence
        confidence = (test_confidence * 0.4 + match_quality * 0.4 + score_diversity * 0.2) * 100
        
        return min(confidence, 95.0)  # Cap at 95%
    
    def _get_applied_filters(self, request: CareerRecommendationRequest) -> Dict[str, Any]:
        """Get dictionary of applied filters."""
        filters = {}
        
        if request.education_level:
            filters["education_level"] = request.education_level.value
        if request.experience_level:
            filters["experience_level"] = request.experience_level.value
        if request.location_preference:
            filters["location"] = request.location_preference
        if request.salary_range:
            filters["salary_range"] = request.salary_range
        if request.work_environment:
            filters["work_environment"] = [env.value for env in request.work_environment]
        
        filters["max_recommendations"] = request.max_recommendations
        filters["include_emerging"] = request.include_emerging_careers
        
        return filters
    
    def _salary_meets_expectations(
        self,
        career_salary: Dict[str, Any],
        expectations: Dict[str, int]
    ) -> bool:
        """Check if career salary meets user expectations."""
        if not career_salary or not expectations:
            return True  # Assume neutral if no data
        
        career_median = career_salary.get("median", 0)
        expected_min = expectations.get("min", 0)
        
        return career_median >= expected_min
    
    # Data loading methods
    
    def _load_career_database(self) -> Dict[str, Dict[str, Any]]:
        """Load career database from JSON file."""
        try:
            data_path = Path(settings.project_root) / "data" / "careers" / "career_database.json"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    careers_list = json.load(f)
                
                # Convert list to dictionary keyed by career ID
                careers_dict = {career["id"]: career for career in careers_list}
                logger.info(f"Loaded {len(careers_dict)} careers from database")
                return careers_dict
            else:
                logger.warning("Career database file not found, using sample data")
                return self._create_sample_career_database()
        except Exception as e:
            logger.error(f"Error loading career database: {e}")
            return self._create_sample_career_database()
    
    def _load_noc_mapping(self) -> Dict[str, Any]:
        """Load NOC (National Occupational Classification) mapping."""
        try:
            data_path = Path(settings.project_root) / "data" / "careers" / "noc_mapping.json"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning("NOC mapping file not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading NOC mapping: {e}")
            return {}
    
    def _load_onet_mapping(self) -> Dict[str, Any]:
        """Load O*NET mapping."""
        try:
            data_path = Path(settings.project_root) / "data" / "careers" / "onet_mapping.json"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning("O*NET mapping file not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading O*NET mapping: {e}")
            return {}
    
    def _load_recommendation_config(self) -> Dict[str, Any]:
        """Load recommendation algorithm configuration."""
        return {
            "raisec_weight": 0.5,
            "preference_weight": 0.3,
            "feasibility_weight": 0.2,
            "confidence_threshold": 0.6,
            "max_emerging_percentage": 0.3
        }
    
    def _create_sample_career_database(self) -> Dict[str, Dict[str, Any]]:
        """Create sample career database for development/testing."""
        sample_careers = {
            "software_engineer": {
                "id": "software_engineer",
                "title": "Software Engineer",
                "category": "information_technology",
                "description": "Design and develop software applications and systems",
                "raisec_profile": {"R": 30, "I": 85, "A": 40, "S": 25, "E": 35, "C": 70},
                "typical_tasks": [
                    "Write and test code",
                    "Debug software issues",
                    "Design system architecture",
                    "Collaborate with team members"
                ],
                "work_environments": ["office", "remote", "hybrid"],
                "education_requirements": ["bachelor", "master"],
                "experience_level": "entry_level",
                "key_skills": [
                    {"name": "Programming", "importance": 5, "category": "technical"},
                    {"name": "Problem Solving", "importance": 5, "category": "soft"},
                    {"name": "Software Design", "importance": 4, "category": "technical"}
                ],
                "salary_data": {
                    "currency": "USD",
                    "entry_level": 75000,
                    "median": 95000,
                    "experienced": 130000,
                    "location": "US National Average"
                },
                "job_market": {
                    "employment_count": 1500000,
                    "projected_growth_rate": 8.5,
                    "outlook": "rapidly_growing",
                    "competitiveness": 3
                }
            }
        }
        
        return sample_careers
    
    # Search and discovery methods
    
    def _search_career_database(self, request: CareerSearchRequest) -> List[Dict[str, Any]]:
        """Search career database based on request criteria."""
        matching_careers = []
        query_lower = request.query.lower()
        
        for career_data in self.career_database.values():
            # Text search in title and description
            title_match = query_lower in career_data.get("title", "").lower()
            desc_match = query_lower in career_data.get("description", "").lower()
            
            if not (title_match or desc_match):
                continue
            
            # Apply filters
            if request.category and career_data.get("category") != request.category.value:
                continue
            
            if request.education_level:
                career_education = career_data.get("education_requirements", [])
                if not any(edu in career_education for edu in [e.value for e in request.education_level]):
                    continue
            
            if request.salary_range:
                salary_data = career_data.get("salary_data", {})
                if not self._salary_meets_expectations(salary_data, request.salary_range):
                    continue
            
            matching_careers.append(career_data)
        
        return matching_careers
    
    def _generate_search_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions based on query."""
        suggestions = []
        
        # Common career-related terms
        career_terms = [
            "analyst", "manager", "engineer", "designer", "consultant",
            "developer", "specialist", "coordinator", "technician", "researcher"
        ]
        
        for term in career_terms:
            if term.startswith(query.lower()[:3]) and term != query.lower():
                suggestions.append(term.title())
        
        return suggestions[:5]
    
    def _get_related_categories(self, careers: List[Dict[str, Any]]) -> List[CareerFieldCategory]:
        """Get related career categories from search results."""
        categories = set()
        for career in careers[:20]:  # Analyze top 20 results
            category = career.get("category")
            if category:
                try:
                    categories.add(CareerFieldCategory(category))
                except ValueError:
                    continue
        
        return list(categories)[:5]
    
    def _get_search_filters_applied(self, request: CareerSearchRequest) -> Dict[str, Any]:
        """Get applied search filters."""
        filters = {"query": request.query}
        
        if request.category:
            filters["category"] = request.category.value
        if request.raisec_dimensions:
            filters["raisec_dimensions"] = [dim.value for dim in request.raisec_dimensions]
        if request.education_level:
            filters["education_level"] = [edu.value for edu in request.education_level]
        if request.salary_range:
            filters["salary_range"] = request.salary_range
        
        return filters
    
    # Career path analysis methods (placeholder implementations)
    
    def _find_career_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Find career by title."""
        title_lower = title.lower()
        for career_data in self.career_database.values():
            if career_data.get("title", "").lower() == title_lower:
                return career_data
        return None
    
    def _generate_career_progression_steps(self, current_career, target_career, current_education, years_experience):
        """Generate career progression steps."""
        # Placeholder implementation
        return []
    
    def _analyze_education_gap(self, current_education, target_career):
        """Analyze education gap."""
        # Placeholder implementation
        return []
    
    def _analyze_skill_gap(self, current_career, target_career):
        """Analyze skill gap."""
        # Placeholder implementation
        return []
    
    # Trends analysis methods (placeholder implementations)
    
    def _analyze_trending_careers(self, location):
        """Analyze trending careers."""
        # Placeholder - would integrate with labor market APIs
        return []
    
    def _identify_emerging_fields(self):
        """Identify emerging career fields."""
        return ["AI/Machine Learning", "Cybersecurity", "Data Science", "Renewable Energy"]
    
    def _identify_declining_careers(self):
        """Identify declining careers."""
        return ["Traditional Manufacturing", "Print Media", "Routine Clerical Work"]
    
    def _analyze_industry_growth(self):
        """Analyze industry growth rates."""
        return {
            "technology": 8.2,
            "healthcare": 6.1,
            "renewable_energy": 12.3,
            "manufacturing": 2.1
        }
    
    def _analyze_skill_demand(self):
        """Analyze skill demand trends."""
        return {
            "programming": 0.85,
            "data_analysis": 0.78,
            "communication": 0.72,
            "problem_solving": 0.91
        }
    
    def _identify_hot_job_markets(self):
        """Identify hot job markets by location."""
        return [
            {"location": "Austin, TX", "growth_rate": 15.2, "key_industries": ["Tech", "Healthcare"]},
            {"location": "Seattle, WA", "growth_rate": 12.8, "key_industries": ["Tech", "Aerospace"]},
            {"location": "Denver, CO", "growth_rate": 11.5, "key_industries": ["Energy", "Tech"]}
        ]
    
    def _assess_automation_risk(self):
        """Assess automation risk by career."""
        return {
            "data_entry_clerk": 0.95,
            "cashier": 0.87,
            "software_engineer": 0.15,
            "nurse": 0.08,
            "creative_director": 0.05
        }
    
    def _predict_future_skills(self):
        """Predict future important skills."""
        return [
            "Artificial Intelligence",
            "Emotional Intelligence",
            "Complex Problem Solving",
            "Digital Literacy",
            "Adaptability",
            "Cross-cultural Communication"
        ]
    
    # Additional helper methods would be implemented here...
    
    async def _find_similar_careers(self, career_id: str, limit: int = 10) -> List[CareerMatch]:
        """Find careers similar to the given career."""
        # Placeholder implementation
        return []
    
    def _generate_personalized_insights(self, career_data, context):
        """Generate personalized insights for a career."""
        # Placeholder implementation
        return {}
    
    def _generate_career_path_info(self, career_data):
        """Generate career path information."""
        # Placeholder implementation
        return {}
    
    def _get_education_pathways(self, career_data):
        """Get educational pathways for a career."""
        # Placeholder implementation
        return []
    
    async def _get_geographic_opportunities(self, career_id):
        """Get geographic opportunities for a career."""
        # Placeholder implementation
        return {}