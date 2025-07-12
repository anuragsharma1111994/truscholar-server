"""Career recommendation schemas for TruScholar API.

This module defines Pydantic schemas for career-related API requests and responses,
including career recommendations, job market data, and career path information.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from src.schemas.base import BaseResponse
from src.utils.constants import RaisecDimension


# Enums for career data

class CareerFieldCategory(str, Enum):
    """Major career field categories."""
    AGRICULTURE = "agriculture"
    ARTS_CULTURE = "arts_culture"
    BUSINESS = "business"
    EDUCATION = "education"
    ENGINEERING = "engineering"
    HEALTHCARE = "healthcare"
    INFORMATION_TECHNOLOGY = "information_technology"
    LAW_PUBLIC_SAFETY = "law_public_safety"
    MANUFACTURING = "manufacturing"
    SCIENCE = "science"
    SOCIAL_SERVICES = "social_services"
    TRANSPORTATION = "transportation"


class EducationLevel(str, Enum):
    """Education level requirements."""
    HIGH_SCHOOL = "high_school"
    CERTIFICATE = "certificate"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORAL = "doctoral"
    PROFESSIONAL = "professional"


class ExperienceLevel(str, Enum):
    """Experience level requirements."""
    ENTRY_LEVEL = "entry_level"
    MID_LEVEL = "mid_level"
    SENIOR_LEVEL = "senior_level"
    EXECUTIVE = "executive"


class JobOutlook(str, Enum):
    """Job market outlook."""
    DECLINING = "declining"
    STABLE = "stable"
    GROWING = "growing"
    RAPIDLY_GROWING = "rapidly_growing"


class WorkEnvironment(str, Enum):
    """Work environment types."""
    OFFICE = "office"
    REMOTE = "remote"
    HYBRID = "hybrid"
    FIELD = "field"
    LABORATORY = "laboratory"
    FACTORY = "factory"
    OUTDOORS = "outdoors"
    HEALTHCARE_FACILITY = "healthcare_facility"


# Request Schemas

class CareerRecommendationRequest(BaseModel):
    """Request schema for career recommendations."""
    
    test_id: str = Field(..., description="Completed test ID for RAISEC scores")
    
    # Optional filters
    education_level: Optional[EducationLevel] = Field(
        default=None, 
        description="Preferred education level"
    )
    experience_level: Optional[ExperienceLevel] = Field(
        default=None,
        description="Current experience level"
    )
    location_preference: Optional[str] = Field(
        default=None,
        description="Preferred work location (city, state, country)"
    )
    salary_range: Optional[Dict[str, int]] = Field(
        default=None,
        description="Salary expectations (min/max)"
    )
    work_environment: Optional[List[WorkEnvironment]] = Field(
        default=None,
        description="Preferred work environments"
    )
    
    # Recommendation parameters
    max_recommendations: int = Field(
        default=20, 
        ge=5, 
        le=50, 
        description="Maximum number of career recommendations"
    )
    include_emerging_careers: bool = Field(
        default=True,
        description="Include emerging/future careers"
    )
    include_market_data: bool = Field(
        default=True,
        description="Include job market and salary data"
    )


class CareerExplorationRequest(BaseModel):
    """Request schema for exploring specific career details."""
    
    career_id: str = Field(..., description="Career ID to explore")
    test_id: Optional[str] = Field(
        default=None,
        description="Test ID for personalized insights"
    )
    include_similar_careers: bool = Field(
        default=True,
        description="Include similar career options"
    )
    include_career_path: bool = Field(
        default=True,
        description="Include career progression paths"
    )


class CareerSearchRequest(BaseModel):
    """Request schema for searching careers."""
    
    query: str = Field(..., min_length=2, description="Search query")
    category: Optional[CareerFieldCategory] = Field(
        default=None,
        description="Filter by career category"
    )
    raisec_dimensions: Optional[List[RaisecDimension]] = Field(
        default=None,
        description="Filter by RAISEC dimensions"
    )
    education_level: Optional[List[EducationLevel]] = Field(
        default=None,
        description="Filter by education requirements"
    )
    salary_range: Optional[Dict[str, int]] = Field(
        default=None,
        description="Salary range filter"
    )
    limit: int = Field(default=20, ge=1, le=100, description="Result limit")


class CareerPathRequest(BaseModel):
    """Request schema for career path analysis."""
    
    current_career: str = Field(..., description="Current or target career")
    target_career: Optional[str] = Field(
        default=None,
        description="Desired career (for transition planning)"
    )
    current_education: EducationLevel = Field(..., description="Current education level")
    years_experience: int = Field(
        default=0, 
        ge=0, 
        description="Years of relevant experience"
    )
    location: Optional[str] = Field(
        default=None,
        description="Geographic location"
    )


# Response Schemas

class SkillRequirement(BaseModel):
    """Skill requirement for a career."""
    
    skill_name: str = Field(..., description="Name of the skill")
    importance_level: int = Field(..., ge=1, le=5, description="Importance (1-5)")
    skill_category: str = Field(..., description="Category (technical, soft, domain)")
    description: Optional[str] = Field(default=None, description="Skill description")


class SalaryData(BaseModel):
    """Salary information for a career."""
    
    currency: str = Field(default="USD", description="Currency code")
    entry_level: Optional[int] = Field(default=None, description="Entry level salary")
    median: Optional[int] = Field(default=None, description="Median salary")
    experienced: Optional[int] = Field(default=None, description="Experienced level salary")
    top_percentile: Optional[int] = Field(default=None, description="Top 10% salary")
    location: Optional[str] = Field(default=None, description="Geographic location")
    last_updated: Optional[str] = Field(default=None, description="Data update date")


class JobMarketData(BaseModel):
    """Job market information for a career."""
    
    employment_count: Optional[int] = Field(default=None, description="Current employment")
    projected_growth_rate: Optional[float] = Field(
        default=None,
        description="Projected growth rate (%)"
    )
    job_openings_annually: Optional[int] = Field(
        default=None,
        description="Annual job openings"
    )
    outlook: JobOutlook = Field(..., description="Employment outlook")
    competitiveness: int = Field(
        ..., 
        ge=1, 
        le=5, 
        description="Job market competitiveness (1-5)"
    )
    location: Optional[str] = Field(default=None, description="Geographic location")


class CareerMatch(BaseModel):
    """Career recommendation with match details."""
    
    career_id: str = Field(..., description="Unique career identifier")
    career_title: str = Field(..., description="Career title")
    category: CareerFieldCategory = Field(..., description="Career category")
    
    # RAISEC matching
    raisec_match_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="RAISEC compatibility score"
    )
    primary_raisec_dimensions: List[RaisecDimension] = Field(
        ..., 
        description="Primary RAISEC dimensions"
    )
    dimensional_fit: Dict[str, float] = Field(
        ..., 
        description="Fit score for each RAISEC dimension"
    )
    
    # Career details
    description: str = Field(..., description="Career description")
    typical_tasks: List[str] = Field(..., description="Typical job tasks")
    work_environment: List[WorkEnvironment] = Field(
        ..., 
        description="Work environments"
    )
    
    # Requirements
    education_requirements: List[EducationLevel] = Field(
        ..., 
        description="Education requirements"
    )
    key_skills: List[SkillRequirement] = Field(..., description="Required skills")
    experience_needed: ExperienceLevel = Field(..., description="Experience level")
    
    # Market data
    salary_data: Optional[SalaryData] = Field(default=None, description="Salary information")
    job_market: Optional[JobMarketData] = Field(default=None, description="Market data")
    
    # Additional insights
    match_reasons: List[str] = Field(..., description="Why this career matches")
    potential_challenges: List[str] = Field(
        default_factory=list,
        description="Potential challenges or considerations"
    )
    similar_careers: List[str] = Field(
        default_factory=list,
        description="Similar career options"
    )


class CareerRecommendationResponse(BaseModel):
    """Response schema for career recommendations."""
    
    test_id: str = Field(..., description="Source test ID")
    user_raisec_code: str = Field(..., description="User's RAISEC code")
    
    # Recommendations
    recommendations: List[CareerMatch] = Field(..., description="Career recommendations")
    total_matches: int = Field(..., description="Total careers evaluated")
    
    # Recommendation metadata
    recommendation_confidence: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Confidence in recommendations"
    )
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters used in recommendation"
    )
    
    # Insights
    key_insights: List[str] = Field(..., description="Key insights from analysis")
    career_themes: List[str] = Field(..., description="Career themes that emerge")
    recommended_next_steps: List[str] = Field(
        ..., 
        description="Recommended actions"
    )
    
    # Metadata
    generated_at: str = Field(..., description="Generation timestamp")
    recommendation_version: str = Field(default="v1.0", description="Algorithm version")


class CareerDetailResponse(BaseModel):
    """Response schema for detailed career information."""
    
    career: CareerMatch = Field(..., description="Detailed career information")
    
    # Personalized insights (if test_id provided)
    personalized_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Personalized career insights"
    )
    
    # Career progression
    career_path: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Career progression information"
    )
    
    # Related information
    similar_careers: List[CareerMatch] = Field(
        default_factory=list,
        description="Similar career options"
    )
    transition_careers: List[CareerMatch] = Field(
        default_factory=list,
        description="Careers for easy transition"
    )
    
    # Educational pathways
    education_pathways: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Educational pathways to this career"
    )
    
    # Geographic data
    geographic_opportunities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Geographic job opportunities"
    )


class CareerSearchResponse(BaseModel):
    """Response schema for career search."""
    
    query: str = Field(..., description="Search query used")
    results: List[CareerMatch] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total matching careers")
    
    # Search metadata
    search_time_ms: float = Field(..., description="Search execution time")
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Applied search filters"
    )
    
    # Suggestions
    suggested_queries: List[str] = Field(
        default_factory=list,
        description="Suggested search refinements"
    )
    related_categories: List[CareerFieldCategory] = Field(
        default_factory=list,
        description="Related career categories"
    )


class CareerPathStep(BaseModel):
    """Individual step in a career path."""
    
    step_number: int = Field(..., description="Step sequence number")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Step description")
    timeline: str = Field(..., description="Expected timeline")
    requirements: List[str] = Field(..., description="Requirements for this step")
    skills_to_develop: List[str] = Field(..., description="Skills to develop")
    resources: List[str] = Field(default_factory=list, description="Helpful resources")


class CareerPathResponse(BaseModel):
    """Response schema for career path analysis."""
    
    current_career: str = Field(..., description="Starting career")
    target_career: Optional[str] = Field(default=None, description="Target career")
    
    # Path analysis
    career_path: List[CareerPathStep] = Field(..., description="Career progression steps")
    estimated_timeline: str = Field(..., description="Total estimated timeline")
    difficulty_level: int = Field(
        ..., 
        ge=1, 
        le=5, 
        description="Path difficulty (1-5)"
    )
    
    # Requirements analysis
    education_gap: List[str] = Field(
        default_factory=list,
        description="Additional education needed"
    )
    skill_gap: List[str] = Field(
        default_factory=list,
        description="Skills to develop"
    )
    experience_requirements: List[str] = Field(
        default_factory=list,
        description="Experience requirements"
    )
    
    # Recommendations
    immediate_actions: List[str] = Field(..., description="Actions to take now")
    long_term_strategy: List[str] = Field(..., description="Long-term strategy")
    alternative_paths: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative career paths"
    )
    
    # Resources
    recommended_courses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recommended courses/programs"
    )
    professional_organizations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant professional organizations"
    )
    networking_opportunities: List[str] = Field(
        default_factory=list,
        description="Networking suggestions"
    )


class CareerTrendsResponse(BaseModel):
    """Response schema for career trends and insights."""
    
    trending_careers: List[CareerMatch] = Field(..., description="Trending careers")
    emerging_fields: List[str] = Field(..., description="Emerging career fields")
    declining_careers: List[str] = Field(..., description="Declining careers")
    
    # Market insights
    industry_growth: Dict[str, float] = Field(
        default_factory=dict,
        description="Industry growth rates"
    )
    skill_demand: Dict[str, float] = Field(
        default_factory=dict,
        description="In-demand skills"
    )
    
    # Geographic trends
    hot_job_markets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Growing job markets by location"
    )
    
    # Future predictions
    automation_risk: Dict[str, float] = Field(
        default_factory=dict,
        description="Automation risk by career"
    )
    future_skills: List[str] = Field(
        default_factory=list,
        description="Skills likely to be important"
    )
    
    # Metadata
    analysis_date: str = Field(..., description="Analysis date")
    data_sources: List[str] = Field(..., description="Data sources used")


# Processing Schemas

class CareerAnalysisContext(BaseModel):
    """Context for career analysis and recommendations."""
    
    user_raisec_scores: Dict[RaisecDimension, float] = Field(
        ..., 
        description="User's RAISEC dimension scores"
    )
    user_raisec_code: str = Field(..., description="User's RAISEC code")
    test_confidence: float = Field(..., description="Assessment confidence level")
    
    # User preferences
    education_level: Optional[EducationLevel] = None
    experience_level: Optional[ExperienceLevel] = None
    location_preference: Optional[str] = None
    salary_expectations: Optional[Dict[str, int]] = None
    work_environment_preferences: List[WorkEnvironment] = Field(default_factory=list)
    
    # Context data
    age_group: Optional[str] = None
    career_interests: Optional[str] = None
    current_career: Optional[str] = None
    
    # Analysis parameters
    recommendation_count: int = Field(default=20, description="Number of recommendations")
    include_emerging: bool = Field(default=True, description="Include emerging careers")
    confidence_threshold: float = Field(default=0.6, description="Minimum match confidence")


class CareerMatchingResult(BaseModel):
    """Result from career matching algorithm."""
    
    career_id: str = Field(..., description="Career identifier")
    match_score: float = Field(..., ge=0, le=1, description="Overall match score")
    
    # Component scores
    raisec_similarity: float = Field(..., ge=0, le=1, description="RAISEC similarity")
    preference_alignment: float = Field(..., ge=0, le=1, description="Preference alignment")
    requirement_feasibility: float = Field(..., ge=0, le=1, description="Requirement feasibility")
    
    # Match details
    matching_dimensions: List[RaisecDimension] = Field(
        ..., 
        description="Strongly matching RAISEC dimensions"
    )
    match_reasoning: List[str] = Field(..., description="Reasons for the match")
    potential_concerns: List[str] = Field(
        default_factory=list,
        description="Potential concerns or mismatches"
    )


# Export all schemas
__all__ = [
    # Enums
    "CareerFieldCategory",
    "EducationLevel",
    "ExperienceLevel", 
    "JobOutlook",
    "WorkEnvironment",
    
    # Request schemas
    "CareerRecommendationRequest",
    "CareerExplorationRequest",
    "CareerSearchRequest",
    "CareerPathRequest",
    
    # Response schemas
    "SkillRequirement",
    "SalaryData",
    "JobMarketData",
    "CareerMatch",
    "CareerRecommendationResponse",
    "CareerDetailResponse",
    "CareerSearchResponse",
    "CareerPathStep",
    "CareerPathResponse",
    "CareerTrendsResponse",
    
    # Processing schemas
    "CareerAnalysisContext",
    "CareerMatchingResult",
]