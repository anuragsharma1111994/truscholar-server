"""Career recommendation models for the TruScholar application.

This module defines models for career recommendations, paths, and related
information based on RAISEC assessment results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, HttpUrl, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import RaisecDimension


class CareerSkill(EmbeddedDocument):
    """Individual skill requirement for a career."""

    name: str = Field(..., min_length=2, max_length=100)
    category: str = Field(..., pattern="^(technical|soft|cognitive|physical)$")
    importance: int = Field(..., ge=1, le=5)  # 1=Nice to have, 5=Essential
    description: Optional[str] = Field(default=None, max_length=300)

    # Skill development
    is_learnable: bool = Field(default=True)
    typical_learning_time_months: Optional[int] = Field(default=None, ge=1)
    learning_resources: List[str] = Field(default_factory=list, max_length=5)


class EducationRequirement(EmbeddedDocument):
    """Education requirements for a career."""

    level: str = Field(..., pattern="^(high_school|diploma|bachelors|masters|doctorate|certification|vocational)$")
    field: Optional[str] = Field(default=None, max_length=100)
    is_mandatory: bool = Field(default=True)
    alternatives: List[str] = Field(default_factory=list, max_length=3)

    # Time and cost estimates
    typical_duration_years: float = Field(default=0.0, ge=0)
    estimated_cost_range: Optional[str] = None  # e.g., "$10,000-$50,000"

    # Specific programs or institutions
    recommended_programs: List[str] = Field(default_factory=list, max_length=5)
    online_options_available: bool = Field(default=False)


class CareerOutlook(EmbeddedDocument):
    """Career outlook and market information."""

    growth_rate_percentage: float = Field(..., ge=-100, le=100)
    growth_classification: str = Field(..., pattern="^(declining|stable|growing|rapidly_growing)$")

    # Job market data
    current_openings_estimate: Optional[int] = Field(default=None, ge=0)
    average_openings_per_year: Optional[int] = Field(default=None, ge=0)
    competition_level: str = Field(default="moderate", pattern="^(low|moderate|high|very_high)$")

    # Salary information
    salary_range_min: Optional[float] = Field(default=None, ge=0)
    salary_range_max: Optional[float] = Field(default=None, ge=0)
    salary_currency: str = Field(default="USD")
    salary_period: str = Field(default="annual", pattern="^(hourly|monthly|annual)$")

    # Geographic considerations
    location_flexibility: str = Field(default="moderate", pattern="^(low|moderate|high|remote)$")
    top_locations: List[str] = Field(default_factory=list, max_length=5)

    # Future trends
    automation_risk: str = Field(default="low", pattern="^(low|moderate|high)$")
    emerging_specializations: List[str] = Field(default_factory=list, max_length=5)


class CareerProgression(EmbeddedDocument):
    """Career progression and advancement information."""

    entry_level_titles: List[str] = Field(default_factory=list, max_length=5)
    mid_level_titles: List[str] = Field(default_factory=list, max_length=5)
    senior_level_titles: List[str] = Field(default_factory=list, max_length=5)

    # Timeline
    typical_entry_to_mid_years: float = Field(default=3.0, ge=0)
    typical_mid_to_senior_years: float = Field(default=5.0, ge=0)

    # Advancement factors
    advancement_factors: List[str] = Field(default_factory=list, max_length=5)
    common_career_transitions: List[str] = Field(default_factory=list, max_length=5)

    # Lateral movements
    related_careers: List[str] = Field(default_factory=list, max_length=5)
    transferable_to: List[str] = Field(default_factory=list, max_length=5)


class DayInLife(EmbeddedDocument):
    """Typical day in the life description for a career."""

    description: str = Field(..., min_length=100, max_length=1000)

    # Time allocation
    typical_activities: List[Dict[str, Any]] = Field(default_factory=list)
    work_environment: str = Field(..., pattern="^(office|remote|hybrid|field|laboratory|studio|outdoor)$")
    interaction_level: str = Field(..., pattern="^(minimal|moderate|high|constant)$")

    # Work characteristics
    physical_demands: str = Field(default="low", pattern="^(low|moderate|high)$")
    stress_level: str = Field(default="moderate", pattern="^(low|moderate|high|variable)$")
    creativity_required: str = Field(default="moderate", pattern="^(low|moderate|high)$")
    problem_solving_frequency: str = Field(default="moderate", pattern="^(rare|occasional|frequent|constant)$")

    # Schedule
    typical_hours_per_week: int = Field(default=40, ge=0, le=80)
    schedule_flexibility: str = Field(default="moderate", pattern="^(rigid|moderate|flexible|highly_flexible)$")
    travel_requirements: str = Field(default="none", pattern="^(none|occasional|frequent|constant)$")


class CareerResource(EmbeddedDocument):
    """Resource for learning more about or pursuing a career."""

    title: str = Field(..., min_length=2, max_length=200)
    type: str = Field(..., pattern="^(website|book|course|video|podcast|article|organization)$")
    url: Optional[HttpUrl] = None
    description: Optional[str] = Field(default=None, max_length=300)
    is_free: bool = Field(default=True)

    # Resource metadata
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    difficulty_level: Optional[str] = Field(default=None, pattern="^(beginner|intermediate|advanced)$")
    estimated_time_hours: Optional[float] = Field(default=None, ge=0)


class Career(BaseDocument):
    """Career information model with comprehensive details."""

    # Basic information
    title: str = Field(..., min_length=2, max_length=200)
    code: str = Field(..., min_length=2, max_length=50)  # NOC or O*NET code
    category: str = Field(..., min_length=2, max_length=100)
    industry: str = Field(..., min_length=2, max_length=100)

    # Descriptions
    summary: str = Field(..., min_length=50, max_length=500)
    detailed_description: str = Field(..., min_length=100, max_length=2000)

    # RAISEC mapping
    primary_raisec_code: str = Field(..., pattern="^[RIASEC]{3}$")
    raisec_scores: Dict[RaisecDimension, float] = Field(...)
    raisec_fit_description: str = Field(..., min_length=50, max_length=500)

    # Career details
    skills_required: List[CareerSkill] = Field(default_factory=list)
    education_requirements: List[EducationRequirement] = Field(default_factory=list)
    outlook: CareerOutlook
    progression: CareerProgression = Field(default_factory=CareerProgression)
    day_in_life: DayInLife

    # Additional information
    certifications: List[str] = Field(default_factory=list, max_length=10)
    professional_associations: List[str] = Field(default_factory=list, max_length=5)

    # Resources
    resources: List[CareerResource] = Field(default_factory=list, max_length=10)

    # Metadata
    data_source: str = Field(default="custom")  # noc, onet, custom
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    popularity_score: float = Field(default=0.0, ge=0, le=100)

    # Tags for search and filtering
    tags: List[str] = Field(default_factory=list, max_length=20)
    keywords: List[str] = Field(default_factory=list, max_length=30)

    @field_validator("raisec_scores")
    @classmethod
    def validate_raisec_scores(cls, v: Dict[RaisecDimension, float]) -> Dict[RaisecDimension, float]:
        """Ensure all RAISEC dimensions have scores."""
        required_dims = {d for d in RaisecDimension}
        if set(v.keys()) != required_dims:
            raise ValueError("All RAISEC dimensions must have scores")

        # Ensure scores are normalized (0-100)
        for dim, score in v.items():
            if not 0 <= score <= 100:
                raise ValueError(f"RAISEC score for {dim} must be between 0 and 100")

        return v

    @model_validator(mode="after")
    def validate_primary_code(self) -> "Career":
        """Ensure primary RAISEC code matches top scoring dimensions."""
        if self.raisec_scores:
            # Get top 3 dimensions by score
            sorted_dims = sorted(
                self.raisec_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            expected_code = "".join([d.value for d, _ in sorted_dims[:3]])

            if self.primary_raisec_code != expected_code:
                # Log warning but don't fail - may be intentional override
                pass

        return self

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the career collection."""
        return [
            ([("code", 1)], {"unique": True}),
            ([("title", 1)], {}),
            ([("primary_raisec_code", 1)], {}),
            ([("category", 1)], {}),
            ([("industry", 1)], {}),
            ([("is_active", 1), ("popularity_score", -1)], {}),
            ([("tags", 1)], {}),
            # Text search index
            ([("title", "text"), ("summary", "text"), ("keywords", "text")], {}),
        ]

    def calculate_fit_score(self, user_raisec_code: str, user_scores: Dict[str, float]) -> float:
        """Calculate how well this career fits a user's RAISEC profile.

        Args:
            user_raisec_code: User's 3-letter RAISEC code
            user_scores: User's dimension scores

        Returns:
            Fit score from 0-100
        """
        fit_score = 0.0

        # Primary code match (40% weight)
        code_match_score = 0.0
        for i, letter in enumerate(user_raisec_code):
            if letter in self.primary_raisec_code:
                position_diff = abs(i - self.primary_raisec_code.index(letter))
                code_match_score += (3 - position_diff) * 13.33  # Max 40 if perfect match

        fit_score += code_match_score

        # Dimension correlation (60% weight)
        correlation_score = 0.0
        for dim, career_score in self.raisec_scores.items():
            user_score = user_scores.get(dim.value, 0)
            # Calculate similarity (inverse of difference)
            diff = abs(career_score - user_score)
            similarity = (100 - diff) / 100
            correlation_score += similarity * 10  # Max 60 for perfect correlation

        fit_score += correlation_score

        return round(fit_score, 2)

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get career summary for display.

        Returns:
            Dictionary with career summary
        """
        return {
            "id": str(self.id),
            "title": self.title,
            "code": self.code,
            "category": self.category,
            "industry": self.industry,
            "summary": self.summary,
            "primary_raisec_code": self.primary_raisec_code,
            "outlook": {
                "growth_rate": self.outlook.growth_rate_percentage,
                "growth_classification": self.outlook.growth_classification,
                "salary_range": f"{self.outlook.salary_currency} {self.outlook.salary_range_min:,.0f}-{self.outlook.salary_range_max:,.0f}" if self.outlook.salary_range_min else "Not specified",
            },
            "education_level": self.education_requirements[0].level if self.education_requirements else "Varies",
            "tags": self.tags[:5],  # Top 5 tags
        }


class CareerRecommendation(BaseDocument):
    """Career recommendation for a specific user test."""

    # References
    test_id: PyObjectId = Field(..., description="Reference to Test")
    user_id: PyObjectId = Field(..., description="Reference to User")
    career_id: PyObjectId = Field(..., description="Reference to Career")

    # Recommendation metadata
    recommendation_number: int = Field(..., ge=1, le=3)  # 1, 2, or 3
    recommendation_type: str = Field(..., pattern="^(traditional|innovative|hybrid)$")

    # Fit analysis
    fit_score: float = Field(..., ge=0, le=100)
    raisec_match_percentage: float = Field(..., ge=0, le=100)
    interest_match_score: float = Field(default=0.0, ge=0, le=100)

    # Detailed reasoning
    reasoning: str = Field(..., min_length=100, max_length=1000)
    strengths_alignment: List[str] = Field(default_factory=list, max_length=5)
    growth_opportunities: List[str] = Field(default_factory=list, max_length=5)
    potential_challenges: List[str] = Field(default_factory=list, max_length=3)

    # Personalized insights
    why_suitable: str = Field(..., min_length=50, max_length=500)
    personality_fit: str = Field(..., min_length=50, max_length=500)
    interest_alignment: Optional[str] = Field(default=None, max_length=500)

    # Next steps
    immediate_actions: List[str] = Field(default_factory=list, max_length=5)
    short_term_goals: List[str] = Field(default_factory=list, max_length=5)
    long_term_goals: List[str] = Field(default_factory=list, max_length=3)

    # Resources specific to user
    personalized_resources: List[CareerResource] = Field(default_factory=list, max_length=5)

    # User context
    considers_current_status: bool = Field(default=False)
    considers_location: bool = Field(default=False)
    considers_education: bool = Field(default=False)

    # Tracking
    generated_by: str = Field(default="gpt-4-turbo-preview")
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    viewed_at: Optional[datetime] = None
    feedback_rating: Optional[int] = Field(default=None, ge=1, le=5)
    feedback_text: Optional[str] = Field(default=None, max_length=500)

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the recommendation collection."""
        return [
            ([("test_id", 1), ("recommendation_number", 1)], {"unique": True}),
            ([("user_id", 1), ("created_at", -1)], {}),
            ([("career_id", 1)], {}),
            ([("fit_score", -1)], {}),
            ([("recommendation_type", 1)], {}),
        ]

    def mark_viewed(self) -> None:
        """Mark recommendation as viewed."""
        if not self.viewed_at:
            self.viewed_at = datetime.utcnow()
            self.update_timestamps()

    def add_feedback(self, rating: int, text: Optional[str] = None) -> None:
        """Add user feedback to recommendation.

        Args:
            rating: Rating from 1-5
            text: Optional feedback text
        """
        self.feedback_rating = rating
        self.feedback_text = text
        self.update_timestamps()

    def to_display_dict(self, include_career: bool = False, career_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert to dictionary for display.

        Args:
            include_career: Whether to include career details
            career_data: Pre-fetched career data to include

        Returns:
            Dictionary with recommendation display data
        """
        display_dict = {
            "recommendation_number": self.recommendation_number,
            "recommendation_type": self.recommendation_type,
            "fit_score": self.fit_score,
            "reasoning": self.reasoning,
            "why_suitable": self.why_suitable,
            "personality_fit": self.personality_fit,
            "strengths_alignment": self.strengths_alignment,
            "growth_opportunities": self.growth_opportunities,
            "potential_challenges": self.potential_challenges,
            "immediate_actions": self.immediate_actions,
            "short_term_goals": self.short_term_goals,
            "long_term_goals": self.long_term_goals,
            "generated_at": self.generation_timestamp.isoformat(),
        }

        if include_career and career_data:
            display_dict["career"] = career_data
        elif include_career:
            display_dict["career_id"] = str(self.career_id)

        if self.interest_alignment:
            display_dict["interest_alignment"] = self.interest_alignment

        if self.personalized_resources:
            display_dict["resources"] = [
                {
                    "title": res.title,
                    "type": res.type,
                    "url": str(res.url) if res.url else None,
                    "description": res.description,
                    "is_free": res.is_free,
                }
                for res in self.personalized_resources[:3]
            ]

        return display_dict


class CareerPath(BaseDocument):
    """Predefined career path for specific RAISEC codes."""

    raisec_code: str = Field(..., pattern="^[RIASEC]{3}$")
    path_name: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., min_length=50, max_length=500)

    # Career progression
    entry_careers: List[PyObjectId] = Field(default_factory=list)  # Career IDs
    mid_careers: List[PyObjectId] = Field(default_factory=list)
    senior_careers: List[PyObjectId] = Field(default_factory=list)

    # Path characteristics
    typical_duration_years: float = Field(..., ge=0)
    education_requirements: List[str] = Field(default_factory=list)
    key_skills: List[str] = Field(default_factory=list, max_length=10)

    # Industries and sectors
    primary_industries: List[str] = Field(default_factory=list, max_length=5)
    emerging_opportunities: List[str] = Field(default_factory=list, max_length=5)

    # Success factors
    success_traits: List[str] = Field(default_factory=list, max_length=10)
    common_backgrounds: List[str] = Field(default_factory=list, max_length=5)

    # Metadata
    is_traditional: bool = Field(default=True)
    popularity_rank: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the career path collection."""
        return [
            ([("raisec_code", 1), ("path_name", 1)], {"unique": True}),
            ([("raisec_code", 1), ("popularity_rank", 1)], {}),
            ([("is_traditional", 1)], {}),
        ]
