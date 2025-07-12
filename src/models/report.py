"""Report model for the TruScholar RAISEC assessment.

This module defines models for comprehensive test reports including
assessment results, career recommendations, and personalized insights.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, HttpUrl, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import AgeGroup, RaisecDimension


class ReportSection(EmbeddedDocument):
    """Base class for report sections."""

    title: str = Field(..., min_length=2, max_length=100)
    content: str = Field(..., min_length=10, max_length=5000)
    order: int = Field(..., ge=0)
    is_visible: bool = Field(default=True)


class PersonalityProfile(ReportSection):
    """Personality profile section of the report."""

    raisec_code: str = Field(..., pattern="^[RIASEC]{3}$")
    primary_type: str = Field(..., min_length=2, max_length=50)
    secondary_type: str = Field(..., min_length=2, max_length=50)
    tertiary_type: str = Field(..., min_length=2, max_length=50)

    # Type descriptions
    primary_description: str = Field(..., min_length=100, max_length=1000)
    secondary_description: str = Field(..., min_length=100, max_length=1000)
    tertiary_description: str = Field(..., min_length=100, max_length=1000)

    # Personality insights
    strengths: List[str] = Field(..., min_length=3, max_length=10)
    work_style: str = Field(..., min_length=50, max_length=500)
    team_dynamics: str = Field(..., min_length=50, max_length=500)
    leadership_style: str = Field(..., min_length=50, max_length=500)

    # Motivations and values
    core_motivations: List[str] = Field(..., min_length=3, max_length=5)
    work_values: List[str] = Field(..., min_length=3, max_length=5)
    ideal_environment: str = Field(..., min_length=50, max_length=500)


class DimensionAnalysis(EmbeddedDocument):
    """Detailed analysis of a RAISEC dimension."""

    dimension: RaisecDimension
    score: float = Field(..., ge=0, le=100)
    percentile: float = Field(..., ge=0, le=100)
    interpretation: str = Field(..., min_length=50, max_length=500)

    # Behavioral indicators
    high_score_behaviors: List[str] = Field(default_factory=list, max_length=5)
    low_score_behaviors: List[str] = Field(default_factory=list, max_length=5)

    # Development suggestions
    development_tips: List[str] = Field(default_factory=list, max_length=5)
    complementary_dimensions: List[RaisecDimension] = Field(default_factory=list, max_length=2)


class AssessmentResults(ReportSection):
    """Assessment results section with detailed scoring."""

    dimension_analyses: List[DimensionAnalysis] = Field(..., min_length=6, max_length=6)

    # Overall metrics
    profile_consistency: float = Field(..., ge=0, le=100)
    profile_differentiation: float = Field(..., ge=0, le=100)
    response_consistency: float = Field(..., ge=0, le=100)

    # Score interpretation
    score_pattern: str = Field(..., pattern="^(balanced|differentiated|specialized|undifferentiated)$")
    pattern_interpretation: str = Field(..., min_length=100, max_length=1000)

    # Comparative analysis
    compared_to_age_group: bool = Field(default=True)
    age_group_comparison: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "AssessmentResults":
        """Ensure all RAISEC dimensions are present."""
        dims = {analysis.dimension for analysis in self.dimension_analyses}
        required_dims = {d for d in RaisecDimension}
        if dims != required_dims:
            raise ValueError("All RAISEC dimensions must be analyzed")
        return self


class CareerInsights(ReportSection):
    """Career insights and recommendations section."""

    recommended_career_count: int = Field(..., ge=1, le=10)
    career_categories: List[str] = Field(..., min_length=1, max_length=10)

    # Career fit analysis
    best_fit_industries: List[str] = Field(..., min_length=3, max_length=10)
    emerging_opportunities: List[str] = Field(default_factory=list, max_length=5)

    # Work environment preferences
    ideal_work_settings: List[str] = Field(..., min_length=3, max_length=5)
    preferred_company_sizes: List[str] = Field(default_factory=list, max_length=3)
    cultural_fit_factors: List[str] = Field(default_factory=list, max_length=5)

    # Skills and development
    natural_abilities: List[str] = Field(..., min_length=3, max_length=10)
    skills_to_develop: List[str] = Field(..., min_length=3, max_length=10)
    learning_recommendations: List[str] = Field(default_factory=list, max_length=5)


class ActionPlan(ReportSection):
    """Actionable next steps and planning section."""

    timeline: str = Field(..., pattern="^(immediate|short_term|long_term)$")

    # Immediate actions (1-4 weeks)
    immediate_steps: List[Dict[str, str]] = Field(..., min_length=3, max_length=5)

    # Short-term goals (1-6 months)
    short_term_goals: List[Dict[str, str]] = Field(..., min_length=3, max_length=5)

    # Long-term vision (6+ months)
    long_term_objectives: List[Dict[str, str]] = Field(..., min_length=2, max_length=3)

    # Resources and support
    recommended_resources: List[Dict[str, Any]] = Field(default_factory=list, max_length=10)
    support_networks: List[str] = Field(default_factory=list, max_length=5)

    # Milestones and checkpoints
    milestones: List[Dict[str, Any]] = Field(default_factory=list, max_length=5)
    success_metrics: List[str] = Field(default_factory=list, max_length=5)


class PersonalNarrative(ReportSection):
    """Personalized narrative section of the report."""

    opening_statement: str = Field(..., min_length=50, max_length=300)
    journey_narrative: str = Field(..., min_length=200, max_length=2000)

    # Personal story elements
    unique_combination: str = Field(..., min_length=100, max_length=500)
    potential_paths: str = Field(..., min_length=100, max_length=1000)
    inspiration_message: str = Field(..., min_length=50, max_length=500)

    # Customization based on user context
    considers_age: bool = Field(default=True)
    considers_interests: bool = Field(default=False)
    considers_current_status: bool = Field(default=False)


class VisualData(EmbeddedDocument):
    """Data for visual representations in the report."""

    chart_type: str = Field(..., pattern="^(radar|bar|pie|hexagon|line)$")
    chart_id: str = Field(..., min_length=2, max_length=50)

    # Chart data
    labels: List[str] = Field(...)
    datasets: List[Dict[str, Any]] = Field(...)

    # Display options
    title: Optional[str] = None
    colors: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)


class Report(BaseDocument):
    """Comprehensive test report with all results and recommendations."""

    # Core references
    test_id: PyObjectId = Field(..., description="Reference to Test")
    user_id: PyObjectId = Field(..., description="Reference to User")

    # Report metadata
    report_version: str = Field(default="1.0")
    report_type: str = Field(default="comprehensive", pattern="^(comprehensive|summary|detailed)$")
    language: str = Field(default="en", pattern="^[a-z]{2}$")

    # User context
    user_name: str = Field(..., min_length=2, max_length=100)
    age_group: AgeGroup
    test_date: datetime

    # RAISEC results
    raisec_code: str = Field(..., pattern="^[RIASEC]{3}$")
    dimension_scores: Dict[RaisecDimension, float] = Field(...)

    # Report sections
    personality_profile: PersonalityProfile
    assessment_results: AssessmentResults
    career_insights: CareerInsights
    action_plan: ActionPlan
    personal_narrative: PersonalNarrative

    # Career recommendations summary
    career_recommendation_ids: List[PyObjectId] = Field(..., min_length=1, max_length=3)
    career_summaries: List[Dict[str, Any]] = Field(default_factory=list)

    # Visual data for charts
    visual_data: List[VisualData] = Field(default_factory=list)

    # Additional insights
    key_takeaways: List[str] = Field(..., min_length=3, max_length=5)
    quotes: List[Dict[str, str]] = Field(default_factory=list, max_length=3)

    # Report status
    is_finalized: bool = Field(default=False)
    finalized_at: Optional[datetime] = None

    # Delivery and access
    access_code: Optional[str] = Field(default=None, min_length=6, max_length=20)
    expires_at: Optional[datetime] = None
    view_count: int = Field(default=0, ge=0)
    last_viewed_at: Optional[datetime] = None

    # Export history
    exported_formats: List[str] = Field(default_factory=list)
    last_exported_at: Optional[datetime] = None

    # Feedback
    satisfaction_rating: Optional[int] = Field(default=None, ge=1, le=5)
    feedback_text: Optional[str] = Field(default=None, max_length=1000)
    feedback_submitted_at: Optional[datetime] = None

    @field_validator("dimension_scores")
    @classmethod
    def validate_dimension_scores(cls, v: Dict[RaisecDimension, float]) -> Dict[RaisecDimension, float]:
        """Ensure all dimensions have scores."""
        required_dims = {d for d in RaisecDimension}
        if set(v.keys()) != required_dims:
            raise ValueError("All RAISEC dimensions must have scores")
        return v

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the report collection."""
        return [
            ([("test_id", 1)], {"unique": True}),
            ([("user_id", 1), ("created_at", -1)], {}),
            ([("access_code", 1)], {"sparse": True}),
            ([("raisec_code", 1)], {}),
            ([("is_finalized", 1), ("created_at", -1)], {}),
            ([("expires_at", 1)], {"expireAfterSeconds": 0}),
        ]

    def generate_visual_data(self) -> None:
        """Generate visual data for charts."""
        # RAISEC Hexagon/Radar Chart
        self.visual_data.append(VisualData(
            chart_type="hexagon",
            chart_id="raisec_hexagon",
            labels=[d.value for d in RaisecDimension],
            datasets=[{
                "label": "Your Scores",
                "data": [self.dimension_scores[d] for d in RaisecDimension],
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 2,
            }],
            title="Your RAISEC Profile",
            options={
                "scale": {
                    "ticks": {
                        "beginAtZero": True,
                        "max": 100
                    }
                }
            }
        ))

        # Dimension Bar Chart
        self.visual_data.append(VisualData(
            chart_type="bar",
            chart_id="dimension_bars",
            labels=[d.name for d in RaisecDimension],
            datasets=[{
                "label": "Dimension Scores",
                "data": [self.dimension_scores[d] for d in RaisecDimension],
                "backgroundColor": [
                    "#FF6384", "#36A2EB", "#FFCE56",
                    "#4BC0C0", "#9966FF", "#FF9F40"
                ],
            }],
            title="RAISEC Dimension Scores",
            options={
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100
                    }
                }
            }
        ))

    def finalize_report(self) -> None:
        """Finalize the report for delivery."""
        if not self.is_finalized:
            self.is_finalized = True
            self.finalized_at = datetime.utcnow()
            self.generate_visual_data()
            self.update_timestamps()

    def record_view(self) -> None:
        """Record a report view."""
        self.view_count += 1
        self.last_viewed_at = datetime.utcnow()
        self.update_timestamps()

    def add_feedback(self, rating: int, text: Optional[str] = None) -> None:
        """Add user feedback to the report.

        Args:
            rating: Satisfaction rating 1-5
            text: Optional feedback text
        """
        self.satisfaction_rating = rating
        self.feedback_text = text
        self.feedback_submitted_at = datetime.utcnow()
        self.update_timestamps()

    def export_format(self, format_type: str) -> None:
        """Record an export event.

        Args:
            format_type: Export format (pdf, html, json)
        """
        if format_type not in self.exported_formats:
            self.exported_formats.append(format_type)
        self.last_exported_at = datetime.utcnow()
        self.update_timestamps()

    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary of the report.

        Returns:
            Dictionary with key report highlights
        """
        return {
            "user_name": self.user_name,
            "test_date": self.test_date.isoformat(),
            "raisec_code": self.raisec_code,
            "primary_type": self.personality_profile.primary_type,
            "top_strengths": self.personality_profile.strengths[:3],
            "best_fit_industries": self.career_insights.best_fit_industries[:3],
            "key_takeaways": self.key_takeaways,
            "immediate_actions": [
                step["action"] for step in self.action_plan.immediate_steps[:3]
            ],
        }

    def to_display_dict(self, include_visuals: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for display.

        Args:
            include_visuals: Whether to include visual data

        Returns:
            Dictionary with display-ready report data
        """
        display_dict = {
            "report_id": str(self.id),
            "user_name": self.user_name,
            "test_date": self.test_date.isoformat(),
            "age_group": self.age_group.value,
            "raisec_code": self.raisec_code,
            "personality_profile": {
                "title": self.personality_profile.title,
                "raisec_code": self.personality_profile.raisec_code,
                "primary_type": self.personality_profile.primary_type,
                "secondary_type": self.personality_profile.secondary_type,
                "tertiary_type": self.personality_profile.tertiary_type,
                "primary_description": self.personality_profile.primary_description,
                "strengths": self.personality_profile.strengths,
                "work_style": self.personality_profile.work_style,
                "core_motivations": self.personality_profile.core_motivations,
                "ideal_environment": self.personality_profile.ideal_environment,
            },
            "assessment_results": {
                "title": self.assessment_results.title,
                "dimension_scores": {
                    analysis.dimension.value: {
                        "score": analysis.score,
                        "percentile": analysis.percentile,
                        "interpretation": analysis.interpretation,
                    }
                    for analysis in self.assessment_results.dimension_analyses
                },
                "profile_consistency": self.assessment_results.profile_consistency,
                "score_pattern": self.assessment_results.score_pattern,
                "pattern_interpretation": self.assessment_results.pattern_interpretation,
            },
            "career_insights": {
                "title": self.career_insights.title,
                "best_fit_industries": self.career_insights.best_fit_industries,
                "ideal_work_settings": self.career_insights.ideal_work_settings,
                "natural_abilities": self.career_insights.natural_abilities,
                "skills_to_develop": self.career_insights.skills_to_develop,
            },
            "action_plan": {
                "title": self.action_plan.title,
                "immediate_steps": self.action_plan.immediate_steps,
                "short_term_goals": self.action_plan.short_term_goals,
                "long_term_objectives": self.action_plan.long_term_objectives,
            },
            "personal_narrative": {
                "title": self.personal_narrative.title,
                "opening_statement": self.personal_narrative.opening_statement,
                "journey_narrative": self.personal_narrative.journey_narrative,
                "inspiration_message": self.personal_narrative.inspiration_message,
            },
            "career_recommendations": self.career_summaries,
            "key_takeaways": self.key_takeaways,
            "generated_at": self.created_at.isoformat(),
        }

        if include_visuals:
            display_dict["visual_data"] = [
                {
                    "chart_type": vis.chart_type,
                    "chart_id": vis.chart_id,
                    "data": {
                        "labels": vis.labels,
                        "datasets": vis.datasets,
                    },
                    "options": vis.options,
                }
                for vis in self.visual_data
            ]

        if self.quotes:
            display_dict["inspirational_quotes"] = self.quotes

        return display_dict

    def __repr__(self) -> str:
        """String representation of Report."""
        return (
            f"<Report(id={self.id}, user={self.user_name}, "
            f"raisec={self.raisec_code}, finalized={self.is_finalized})>"
        )


class ReportTemplate(BaseDocument):
    """Templates for generating report content."""

    template_name: str = Field(..., min_length=2, max_length=100)
    template_type: str = Field(..., pattern="^(personality|career|action_plan|narrative)$")
    age_group: Optional[AgeGroup] = None
    raisec_code_pattern: Optional[str] = Field(default=None, pattern="^[RIASEC*]{3}$")

    # Template content
    template_text: str = Field(..., min_length=50, max_length=5000)
    variables: List[str] = Field(default_factory=list)

    # Metadata
    version: str = Field(default="1.0")
    is_active: bool = Field(default=True)
    usage_count: int = Field(default=0, ge=0)

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the template collection."""
        return [
            ([("template_name", 1), ("version", 1)], {"unique": True}),
            ([("template_type", 1), ("is_active", 1)], {}),
            ([("age_group", 1)], {"sparse": True}),
        ]
