"""LangChain-based career recommendation system.

This module provides intelligent career recommendations using LangChain chains,
personalized analysis, and AI-powered career insights based on RAISEC assessments.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from src.core.config import get_settings
from src.utils.constants import RaisecDimension
from src.utils.exceptions import TruScholarError
from src.utils.logger import get_logger
from src.schemas.career_schemas import (
    CareerAnalysisContext,
    CareerMatch,
    EducationLevel,
    ExperienceLevel,
    WorkEnvironment
)
from src.langchain_handlers.chains.career_chain import CareerChain
from src.langchain_handlers.parsers.career_parser import CareerParser
from src.langchain_handlers.prompts.career_prompts import CareerPrompts

settings = get_settings()
logger = get_logger(__name__)


class CareerRecommendationError(TruScholarError):
    """Exception raised when career recommendation generation fails."""
    pass


@dataclass
class RecommendationContext:
    """Context for career recommendation generation."""
    user_raisec_code: str
    user_raisec_scores: Dict[str, float]
    user_preferences: Dict[str, Any]
    career_database: Dict[str, Any]
    recommendation_count: int = 20
    include_explanations: bool = True
    focus_areas: Optional[List[str]] = None


class CareerRecommender:
    """LangChain-based career recommendation system."""
    
    def __init__(
        self,
        llm=None,
        enable_caching: bool = True,
        max_retries: int = 3,
        temperature: float = 0.7
    ):
        """Initialize career recommender.
        
        Args:
            llm: Language model instance
            enable_caching: Whether to enable response caching
            max_retries: Maximum retry attempts for failed generations
            temperature: LLM temperature for creativity vs consistency
        """
        self.llm = llm
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        self.temperature = temperature
        
        # Initialize components
        self.career_chain = CareerChain(llm=llm, temperature=temperature)
        self.career_parser = CareerParser()
        self.prompts = CareerPrompts()
        
        # Statistics tracking
        self.generation_stats = {
            "total_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "cache_hits": 0,
            "average_generation_time": 0.0
        }
    
    async def generate_career_recommendations(
        self,
        context: RecommendationContext,
        recommendation_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Generate personalized career recommendations.
        
        Args:
            context: Recommendation context with user data
            recommendation_type: Type of recommendations (comprehensive, focused, exploratory)
            
        Returns:
            Dictionary containing career recommendations and insights
            
        Raises:
            CareerRecommendationError: If recommendation generation fails
        """
        start_time = datetime.utcnow()
        logger.info(f"Generating {recommendation_type} career recommendations for RAISEC: {context.user_raisec_code}")
        
        try:
            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(context, recommendation_type)
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    self.generation_stats["cache_hits"] += 1
                    return cached_result
            
            # Generate recommendations based on type
            if recommendation_type == "comprehensive":
                result = await self._generate_comprehensive_recommendations(context)
            elif recommendation_type == "focused":
                result = await self._generate_focused_recommendations(context)
            elif recommendation_type == "exploratory":
                result = await self._generate_exploratory_recommendations(context)
            else:
                raise ValueError(f"Unknown recommendation type: {recommendation_type}")
            
            # Enhance with AI insights
            result = await self._enhance_with_ai_insights(result, context)
            
            # Add metadata
            result["generation_metadata"] = {
                "recommendation_type": recommendation_type,
                "generated_at": datetime.utcnow().isoformat(),
                "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "raisec_code": context.user_raisec_code,
                "model_version": "v1.0"
            }
            
            # Cache result if enabled
            if self.enable_caching:
                await self._cache_result(cache_key, result)
            
            # Update statistics
            self._update_generation_stats(True, start_time)
            
            logger.info(f"Successfully generated career recommendations")
            return result
            
        except Exception as e:
            self._update_generation_stats(False, start_time)
            logger.error(f"Failed to generate career recommendations: {str(e)}")
            raise CareerRecommendationError(f"Recommendation generation failed: {str(e)}")
    
    async def generate_career_insights(
        self,
        career_id: str,
        user_context: RecommendationContext,
        insight_type: str = "match_analysis"
    ) -> Dict[str, Any]:
        """Generate personalized insights for a specific career.
        
        Args:
            career_id: Career identifier
            user_context: User context for personalization
            insight_type: Type of insights (match_analysis, growth_potential, challenges)
            
        Returns:
            Dictionary containing career insights
        """
        logger.info(f"Generating {insight_type} insights for career: {career_id}")
        
        try:
            career_data = user_context.career_database.get(career_id)
            if not career_data:
                raise ValueError(f"Career not found: {career_id}")
            
            # Generate insights based on type
            if insight_type == "match_analysis":
                insights = await self._generate_match_analysis(career_data, user_context)
            elif insight_type == "growth_potential":
                insights = await self._generate_growth_potential_analysis(career_data, user_context)
            elif insight_type == "challenges":
                insights = await self._generate_challenges_analysis(career_data, user_context)
            elif insight_type == "transition_path":
                insights = await self._generate_transition_path_analysis(career_data, user_context)
            else:
                raise ValueError(f"Unknown insight type: {insight_type}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate career insights: {str(e)}")
            raise CareerRecommendationError(f"Insight generation failed: {str(e)}")
    
    async def generate_career_comparison(
        self,
        career_ids: List[str],
        user_context: RecommendationContext,
        comparison_aspects: List[str] = None
    ) -> Dict[str, Any]:
        """Generate comparison between multiple careers.
        
        Args:
            career_ids: List of career IDs to compare
            user_context: User context for personalization
            comparison_aspects: Specific aspects to compare
            
        Returns:
            Dictionary containing career comparison
        """
        logger.info(f"Generating comparison for careers: {career_ids}")
        
        if len(career_ids) < 2:
            raise ValueError("At least 2 careers required for comparison")
        
        try:
            # Get career data
            careers_data = []
            for career_id in career_ids:
                career_data = user_context.career_database.get(career_id)
                if career_data:
                    careers_data.append(career_data)
            
            if len(careers_data) < 2:
                raise ValueError("Not enough valid careers found for comparison")
            
            # Generate comparison
            comparison_prompt = self.prompts.get_career_comparison_prompt(
                careers_data=careers_data,
                user_raisec_code=user_context.user_raisec_code,
                user_preferences=user_context.user_preferences,
                comparison_aspects=comparison_aspects or [
                    "raisec_fit", "growth_potential", "salary", "work_life_balance", 
                    "education_requirements", "job_market"
                ]
            )
            
            comparison_result = await self.career_chain.ainvoke({
                "comparison_prompt": comparison_prompt,
                "careers_count": len(careers_data),
                "user_raisec": user_context.user_raisec_code
            })
            
            # Parse and structure the comparison
            parsed_comparison = self.career_parser.parse_career_comparison(comparison_result)
            
            return {
                "careers_compared": career_ids,
                "comparison_aspects": comparison_aspects,
                "comparison_results": parsed_comparison,
                "recommendation": await self._generate_comparison_recommendation(
                    parsed_comparison, user_context
                ),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate career comparison: {str(e)}")
            raise CareerRecommendationError(f"Comparison generation failed: {str(e)}")
    
    async def generate_personalized_career_narrative(
        self,
        recommended_careers: List[Dict[str, Any]],
        user_context: RecommendationContext
    ) -> str:
        """Generate a personalized narrative explaining career recommendations.
        
        Args:
            recommended_careers: List of recommended career data
            user_context: User context for personalization
            
        Returns:
            Personalized narrative string
        """
        logger.info("Generating personalized career narrative")
        
        try:
            narrative_prompt = self.prompts.get_career_narrative_prompt(
                user_raisec_code=user_context.user_raisec_code,
                user_raisec_scores=user_context.user_raisec_scores,
                recommended_careers=recommended_careers[:5],  # Top 5 for narrative
                user_preferences=user_context.user_preferences
            )
            
            narrative_result = await self.career_chain.ainvoke({
                "narrative_prompt": narrative_prompt,
                "user_raisec": user_context.user_raisec_code,
                "career_count": len(recommended_careers)
            })
            
            # Parse and enhance narrative
            narrative = self.career_parser.parse_career_narrative(narrative_result)
            
            return narrative
            
        except Exception as e:
            logger.error(f"Failed to generate career narrative: {str(e)}")
            raise CareerRecommendationError(f"Narrative generation failed: {str(e)}")
    
    async def generate_skill_development_plan(
        self,
        target_career: Dict[str, Any],
        user_context: RecommendationContext,
        current_skills: List[str] = None,
        timeline: str = "1_year"
    ) -> Dict[str, Any]:
        """Generate personalized skill development plan for a target career.
        
        Args:
            target_career: Target career data
            user_context: User context
            current_skills: User's current skills
            timeline: Development timeline (6_months, 1_year, 2_years)
            
        Returns:
            Skill development plan
        """
        logger.info(f"Generating skill development plan for: {target_career.get('title', 'Unknown')}")
        
        try:
            skill_plan_prompt = self.prompts.get_skill_development_prompt(
                target_career=target_career,
                user_raisec_code=user_context.user_raisec_code,
                current_skills=current_skills or [],
                timeline=timeline,
                user_preferences=user_context.user_preferences
            )
            
            plan_result = await self.career_chain.ainvoke({
                "skill_plan_prompt": skill_plan_prompt,
                "timeline": timeline,
                "career_title": target_career.get("title", "")
            })
            
            # Parse skill development plan
            parsed_plan = self.career_parser.parse_skill_development_plan(plan_result)
            
            return parsed_plan
            
        except Exception as e:
            logger.error(f"Failed to generate skill development plan: {str(e)}")
            raise CareerRecommendationError(f"Skill plan generation failed: {str(e)}")
    
    async def get_recommendation_statistics(self) -> Dict[str, Any]:
        """Get recommendation generation statistics.
        
        Returns:
            Dictionary containing generation statistics
        """
        success_rate = 0.0
        if self.generation_stats["total_generated"] > 0:
            success_rate = (
                self.generation_stats["successful_generations"] / 
                self.generation_stats["total_generated"]
            )
        
        return {
            "total_generated": self.generation_stats["total_generated"],
            "success_rate": success_rate,
            "cache_hit_rate": (
                self.generation_stats["cache_hits"] / 
                max(self.generation_stats["total_generated"], 1)
            ),
            "average_generation_time": self.generation_stats["average_generation_time"]
        }
    
    # Private helper methods
    
    async def _generate_comprehensive_recommendations(
        self,
        context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate comprehensive career recommendations."""
        
        # Create comprehensive prompt
        recommendation_prompt = self.prompts.get_comprehensive_recommendation_prompt(
            user_raisec_code=context.user_raisec_code,
            user_raisec_scores=context.user_raisec_scores,
            user_preferences=context.user_preferences,
            recommendation_count=context.recommendation_count
        )
        
        # Generate recommendations
        recommendation_result = await self.career_chain.ainvoke({
            "recommendation_prompt": recommendation_prompt,
            "user_raisec": context.user_raisec_code,
            "recommendation_count": context.recommendation_count
        })
        
        # Parse recommendations
        parsed_recommendations = self.career_parser.parse_career_recommendations(
            recommendation_result
        )
        
        return {
            "recommendation_type": "comprehensive",
            "recommendations": parsed_recommendations,
            "analysis_depth": "detailed",
            "confidence_level": self._calculate_recommendation_confidence(parsed_recommendations)
        }
    
    async def _generate_focused_recommendations(
        self,
        context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate focused career recommendations based on strongest dimensions."""
        
        # Identify top 2 RAISEC dimensions
        top_dimensions = sorted(
            context.user_raisec_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        focus_areas = [dim[0] for dim in top_dimensions]
        
        # Create focused prompt
        focused_prompt = self.prompts.get_focused_recommendation_prompt(
            user_raisec_code=context.user_raisec_code,
            focus_dimensions=focus_areas,
            user_preferences=context.user_preferences,
            recommendation_count=min(context.recommendation_count, 15)
        )
        
        # Generate focused recommendations
        recommendation_result = await self.career_chain.ainvoke({
            "focused_prompt": focused_prompt,
            "focus_areas": focus_areas,
            "user_raisec": context.user_raisec_code
        })
        
        # Parse recommendations
        parsed_recommendations = self.career_parser.parse_career_recommendations(
            recommendation_result
        )
        
        return {
            "recommendation_type": "focused",
            "focus_areas": focus_areas,
            "recommendations": parsed_recommendations,
            "analysis_depth": "targeted",
            "confidence_level": self._calculate_recommendation_confidence(parsed_recommendations)
        }
    
    async def _generate_exploratory_recommendations(
        self,
        context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate exploratory career recommendations including emerging fields."""
        
        # Create exploratory prompt
        exploratory_prompt = self.prompts.get_exploratory_recommendation_prompt(
            user_raisec_code=context.user_raisec_code,
            user_raisec_scores=context.user_raisec_scores,
            include_emerging=True,
            include_unconventional=True,
            recommendation_count=context.recommendation_count
        )
        
        # Generate exploratory recommendations
        recommendation_result = await self.career_chain.ainvoke({
            "exploratory_prompt": exploratory_prompt,
            "user_raisec": context.user_raisec_code,
            "exploration_level": "high"
        })
        
        # Parse recommendations
        parsed_recommendations = self.career_parser.parse_career_recommendations(
            recommendation_result
        )
        
        return {
            "recommendation_type": "exploratory",
            "includes_emerging": True,
            "includes_unconventional": True,
            "recommendations": parsed_recommendations,
            "analysis_depth": "broad",
            "confidence_level": self._calculate_recommendation_confidence(parsed_recommendations)
        }
    
    async def _enhance_with_ai_insights(
        self,
        recommendations: Dict[str, Any],
        context: RecommendationContext
    ) -> Dict[str, Any]:
        """Enhance recommendations with AI-generated insights."""
        
        # Generate overall insights
        insights_prompt = self.prompts.get_career_insights_prompt(
            user_raisec_code=context.user_raisec_code,
            recommendations=recommendations.get("recommendations", []),
            user_preferences=context.user_preferences
        )
        
        insights_result = await self.career_chain.ainvoke({
            "insights_prompt": insights_prompt,
            "user_raisec": context.user_raisec_code
        })
        
        # Parse insights
        parsed_insights = self.career_parser.parse_career_insights(insights_result)
        
        # Add insights to recommendations
        recommendations["ai_insights"] = parsed_insights
        
        # Generate success factors
        success_factors = await self._generate_success_factors(
            recommendations.get("recommendations", []),
            context
        )
        recommendations["success_factors"] = success_factors
        
        # Generate development priorities
        development_priorities = await self._generate_development_priorities(
            recommendations.get("recommendations", []),
            context
        )
        recommendations["development_priorities"] = development_priorities
        
        return recommendations
    
    async def _generate_match_analysis(
        self,
        career_data: Dict[str, Any],
        user_context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate detailed match analysis for a specific career."""
        
        match_prompt = self.prompts.get_match_analysis_prompt(
            career_data=career_data,
            user_raisec_code=user_context.user_raisec_code,
            user_raisec_scores=user_context.user_raisec_scores,
            user_preferences=user_context.user_preferences
        )
        
        match_result = await self.career_chain.ainvoke({
            "match_prompt": match_prompt,
            "career_title": career_data.get("title", ""),
            "user_raisec": user_context.user_raisec_code
        })
        
        return self.career_parser.parse_match_analysis(match_result)
    
    async def _generate_growth_potential_analysis(
        self,
        career_data: Dict[str, Any],
        user_context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate growth potential analysis for a career."""
        
        growth_prompt = self.prompts.get_growth_potential_prompt(
            career_data=career_data,
            user_raisec_code=user_context.user_raisec_code,
            user_preferences=user_context.user_preferences
        )
        
        growth_result = await self.career_chain.ainvoke({
            "growth_prompt": growth_prompt,
            "career_title": career_data.get("title", "")
        })
        
        return self.career_parser.parse_growth_potential(growth_result)
    
    async def _generate_challenges_analysis(
        self,
        career_data: Dict[str, Any],
        user_context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate challenges analysis for a career."""
        
        challenges_prompt = self.prompts.get_challenges_analysis_prompt(
            career_data=career_data,
            user_raisec_code=user_context.user_raisec_code,
            user_preferences=user_context.user_preferences
        )
        
        challenges_result = await self.career_chain.ainvoke({
            "challenges_prompt": challenges_prompt,
            "career_title": career_data.get("title", "")
        })
        
        return self.career_parser.parse_challenges_analysis(challenges_result)
    
    async def _generate_transition_path_analysis(
        self,
        career_data: Dict[str, Any],
        user_context: RecommendationContext
    ) -> Dict[str, Any]:
        """Generate transition path analysis for a career."""
        
        transition_prompt = self.prompts.get_transition_path_prompt(
            target_career=career_data,
            user_raisec_code=user_context.user_raisec_code,
            user_preferences=user_context.user_preferences
        )
        
        transition_result = await self.career_chain.ainvoke({
            "transition_prompt": transition_prompt,
            "target_career": career_data.get("title", "")
        })
        
        return self.career_parser.parse_transition_path(transition_result)
    
    async def _generate_comparison_recommendation(
        self,
        comparison_data: Dict[str, Any],
        user_context: RecommendationContext
    ) -> str:
        """Generate recommendation based on career comparison."""
        
        recommendation_prompt = self.prompts.get_comparison_recommendation_prompt(
            comparison_data=comparison_data,
            user_raisec_code=user_context.user_raisec_code,
            user_preferences=user_context.user_preferences
        )
        
        recommendation_result = await self.career_chain.ainvoke({
            "recommendation_prompt": recommendation_prompt,
            "user_raisec": user_context.user_raisec_code
        })
        
        return self.career_parser.parse_recommendation_text(recommendation_result)
    
    async def _generate_success_factors(
        self,
        recommendations: List[Dict[str, Any]],
        context: RecommendationContext
    ) -> List[str]:
        """Generate success factors for recommended careers."""
        
        if not recommendations:
            return []
        
        success_prompt = self.prompts.get_success_factors_prompt(
            recommendations=recommendations[:5],  # Top 5
            user_raisec_code=context.user_raisec_code,
            user_preferences=context.user_preferences
        )
        
        success_result = await self.career_chain.ainvoke({
            "success_prompt": success_prompt,
            "user_raisec": context.user_raisec_code
        })
        
        return self.career_parser.parse_success_factors(success_result)
    
    async def _generate_development_priorities(
        self,
        recommendations: List[Dict[str, Any]],
        context: RecommendationContext
    ) -> List[Dict[str, Any]]:
        """Generate development priorities based on recommendations."""
        
        if not recommendations:
            return []
        
        priorities_prompt = self.prompts.get_development_priorities_prompt(
            recommendations=recommendations[:5],  # Top 5
            user_raisec_code=context.user_raisec_code,
            user_preferences=context.user_preferences
        )
        
        priorities_result = await self.career_chain.ainvoke({
            "priorities_prompt": priorities_prompt,
            "user_raisec": context.user_raisec_code
        })
        
        return self.career_parser.parse_development_priorities(priorities_result)
    
    def _calculate_recommendation_confidence(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence level for recommendations."""
        if not recommendations:
            return 0.0
        
        # Base confidence on number and quality of recommendations
        base_confidence = min(len(recommendations) / 10, 1.0) * 0.6
        
        # Add confidence based on recommendation scores (if available)
        scores = [rec.get("match_score", 75) for rec in recommendations[:5]]
        if scores:
            avg_score = sum(scores) / len(scores)
            score_confidence = (avg_score / 100) * 0.4
        else:
            score_confidence = 0.3
        
        return min(base_confidence + score_confidence, 0.95)
    
    def _generate_cache_key(
        self,
        context: RecommendationContext,
        recommendation_type: str
    ) -> str:
        """Generate cache key for recommendations."""
        key_parts = [
            context.user_raisec_code,
            recommendation_type,
            str(context.recommendation_count),
            str(hash(str(context.user_preferences)))
        ]
        return f"career_rec:{'_'.join(key_parts)}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached recommendation result."""
        # Placeholder for cache implementation
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache recommendation result."""
        # Placeholder for cache implementation
        pass
    
    def _update_generation_stats(self, success: bool, start_time: datetime) -> None:
        """Update generation statistics."""
        self.generation_stats["total_generated"] += 1
        
        if success:
            self.generation_stats["successful_generations"] += 1
        else:
            self.generation_stats["failed_generations"] += 1
        
        # Update average generation time
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        current_avg = self.generation_stats["average_generation_time"]
        total_count = self.generation_stats["total_generated"]
        
        new_avg = ((current_avg * (total_count - 1)) + generation_time) / total_count
        self.generation_stats["average_generation_time"] = new_avg


# Convenience functions for direct use

async def generate_career_recommendations(
    user_raisec_code: str,
    user_raisec_scores: Dict[str, float],
    user_preferences: Dict[str, Any] = None,
    career_database: Dict[str, Any] = None,
    recommendation_count: int = 20,
    recommendation_type: str = "comprehensive"
) -> Dict[str, Any]:
    """Convenience function to generate career recommendations.
    
    Args:
        user_raisec_code: User's RAISEC code
        user_raisec_scores: User's RAISEC dimension scores
        user_preferences: User preferences dictionary
        career_database: Career database
        recommendation_count: Number of recommendations
        recommendation_type: Type of recommendations
        
    Returns:
        Career recommendations dictionary
    """
    recommender = CareerRecommender()
    
    context = RecommendationContext(
        user_raisec_code=user_raisec_code,
        user_raisec_scores=user_raisec_scores,
        user_preferences=user_preferences or {},
        career_database=career_database or {},
        recommendation_count=recommendation_count
    )
    
    return await recommender.generate_career_recommendations(context, recommendation_type)


async def generate_career_narrative(
    user_raisec_code: str,
    recommended_careers: List[Dict[str, Any]],
    user_preferences: Dict[str, Any] = None
) -> str:
    """Convenience function to generate career narrative.
    
    Args:
        user_raisec_code: User's RAISEC code
        recommended_careers: List of recommended careers
        user_preferences: User preferences
        
    Returns:
        Personalized career narrative
    """
    recommender = CareerRecommender()
    
    context = RecommendationContext(
        user_raisec_code=user_raisec_code,
        user_raisec_scores={},  # Not needed for narrative
        user_preferences=user_preferences or {},
        career_database={}  # Not needed for narrative
    )
    
    return await recommender.generate_personalized_career_narrative(
        recommended_careers, context
    )


async def compare_careers(
    career_ids: List[str],
    user_raisec_code: str,
    career_database: Dict[str, Any],
    user_preferences: Dict[str, Any] = None,
    comparison_aspects: List[str] = None
) -> Dict[str, Any]:
    """Convenience function to compare careers.
    
    Args:
        career_ids: List of career IDs to compare
        user_raisec_code: User's RAISEC code
        career_database: Career database
        user_preferences: User preferences
        comparison_aspects: Aspects to compare
        
    Returns:
        Career comparison results
    """
    recommender = CareerRecommender()
    
    context = RecommendationContext(
        user_raisec_code=user_raisec_code,
        user_raisec_scores={},  # Not needed for comparison
        user_preferences=user_preferences or {},
        career_database=career_database
    )
    
    return await recommender.generate_career_comparison(
        career_ids, context, comparison_aspects
    )