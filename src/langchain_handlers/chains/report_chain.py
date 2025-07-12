"""Report generation chain for RAISEC assessment results.

This module provides LangChain-based report generation that creates
comprehensive, personalized reports from assessment results.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

from src.llm.base_llm import BaseLLM, LLMRequest, LLMMessage, LLMRole
from src.llm.llm_factory import LLMFactory
from src.utils.constants import RaisecDimension, AgeGroup, ReportType
from src.utils.logger import get_logger
from ..prompts.report_prompts import ReportPrompts
from ..parsers.report_parser import ReportParser

logger = get_logger(__name__)


class ReportChain(Chain):
    """LangChain for generating comprehensive RAISEC assessment reports.
    
    This chain creates detailed, personalized reports that include:
    - RAISEC profile analysis
    - Career recommendations
    - Development suggestions
    - Next steps guidance
    """
    
    llm: BaseLLM
    prompt_template: BasePromptTemplate
    output_parser: BaseOutputParser
    report_type: ReportType
    verbose: bool = False
    
    # Chain configuration
    input_keys: List[str] = [
        "assessment_results",
        "user_profile",
        "career_recommendations",
        "additional_data"
    ]
    output_keys: List[str] = [
        "report_content",
        "report_metadata"
    ]
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        report_type: ReportType = ReportType.COMPREHENSIVE,
        **kwargs
    ):
        """Initialize report generation chain.
        
        Args:
            llm: Language model to use (creates default if None)
            report_type: Type of report to generate
            **kwargs: Additional chain arguments
        """
        # Set up LLM
        if llm is None:
            llm = LLMFactory.create_from_settings()
        
        # Get appropriate prompt template and parser
        prompt_template = ReportPrompts.get_template(report_type)
        output_parser = ReportParser(report_type=report_type)
        
        super().__init__(
            llm=llm,
            prompt_template=prompt_template,
            output_parser=output_parser,
            report_type=report_type,
            **kwargs
        )
    
    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "report_generation"
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        required_keys = {"assessment_results", "user_profile"}
        missing_keys = required_keys - set(inputs.keys())
        
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        
        # Validate assessment results
        assessment_results = inputs.get("assessment_results", {})
        if not isinstance(assessment_results, dict):
            raise ValueError("assessment_results must be a dictionary")
        
        required_assessment_keys = {"raisec_scores", "raisec_code", "total_score"}
        missing_assessment_keys = required_assessment_keys - set(assessment_results.keys())
        if missing_assessment_keys:
            raise ValueError(f"Missing assessment result keys: {missing_assessment_keys}")
        
        # Validate user profile
        user_profile = inputs.get("user_profile", {})
        if not isinstance(user_profile, dict):
            raise ValueError("user_profile must be a dictionary")
    
    def _prepare_llm_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the language model.
        
        Args:
            inputs: Raw input dictionary
            
        Returns:
            Dict[str, Any]: Formatted input for prompt template
        """
        assessment_results = inputs["assessment_results"]
        user_profile = inputs["user_profile"]
        career_recommendations = inputs.get("career_recommendations", [])
        
        # Process RAISEC scores
        raisec_scores = assessment_results["raisec_scores"]
        sorted_dimensions = sorted(
            raisec_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format dimension analysis
        dimension_analysis = []
        for i, (dim_code, score) in enumerate(sorted_dimensions):
            dimension = RaisecDimension(dim_code)
            normalized_score = min(100, max(0, score))
            
            dimension_analysis.append({
                "rank": i + 1,
                "code": dimension.value,
                "name": dimension.name_full,
                "description": dimension.description,
                "score": score,
                "normalized_score": normalized_score,
                "percentage": normalized_score,
                "is_dominant": i < 3,  # Top 3 are dominant
                "strength_level": self._get_strength_level(normalized_score),
                "keywords": []  # Could be enhanced with keywords
            })
        
        # Analyze score patterns
        score_analysis = self._analyze_score_patterns(sorted_dimensions)
        
        # Extract user demographics
        age = user_profile.get("age", 25)
        age_group = AgeGroup.from_age(age) if isinstance(age, int) else AgeGroup.YOUNG_ADULT
        
        # Prepare career recommendations summary
        career_summary = self._summarize_career_recommendations(career_recommendations)
        
        # Calculate completion and confidence metrics
        completion_metrics = self._calculate_completion_metrics(assessment_results)
        
        return {
            "report_type": self.report_type.value,
            "raisec_code": assessment_results["raisec_code"],
            "dominant_code": assessment_results["raisec_code"][:3],
            "primary_dimension": dimension_analysis[0] if dimension_analysis else None,
            "secondary_dimension": dimension_analysis[1] if len(dimension_analysis) > 1 else None,
            "tertiary_dimension": dimension_analysis[2] if len(dimension_analysis) > 2 else None,
            "all_dimensions": dimension_analysis,
            "dimension_count": len(dimension_analysis),
            "score_analysis": score_analysis,
            "total_assessment_score": assessment_results.get("total_score", 0),
            "user_name": user_profile.get("name", "User"),
            "user_age": age,
            "age_group": age_group.value,
            "age_range": f"{age_group.get_age_range()[0]}-{age_group.get_age_range()[1]}",
            "user_location": user_profile.get("location", "India"),
            "education_level": user_profile.get("education_level", ""),
            "experience_level": user_profile.get("experience_level", ""),
            "career_stage": self._determine_career_stage(age, user_profile),
            "career_recommendations": career_summary,
            "completion_metrics": completion_metrics,
            "assessment_date": assessment_results.get("completed_at", datetime.utcnow().isoformat()),
            "report_generation_date": datetime.utcnow().isoformat(),
            "additional_insights": inputs.get("additional_data", {}),
        }
    
    def _get_strength_level(self, score: float) -> str:
        """Determine strength level based on score.
        
        Args:
            score: Normalized score (0-100)
            
        Returns:
            str: Strength level description
        """
        if score >= 80:
            return "Very Strong"
        elif score >= 60:
            return "Strong"
        elif score >= 40:
            return "Moderate"
        elif score >= 20:
            return "Developing"
        else:
            return "Limited"
    
    def _analyze_score_patterns(self, sorted_dimensions: List[tuple]) -> Dict[str, Any]:
        """Analyze patterns in RAISEC scores.
        
        Args:
            sorted_dimensions: List of (dimension, score) tuples sorted by score
            
        Returns:
            Dict[str, Any]: Score pattern analysis
        """
        if not sorted_dimensions:
            return {}
        
        scores = [score for _, score in sorted_dimensions]
        
        highest_score = scores[0]
        lowest_score = scores[-1]
        score_range = highest_score - lowest_score
        mean_score = sum(scores) / len(scores)
        
        # Determine profile type
        if score_range < 20:
            profile_type = "Balanced"
            profile_description = "You show relatively balanced interests across multiple areas."
        elif score_range < 40:
            profile_type = "Moderately Focused"
            profile_description = "You have some clear preferences with moderate variation."
        else:
            profile_type = "Highly Focused"
            profile_description = "You have very distinct preferences and clear areas of strength."
        
        # Calculate score distribution
        high_scores = len([s for s in scores if s >= mean_score + 10])
        low_scores = len([s for s in scores if s <= mean_score - 10])
        
        return {
            "profile_type": profile_type,
            "profile_description": profile_description,
            "score_range": score_range,
            "mean_score": mean_score,
            "highest_score": highest_score,
            "lowest_score": lowest_score,
            "high_scoring_dimensions": high_scores,
            "low_scoring_dimensions": low_scores,
            "is_balanced": score_range < 20,
            "is_focused": score_range > 40,
        }
    
    def _determine_career_stage(self, age: int, profile: Dict[str, Any]) -> str:
        """Determine career stage based on age and profile.
        
        Args:
            age: User's age
            profile: User profile data
            
        Returns:
            str: Career stage description
        """
        experience = profile.get("experience_level", "").lower()
        
        if age < 18:
            return "Career Exploration"
        elif age <= 22:
            return "Career Entry"
        elif age <= 28:
            if "senior" in experience or "lead" in experience:
                return "Career Advancement"
            return "Early Career Development"
        elif age <= 35:
            if "manager" in experience or "director" in experience:
                return "Leadership Development"
            return "Career Advancement"
        else:
            return "Senior Career Leadership"
    
    def _summarize_career_recommendations(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize career recommendations for report inclusion.
        
        Args:
            recommendations: List of career recommendation dictionaries
            
        Returns:
            Dict[str, Any]: Summary of recommendations
        """
        if not recommendations:
            return {
                "total_recommendations": 0,
                "top_recommendations": [],
                "recommendation_categories": [],
            }
        
        # Extract top recommendations
        top_recommendations = recommendations[:5]  # Top 5
        
        # Categorize recommendations
        categories = {}
        for rec in recommendations:
            category = rec.get("category", "General")
            if category not in categories:
                categories[category] = []
            categories[category].append(rec)
        
        return {
            "total_recommendations": len(recommendations),
            "top_recommendations": [
                {
                    "title": rec.get("title", ""),
                    "category": rec.get("category", ""),
                    "match_score": rec.get("match_score", 0),
                    "description": rec.get("description", "")[:200] + "..." if len(rec.get("description", "")) > 200 else rec.get("description", "")
                }
                for rec in top_recommendations
            ],
            "recommendation_categories": list(categories.keys()),
            "category_counts": {cat: len(recs) for cat, recs in categories.items()},
        }
    
    def _calculate_completion_metrics(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate assessment completion metrics.
        
        Args:
            assessment_results: Assessment result data
            
        Returns:
            Dict[str, Any]: Completion metrics
        """
        return {
            "questions_answered": assessment_results.get("questions_answered", 0),
            "total_questions": assessment_results.get("total_questions", 12),
            "completion_percentage": (assessment_results.get("questions_answered", 0) / max(1, assessment_results.get("total_questions", 12))) * 100,
            "time_spent_minutes": assessment_results.get("time_spent_seconds", 0) / 60,
            "confidence_score": assessment_results.get("confidence_score", 85),
            "quality_score": assessment_results.get("quality_score", 90),
        }
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the report generation chain synchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Callback manager for the run
            
        Returns:
            Dict[str, Any]: Generated report content and metadata
        """
        # Run async version in sync context
        return asyncio.run(self._acall(inputs, run_manager))
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the report generation chain asynchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Async callback manager for the run
            
        Returns:
            Dict[str, Any]: Generated report content and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate inputs
            self._validate_inputs(inputs)
            
            # Prepare input for LLM
            formatted_input = self._prepare_llm_input(inputs)
            
            if run_manager:
                run_manager.on_text(f"Generating {self.report_type.value} report for RAISEC: {formatted_input['raisec_code']}")
            
            # Format prompt
            prompt_messages = self.prompt_template.format_messages(**formatted_input)
            
            # Convert LangChain messages to our LLM format
            llm_messages = []
            for msg in prompt_messages:
                if isinstance(msg, SystemMessage):
                    role = LLMRole.SYSTEM
                elif isinstance(msg, HumanMessage):
                    role = LLMRole.USER
                else:
                    role = LLMRole.USER  # Default fallback
                
                llm_messages.append(LLMMessage(role=role, content=msg.content))
            
            # Create LLM request with higher token limit for reports
            llm_request = LLMRequest(
                messages=llm_messages,
                model=self.llm.model,
                max_tokens=4000,  # Higher limit for comprehensive reports
                temperature=0.5,  # Balanced creativity and consistency
                metadata={
                    "report_type": self.report_type.value,
                    "raisec_code": formatted_input["raisec_code"],
                    "user_age": formatted_input.get("user_age")
                }
            )
            
            # Generate response
            response = await self.llm.generate(llm_request)
            
            if run_manager:
                run_manager.on_text(f"Generated response: {len(response.content)} characters")
            
            # Parse the response
            parsed_result = await self.output_parser.aparse(response.content)
            
            # Calculate generation time
            generation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Prepare output
            result = {
                "report_content": parsed_result,
                "report_metadata": {
                    "report_type": self.report_type.value,
                    "raisec_code": formatted_input["raisec_code"],
                    "user_name": formatted_input.get("user_name"),
                    "user_age": formatted_input.get("user_age"),
                    "career_stage": formatted_input.get("career_stage"),
                    "generation_time_seconds": generation_time,
                    "model_used": self.llm.model.value,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "cost_estimated": response.usage.estimated_cost if response.usage else 0.0,
                    "generated_at": start_time.isoformat(),
                    "parser_version": self.output_parser.get_version(),
                    "word_count": len(response.content.split()) if response.content else 0,
                    "character_count": len(response.content) if response.content else 0,
                }
            }
            
            logger.info(
                f"Generated {self.report_type.value} report for RAISEC {formatted_input['raisec_code']} "
                f"in {generation_time:.2f}s ({len(response.content)} chars)"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate report: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return error result
            return {
                "report_content": None,
                "report_metadata": {
                    "error": error_msg,
                    "report_type": self.report_type.value,
                    "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                    "generated_at": start_time.isoformat(),
                }
            }
    
    def generate_report(
        self,
        assessment_results: Dict[str, Any],
        user_profile: Dict[str, Any],
        career_recommendations: Optional[List[Dict[str, Any]]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive report for assessment results.
        
        Args:
            assessment_results: Complete assessment results including scores
            user_profile: User demographic and profile information
            career_recommendations: Optional career recommendations
            additional_data: Optional additional insights or data
            
        Returns:
            Dict[str, Any]: Generated report content and metadata
        """
        inputs = {
            "assessment_results": assessment_results,
            "user_profile": user_profile,
            "career_recommendations": career_recommendations or [],
            "additional_data": additional_data or {}
        }
        
        return self(inputs)
    
    async def agenerate_report(
        self,
        assessment_results: Dict[str, Any],
        user_profile: Dict[str, Any],
        career_recommendations: Optional[List[Dict[str, Any]]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously generate a comprehensive report for assessment results.
        
        Args:
            assessment_results: Complete assessment results including scores
            user_profile: User demographic and profile information
            career_recommendations: Optional career recommendations
            additional_data: Optional additional insights or data
            
        Returns:
            Dict[str, Any]: Generated report content and metadata
        """
        inputs = {
            "assessment_results": assessment_results,
            "user_profile": user_profile,
            "career_recommendations": career_recommendations or [],
            "additional_data": additional_data or {}
        }
        
        return await self.acall(inputs)
    
    def get_supported_report_types(self) -> List[ReportType]:
        """Get list of supported report types.
        
        Returns:
            List[ReportType]: Supported report types
        """
        return [
            ReportType.COMPREHENSIVE,
            ReportType.SUMMARY,
            ReportType.DETAILED,
        ]
    
    @classmethod
    def create_for_type(
        cls,
        report_type: ReportType,
        llm: Optional[BaseLLM] = None
    ) -> "ReportChain":
        """Create a chain for specific report type.
        
        Args:
            report_type: Type of report to generate
            llm: Optional LLM instance
            
        Returns:
            ReportChain: Configured chain instance
        """
        return cls(
            llm=llm,
            report_type=report_type,
            verbose=True
        )


# Export the chain
__all__ = ["ReportChain"]