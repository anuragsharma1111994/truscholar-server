"""Enhanced career recommendation chain for RAISEC assessment results.

This module provides sophisticated LangChain-based career recommendations with
advanced processing capabilities, multi-step analysis, and comprehensive
career matching algorithms based on user RAISEC profiles.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from langchain.chains.base import Chain
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, BaseOutputParser, OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser as CoreBaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.load import serializable

from src.llm.base_llm import BaseLLM, LLMRequest, LLMMessage, LLMRole
from src.llm.llm_factory import LLMFactory
from src.utils.constants import RaisecDimension, AgeGroup, RecommendationType
from src.utils.logger import get_logger
from src.utils.exceptions import TruScholarError
from ..prompts.career_prompts import CareerPrompts
from ..parsers.career_parser import CareerParser

logger = get_logger(__name__)


class ChainType(Enum):
    """Types of career recommendation chains."""
    BASIC_RECOMMENDATION = "basic_recommendation"
    ADVANCED_ANALYSIS = "advanced_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    NARRATIVE_GENERATION = "narrative_generation"
    SKILL_ASSESSMENT = "skill_assessment"
    MARKET_ANALYSIS = "market_analysis"


class CareerChainError(TruScholarError):
    """Exception raised when career chain processing fails."""
    pass


@dataclass
class ChainConfig:
    """Configuration for career chains."""
    temperature: float = 0.7
    max_tokens: int = 2000
    enable_memory: bool = True
    memory_type: str = "buffer"
    enable_caching: bool = True
    max_retries: int = 3
    timeout_seconds: int = 60


class EnhancedCareerOutputParser(BaseOutputParser):
    """Enhanced output parser for career recommendation responses."""
    
    def __init__(self, output_format: str = "json", **kwargs):
        super().__init__(**kwargs)
        self.output_format = output_format
        self.fallback_enabled = True
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured career data."""
        try:
            cleaned_text = self._clean_output_text(text)
            
            if self.output_format == "json":
                return self._parse_json_output(cleaned_text)
            elif self.output_format == "structured":
                return self._parse_structured_output(cleaned_text)
            else:
                return {"raw_output": cleaned_text}
                
        except Exception as e:
            if self.fallback_enabled:
                logger.warning(f"Primary parsing failed, using fallback: {str(e)}")
                return self._fallback_parse(text)
            else:
                raise OutputParserException(f"Failed to parse career output: {str(e)}")
    
    def _clean_output_text(self, text: str) -> str:
        """Clean and normalize output text."""
        text = text.strip()
        text = text.replace("```json", "").replace("```", "")
        text = text.replace("Here's", "").replace("Here is", "")
        
        # Find JSON-like content
        start_markers = ["{", "["]
        end_markers = ["}", "]"]
        
        for start_marker in start_markers:
            start_idx = text.find(start_marker)
            if start_idx != -1:
                for end_marker in end_markers:
                    end_idx = text.rfind(end_marker)
                    if end_idx != -1 and end_idx > start_idx:
                        return text[start_idx:end_idx + 1]
        
        return text
    
    def _parse_json_output(self, text: str) -> Dict[str, Any]:
        """Parse JSON formatted output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            fixed_text = self._fix_json_issues(text)
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                raise OutputParserException(f"Invalid JSON output: {str(e)}")
    
    def _parse_structured_output(self, text: str) -> Dict[str, Any]:
        """Parse structured text output."""
        result = {"sections": []}
        current_section = None
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.endswith(':') or line.startswith('#'):
                if current_section:
                    result["sections"].append(current_section)
                current_section = {
                    "title": line.rstrip(':').lstrip('#').strip(),
                    "content": []
                }
            elif current_section:
                current_section["content"].append(line)
            else:
                # Content without section
                if "general" not in result:
                    result["general"] = []
                result["general"].append(line)
        
        if current_section:
            result["sections"].append(current_section)
        
        return result
    
    def _fix_json_issues(self, text: str) -> str:
        """Attempt to fix common JSON formatting issues."""
        # Fix trailing commas
        text = text.replace(",}", "}").replace(",]", "]")
        
        # Fix missing quotes on keys
        import re
        text = re.sub(r'(\w+):', r'"\1":', text)
        
        # Fix single quotes
        text = text.replace("'", '"')
        
        return text
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Fallback parser for when primary parsing fails."""
        return {
            "raw_output": text,
            "parsed": False,
            "fallback_used": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @property
    def _type(self) -> str:
        return "enhanced_career_output_parser"


class CareerChain(Chain):
    """Enhanced LangChain for generating sophisticated career recommendations.
    
    This chain provides multi-step analysis, advanced matching algorithms,
    and comprehensive career recommendations based on RAISEC profiles with
    support for various analysis types and processing modes.
    """
    
    llm: BaseLLM
    prompt_template: BasePromptTemplate
    output_parser: Union[BaseOutputParser, EnhancedCareerOutputParser]
    recommendation_type: RecommendationType
    chain_type: ChainType
    config: ChainConfig
    memory: Optional[ConversationBufferMemory]
    verbose: bool = False
    
    # Enhanced chain configuration
    input_keys: List[str] = [
        "raisec_scores",
        "raisec_code", 
        "user_profile",
        "interests",
        "constraints",
        "career_database",
        "analysis_context"
    ]
    output_keys: List[str] = [
        "career_recommendations",
        "analysis_metadata",
        "processing_stats",
        "confidence_scores"
    ]
    
    # Performance tracking
    execution_stats: Dict[str, Any]
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        recommendation_type: RecommendationType = RecommendationType.TRADITIONAL,
        chain_type: ChainType = ChainType.BASIC_RECOMMENDATION,
        config: Optional[ChainConfig] = None,
        enable_enhanced_parsing: bool = True,
        **kwargs
    ):
        """Initialize enhanced career recommendation chain.
        
        Args:
            llm: Language model to use (creates default if None)
            recommendation_type: Type of recommendations to generate
            chain_type: Type of chain processing to use
            config: Chain configuration settings
            enable_enhanced_parsing: Use enhanced output parser
            **kwargs: Additional chain arguments
        """
        # Set up LLM
        if llm is None:
            llm = LLMFactory.create_from_settings()
        
        # Set up configuration
        if config is None:
            config = ChainConfig()
        
        # Get appropriate prompt template and parser
        prompt_template = CareerPrompts.get_template(recommendation_type)
        
        if enable_enhanced_parsing:
            output_parser = EnhancedCareerOutputParser(output_format="json")
        else:
            output_parser = CareerParser(recommendation_type=recommendation_type)
        
        # Initialize memory if enabled
        memory = None
        if config.enable_memory:
            if config.memory_type == "summary":
                memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=1000,
                    return_messages=True
                )
            else:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
        
        # Initialize execution stats
        execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "cache_hits": 0,
            "retry_count": 0
        }
        
        super().__init__(
            llm=llm,
            prompt_template=prompt_template,
            output_parser=output_parser,
            recommendation_type=recommendation_type,
            chain_type=chain_type,
            config=config,
            memory=memory,
            execution_stats=execution_stats,
            **kwargs
        )
    
    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return f"enhanced_career_recommendation_{self.chain_type.value}"
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """Get comprehensive chain execution statistics."""
        success_rate = 0.0
        if self.execution_stats["total_executions"] > 0:
            success_rate = (
                self.execution_stats["successful_executions"] / 
                self.execution_stats["total_executions"]
            )
        
        cache_hit_rate = 0.0
        if self.execution_stats["total_executions"] > 0:
            cache_hit_rate = (
                self.execution_stats["cache_hits"] / 
                self.execution_stats["total_executions"]
            )
        
        return {
            "chain_type": self.chain_type.value,
            "recommendation_type": self.recommendation_type.value,
            "total_executions": self.execution_stats["total_executions"],
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "average_execution_time": self.execution_stats["average_execution_time"],
            "retry_count": self.execution_stats["retry_count"],
            "memory_enabled": self.config.enable_memory,
            "enhanced_parsing": isinstance(self.output_parser, EnhancedCareerOutputParser)
        }
    
    def reset_chain_memory(self) -> None:
        """Reset conversation memory if enabled."""
        if self.memory:
            self.memory.clear()
            logger.info("Career chain memory reset")
    
    def _update_execution_stats(self, success: bool, execution_time: float, used_cache: bool = False) -> None:
        """Update execution statistics."""
        self.execution_stats["total_executions"] += 1
        
        if success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        if used_cache:
            self.execution_stats["cache_hits"] += 1
        
        # Update average execution time
        current_avg = self.execution_stats["average_execution_time"]
        total_count = self.execution_stats["total_executions"]
        new_avg = ((current_avg * (total_count - 1)) + execution_time) / total_count
        self.execution_stats["average_execution_time"] = new_avg
    
    def _calculate_confidence_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for the recommendations."""
        confidence_scores = {
            "overall_confidence": 0.8,
            "data_quality": 0.9,
            "model_certainty": 0.85,
            "recommendation_strength": 0.8
        }
        
        # Adjust based on result quality
        recommendations = result.get("recommendations", [])
        if recommendations:
            if len(recommendations) >= 5:
                confidence_scores["recommendation_strength"] += 0.1
            
            # Check for fallback usage
            if any(rec.get("fallback_used", False) for rec in recommendations):
                confidence_scores["model_certainty"] -= 0.2
        
        # Normalize scores
        for key in confidence_scores:
            confidence_scores[key] = max(0.0, min(1.0, confidence_scores[key]))
        
        return confidence_scores
    
    def _build_advanced_prompt_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Build advanced context for enhanced prompting."""
        context = self._prepare_llm_input(inputs)
        
        # Add advanced analysis context
        career_database = inputs.get("career_database", {})
        analysis_context = inputs.get("analysis_context", {})
        
        # Calculate additional metrics
        raisec_scores = inputs["raisec_scores"]
        score_variance = self._calculate_score_variance(raisec_scores)
        dominant_dimensions = self._identify_dominant_dimensions(raisec_scores)
        
        context.update({
            "score_variance": score_variance,
            "dominant_dimensions": dominant_dimensions,
            "profile_clarity": self._assess_profile_clarity(raisec_scores),
            "career_database_size": len(career_database),
            "analysis_depth": analysis_context.get("depth", "standard"),
            "include_market_data": analysis_context.get("include_market_data", True),
            "focus_areas": analysis_context.get("focus_areas", []),
            "chain_type": self.chain_type.value
        })
        
        return context
    
    def _calculate_score_variance(self, scores: Dict[str, float]) -> float:
        """Calculate variance in RAISEC scores."""
        if not scores:
            return 0.0
        
        values = list(scores.values())
        mean_score = sum(values) / len(values)
        variance = sum((score - mean_score) ** 2 for score in values) / len(values)
        return variance
    
    def _identify_dominant_dimensions(self, scores: Dict[str, float], threshold: float = 10.0) -> List[str]:
        """Identify dominant RAISEC dimensions."""
        if not scores:
            return []
        
        max_score = max(scores.values())
        dominant = [dim for dim, score in scores.items() if score >= (max_score - threshold)]
        return dominant[:3]  # Limit to top 3
    
    def _assess_profile_clarity(self, scores: Dict[str, float]) -> str:
        """Assess clarity of RAISEC profile."""
        if not scores:
            return "unclear"
        
        variance = self._calculate_score_variance(scores)
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if variance > 400:  # High variance
            return "very_clear"
        elif variance > 200:
            return "clear"
        elif variance > 100:
            return "moderate"
        else:
            return "unclear"
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        required_keys = {"raisec_scores", "raisec_code", "user_profile"}
        missing_keys = required_keys - set(inputs.keys())
        
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        
        # Validate RAISEC scores
        raisec_scores = inputs.get("raisec_scores", {})
        if not isinstance(raisec_scores, dict):
            raise ValueError("raisec_scores must be a dictionary")
        
        valid_dimensions = [dim.value for dim in RaisecDimension]
        for dim_code, score in raisec_scores.items():
            if dim_code not in valid_dimensions:
                raise ValueError(f"Invalid dimension code: {dim_code}")
            if not isinstance(score, (int, float)) or score < 0:
                raise ValueError(f"Invalid score for {dim_code}: {score}")
        
        # Validate RAISEC code
        raisec_code = inputs.get("raisec_code", "")
        if not isinstance(raisec_code, str) or len(raisec_code) != 3:
            raise ValueError("raisec_code must be a 3-character string")
        
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
        raisec_scores = inputs["raisec_scores"]
        user_profile = inputs["user_profile"]
        
        # Sort dimensions by score
        sorted_dimensions = sorted(
            raisec_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format dimension analysis
        dimension_analysis = []
        for dim_code, score in sorted_dimensions:
            dimension = RaisecDimension(dim_code)
            dimension_analysis.append({
                "code": dimension.value,
                "name": dimension.name_full,
                "description": dimension.description,
                "score": score,
                "normalized_score": min(100, max(0, score)),  # Ensure 0-100 range
                "keywords": []  # Could be enhanced with keywords
            })
        
        # Extract user demographics
        age = user_profile.get("age", 25)
        age_group = AgeGroup.from_age(age) if isinstance(age, int) else AgeGroup.YOUNG_ADULT
        
        return {
            "raisec_code": inputs["raisec_code"],
            "top_three_dimensions": dimension_analysis[:3],
            "all_dimensions": dimension_analysis,
            "total_dimensions": len(dimension_analysis),
            "highest_score": dimension_analysis[0]["score"] if dimension_analysis else 0,
            "lowest_score": dimension_analysis[-1]["score"] if dimension_analysis else 0,
            "score_spread": (dimension_analysis[0]["score"] - dimension_analysis[-1]["score"]) if len(dimension_analysis) > 1 else 0,
            "user_age": age,
            "age_group": age_group.value,
            "age_range": f"{age_group.get_age_range()[0]}-{age_group.get_age_range()[1]}",
            "user_location": user_profile.get("location", "India"),
            "education_level": user_profile.get("education_level", ""),
            "experience_level": user_profile.get("experience_level", ""),
            "interests": inputs.get("interests", []),
            "career_stage": self._determine_career_stage(age, user_profile),
            "recommendation_type": self.recommendation_type.value,
            "constraints": inputs.get("constraints", {}),
            "current_timestamp": datetime.utcnow().isoformat(),
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
            return "pre_career"
        elif age <= 22:
            return "entry_level"
        elif age <= 28:
            if "senior" in experience or "lead" in experience:
                return "mid_career"
            return "early_career"
        elif age <= 35:
            if "manager" in experience or "director" in experience:
                return "senior_career"
            return "mid_career"
        else:
            return "senior_career"
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the career recommendation chain synchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Callback manager for the run
            
        Returns:
            Dict[str, Any]: Career recommendations and analysis metadata
        """
        # Run async version in sync context
        return asyncio.run(self._acall(inputs, run_manager))
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the career recommendation chain asynchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Async callback manager for the run
            
        Returns:
            Dict[str, Any]: Career recommendations and analysis metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate inputs
            self._validate_inputs(inputs)
            
            # Prepare input for LLM
            formatted_input = self._prepare_llm_input(inputs)
            
            if run_manager:
                run_manager.on_text(f"Generating {self.recommendation_type.value} career recommendations for RAISEC: {inputs['raisec_code']}")
            
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
            
            # Create LLM request
            llm_request = LLMRequest(
                messages=llm_messages,
                model=self.llm.model,
                max_tokens=3000,
                temperature=0.6,  # Slightly more creative for recommendations
                metadata={
                    "recommendation_type": self.recommendation_type.value,
                    "raisec_code": inputs["raisec_code"],
                    "age_group": formatted_input.get("age_group")
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
                "career_recommendations": parsed_result,
                "analysis_metadata": {
                    "recommendation_type": self.recommendation_type.value,
                    "raisec_code": inputs["raisec_code"],
                    "raisec_scores": inputs["raisec_scores"],
                    "user_age": formatted_input.get("user_age"),
                    "age_group": formatted_input.get("age_group"),
                    "career_stage": formatted_input.get("career_stage"),
                    "generation_time_seconds": generation_time,
                    "model_used": self.llm.model.value,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "cost_estimated": response.usage.estimated_cost if response.usage else 0.0,
                    "generated_at": start_time.isoformat(),
                    "parser_version": self.output_parser.get_version(),
                }
            }
            
            logger.info(
                f"Generated {self.recommendation_type.value} career recommendations "
                f"for RAISEC {inputs['raisec_code']} in {generation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate career recommendations: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return error result
            return {
                "career_recommendations": None,
                "analysis_metadata": {
                    "error": error_msg,
                    "recommendation_type": self.recommendation_type.value,
                    "raisec_code": inputs.get("raisec_code", ""),
                    "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                    "generated_at": start_time.isoformat(),
                }
            }
    
    def generate_recommendations(
        self,
        raisec_scores: Dict[str, float],
        raisec_code: str,
        user_profile: Dict[str, Any],
        interests: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate career recommendations for a user.
        
        Args:
            raisec_scores: Dictionary of RAISEC dimension scores
            raisec_code: 3-letter RAISEC code
            user_profile: User demographic and profile information
            interests: Optional list of user interests
            constraints: Optional constraints for recommendations
            
        Returns:
            Dict[str, Any]: Career recommendations and metadata
        """
        inputs = {
            "raisec_scores": raisec_scores,
            "raisec_code": raisec_code,
            "user_profile": user_profile,
            "interests": interests or [],
            "constraints": constraints or {}
        }
        
        return self(inputs)
    
    async def agenerate_recommendations(
        self,
        raisec_scores: Dict[str, float],
        raisec_code: str,
        user_profile: Dict[str, Any],
        interests: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously generate career recommendations for a user.
        
        Args:
            raisec_scores: Dictionary of RAISEC dimension scores
            raisec_code: 3-letter RAISEC code
            user_profile: User demographic and profile information
            interests: Optional list of user interests
            constraints: Optional constraints for recommendations
            
        Returns:
            Dict[str, Any]: Career recommendations and metadata
        """
        inputs = {
            "raisec_scores": raisec_scores,
            "raisec_code": raisec_code,
            "user_profile": user_profile,
            "interests": interests or [],
            "constraints": constraints or {}
        }
        
        return await self.acall(inputs)
    
    def compare_career_fit(
        self,
        raisec_scores: Dict[str, float],
        career_profiles: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Compare user RAISEC profile against career profiles.
        
        Args:
            raisec_scores: User's RAISEC scores
            career_profiles: List of career profile dictionaries
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Careers with fit scores
        """
        career_fits = []
        
        for career in career_profiles:
            career_raisec = career.get("raisec_profile", {})
            
            # Calculate fit score based on correlation
            fit_score = self._calculate_career_fit(raisec_scores, career_raisec)
            career_fits.append((career, fit_score))
        
        # Sort by fit score (descending)
        career_fits.sort(key=lambda x: x[1], reverse=True)
        
        return career_fits
    
    def _calculate_career_fit(
        self,
        user_scores: Dict[str, float],
        career_profile: Dict[str, float]
    ) -> float:
        """Calculate fit score between user and career profile.
        
        Args:
            user_scores: User's RAISEC scores
            career_profile: Career's RAISEC profile
            
        Returns:
            float: Fit score (0-100)
        """
        if not career_profile:
            return 0.0
        
        # Normalize scores to 0-100 range
        user_normalized = {k: min(100, max(0, v)) for k, v in user_scores.items()}
        career_normalized = {k: min(100, max(0, v)) for k, v in career_profile.items()}
        
        # Calculate correlation coefficient
        dimensions = set(user_normalized.keys()) & set(career_normalized.keys())
        
        if len(dimensions) < 3:  # Need at least 3 dimensions for meaningful comparison
            return 0.0
        
        user_values = [user_normalized[dim] for dim in dimensions]
        career_values = [career_normalized[dim] for dim in dimensions]
        
        # Simple correlation calculation
        from statistics import mean
        
        user_mean = mean(user_values)
        career_mean = mean(career_values)
        
        numerator = sum((u - user_mean) * (c - career_mean) for u, c in zip(user_values, career_values))
        user_variance = sum((u - user_mean) ** 2 for u in user_values)
        career_variance = sum((c - career_mean) ** 2 for c in career_values)
        
        if user_variance == 0 or career_variance == 0:
            return 0.0
        
        correlation = numerator / (user_variance * career_variance) ** 0.5
        
        # Convert correlation to 0-100 score
        return max(0, min(100, (correlation + 1) * 50))
    
    def get_supported_recommendation_types(self) -> List[RecommendationType]:
        """Get list of supported recommendation types.
        
        Returns:
            List[RecommendationType]: Supported recommendation types
        """
        return [
            RecommendationType.TRADITIONAL,
            RecommendationType.INNOVATIVE,
            RecommendationType.HYBRID,
        ]
    
    @classmethod
    def create_for_type(
        cls,
        recommendation_type: RecommendationType,
        llm: Optional[BaseLLM] = None
    ) -> "CareerChain":
        """Create a chain for specific recommendation type.
        
        Args:
            recommendation_type: Type of recommendations to generate
            llm: Optional LLM instance
            
        Returns:
            CareerChain: Configured chain instance
        """
        return cls(
            llm=llm,
            recommendation_type=recommendation_type,
            verbose=True
        )


# Export the chain
__all__ = ["CareerChain"]