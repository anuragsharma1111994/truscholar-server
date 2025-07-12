"""LangChain-based question generator for RAISEC assessment.

This module provides intelligent question generation using LangChain
with structured output parsing and validation.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnablePassthrough

from src.langchain_handlers.chains.question_chain import QuestionChain
from src.langchain_handlers.prompts.question_prompts import QuestionPrompts
from src.langchain_handlers.parsers.question_parser import QuestionParser
from src.langchain_handlers.validation import ResponseValidator
from src.utils.constants import QuestionType, AgeGroup, RaisecDimension
from src.utils.logger import get_logger
from src.core.config import get_settings

settings = get_settings()
logger = get_logger(__name__)


class QuestionGenerationError(Exception):
    """Custom exception for question generation errors."""
    pass


class QuestionGenerator:
    """LangChain-based question generator for RAISEC assessments."""
    
    def __init__(self, enable_caching: bool = True, max_retries: int = 3):
        """Initialize the question generator.
        
        Args:
            enable_caching: Whether to enable response caching
            max_retries: Maximum number of generation retries
        """
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        self.validator = ResponseValidator()
        self._chains: Dict[Tuple[QuestionType, AgeGroup], QuestionChain] = {}
        
    async def generate_question(
        self,
        question_type: QuestionType,
        age_group: AgeGroup,
        question_number: int,
        dimensions_focus: Optional[List[RaisecDimension]] = None,
        context: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate a single question using LangChain.
        
        Args:
            question_type: Type of question to generate
            age_group: Target age group
            question_number: Question number in sequence
            dimensions_focus: RAISEC dimensions to focus on
            context: Additional context for generation
            constraints: Generation constraints
            
        Returns:
            Optional[Dict[str, Any]]: Generated question data or None if failed
            
        Raises:
            QuestionGenerationError: If generation fails after retries
        """
        logger.info(
            f"Generating {question_type.value} question for {age_group.value}",
            extra={
                "question_type": question_type.value,
                "age_group": age_group.value,
                "question_number": question_number
            }
        )
        
        # Prepare generation parameters
        generation_params = self._prepare_generation_params(
            question_type, age_group, question_number,
            dimensions_focus, context, constraints
        )
        
        # Try generation with retries
        for attempt in range(self.max_retries):
            try:
                result = await self._generate_with_retry(
                    question_type, age_group, generation_params, attempt + 1
                )
                
                if result:
                    # Add generation metadata
                    result["generation_metadata"] = {
                        "generator": "langchain",
                        "question_type": question_type.value,
                        "age_group": age_group.value,
                        "generation_attempt": attempt + 1,
                        "generated_at": datetime.utcnow().isoformat(),
                        "prompt_version": "1.0"
                    }
                    
                    logger.info(
                        f"Successfully generated question after {attempt + 1} attempts",
                        extra={"question_number": question_number, "attempts": attempt + 1}
                    )
                    
                    return result
                    
            except Exception as e:
                logger.warning(
                    f"Generation attempt {attempt + 1} failed: {str(e)}",
                    extra={
                        "attempt": attempt + 1,
                        "error": str(e),
                        "question_type": question_type.value
                    }
                )
                
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"All {self.max_retries} generation attempts failed",
                        extra={"question_type": question_type.value, "final_error": str(e)}
                    )
                    raise QuestionGenerationError(
                        f"Failed to generate {question_type.value} question after {self.max_retries} attempts: {str(e)}"
                    )
                
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    async def generate_batch_questions(
        self,
        question_configs: List[Dict[str, Any]],
        concurrent_limit: int = 5
    ) -> List[Optional[Dict[str, Any]]]:
        """Generate multiple questions concurrently.
        
        Args:
            question_configs: List of question configuration dictionaries
            concurrent_limit: Maximum concurrent generations
            
        Returns:
            List[Optional[Dict[str, Any]]]: Generated questions (same order as input)
        """
        logger.info(f"Starting batch generation of {len(question_configs)} questions")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def generate_single(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.generate_question(**config)
        
        # Execute all generations concurrently
        tasks = [generate_single(config) for config in question_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        successful = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Batch generation failed for question {i + 1}: {str(result)}",
                    extra={"question_index": i, "error": str(result)}
                )
                processed_results.append(None)
            else:
                processed_results.append(result)
                if result is not None:
                    successful += 1
        
        logger.info(
            f"Batch generation completed: {successful}/{len(question_configs)} successful",
            extra={"total": len(question_configs), "successful": successful}
        )
        
        return processed_results
    
    async def validate_generated_question(
        self,
        question_data: Dict[str, Any],
        question_type: QuestionType
    ) -> Tuple[bool, List[str]]:
        """Validate a generated question.
        
        Args:
            question_data: Generated question data
            question_type: Expected question type
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, errors)
        """
        try:
            # Convert to JSON string for validator
            question_json = json.dumps(question_data)
            return self.validator.validate_question_response(question_json, question_type)
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    async def get_generation_statistics(
        self,
        question_type: Optional[QuestionType] = None,
        age_group: Optional[AgeGroup] = None
    ) -> Dict[str, Any]:
        """Get generation statistics and performance metrics.
        
        Args:
            question_type: Filter by question type
            age_group: Filter by age group
            
        Returns:
            Dict[str, Any]: Generation statistics
        """
        # In a production system, this would query actual metrics
        # For now, return mock statistics
        return {
            "total_generated": 150,
            "success_rate": 0.95,
            "average_generation_time_seconds": 3.2,
            "by_question_type": {
                "mcq": {"generated": 45, "success_rate": 0.98},
                "statement_set": {"generated": 30, "success_rate": 0.93},
                "scenario_mcq": {"generated": 25, "success_rate": 0.96},
                "this_or_that": {"generated": 20, "success_rate": 0.99},
                "scale_rating": {"generated": 15, "success_rate": 0.91},
                "plot_day": {"generated": 10, "success_rate": 0.87},
                "scenario_multi_select": {"generated": 5, "success_rate": 0.89}
            },
            "by_age_group": {
                "13-17": {"generated": 50, "success_rate": 0.94},
                "18-25": {"generated": 55, "success_rate": 0.96},
                "26-35": {"generated": 45, "success_rate": 0.95}
            },
            "common_failure_reasons": [
                "Invalid JSON format",
                "Missing required fields",
                "Inappropriate content for age group",
                "Insufficient RAISEC dimension coverage"
            ]
        }
    
    def _prepare_generation_params(
        self,
        question_type: QuestionType,
        age_group: AgeGroup,
        question_number: int,
        dimensions_focus: Optional[List[RaisecDimension]],
        context: Optional[str],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare parameters for question generation.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            question_number: Question number
            dimensions_focus: RAISEC dimensions to focus on
            context: Additional context
            constraints: Generation constraints
            
        Returns:
            Dict[str, Any]: Prepared generation parameters
        """
        # Default dimensions focus if not provided
        if not dimensions_focus:
            dimensions_focus = self._get_default_dimensions_focus(question_type)
        
        # Default context if not provided
        if not context:
            context = self._get_default_context(question_type, age_group)
        
        # Default constraints if not provided
        if not constraints:
            constraints = self._get_default_constraints(question_type, age_group)
        
        return {
            "question_number": question_number,
            "dimensions_focus": [dim.value for dim in dimensions_focus],
            "context": context,
            "constraints": constraints,
            "age_group": f"{age_group.value[0]}-{age_group.value[1]}",
            "age_range": f"{age_group.value[0]}-{age_group.value[1]}"
        }
    
    async def _generate_with_retry(
        self,
        question_type: QuestionType,
        age_group: AgeGroup,
        generation_params: Dict[str, Any],
        attempt: int
    ) -> Optional[Dict[str, Any]]:
        """Generate question with single retry attempt.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            generation_params: Generation parameters
            attempt: Attempt number
            
        Returns:
            Optional[Dict[str, Any]]: Generated question data or None
        """
        try:
            # Get or create chain for this question type and age group
            chain = await self._get_chain(question_type, age_group)
            
            # Generate the question
            result = await chain.ainvoke(generation_params)
            
            # Validate the result
            if result and isinstance(result, dict):
                is_valid, errors = await self.validate_generated_question(result, question_type)
                
                if is_valid:
                    return result
                else:
                    logger.warning(
                        f"Generated question failed validation: {errors}",
                        extra={"attempt": attempt, "errors": errors}
                    )
                    return None
            else:
                logger.warning(
                    f"Invalid result format from chain: {type(result)}",
                    extra={"attempt": attempt, "result_type": type(result).__name__}
                )
                return None
                
        except OutputParserException as e:
            logger.warning(
                f"Parser error on attempt {attempt}: {str(e)}",
                extra={"attempt": attempt, "parser_error": str(e)}
            )
            return None
        
        except Exception as e:
            logger.error(
                f"Unexpected error on attempt {attempt}: {str(e)}",
                extra={"attempt": attempt, "error": str(e)}
            )
            raise
    
    async def _get_chain(
        self,
        question_type: QuestionType,
        age_group: AgeGroup
    ) -> QuestionChain:
        """Get or create a chain for the given question type and age group.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            
        Returns:
            QuestionChain: Chain for generation
        """
        chain_key = (question_type, age_group)
        
        if chain_key not in self._chains:
            self._chains[chain_key] = QuestionChain(
                question_type=question_type,
                age_group=age_group
            )
            logger.debug(f"Created new chain for {question_type.value}/{age_group.value}")
        
        return self._chains[chain_key]
    
    def _get_default_dimensions_focus(
        self,
        question_type: QuestionType
    ) -> List[RaisecDimension]:
        """Get default RAISEC dimensions focus for question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            List[RaisecDimension]: Default dimensions to focus on
        """
        # Different question types naturally evaluate different dimensions
        type_dimension_mapping = {
            QuestionType.MCQ: [RaisecDimension.REALISTIC, RaisecDimension.INVESTIGATIVE, RaisecDimension.ARTISTIC],
            QuestionType.STATEMENT_SET: [RaisecDimension.SOCIAL, RaisecDimension.ENTERPRISING, RaisecDimension.CONVENTIONAL],
            QuestionType.SCENARIO_MCQ: [RaisecDimension.ENTERPRISING, RaisecDimension.SOCIAL],
            QuestionType.SCENARIO_MULTI_SELECT: [RaisecDimension.INVESTIGATIVE, RaisecDimension.CONVENTIONAL],
            QuestionType.THIS_OR_THAT: [RaisecDimension.REALISTIC, RaisecDimension.ARTISTIC],
            QuestionType.SCALE_RATING: [RaisecDimension.SOCIAL, RaisecDimension.INVESTIGATIVE],
            QuestionType.PLOT_DAY: list(RaisecDimension)  # All dimensions for plot day
        }
        
        return type_dimension_mapping.get(question_type, [RaisecDimension.REALISTIC, RaisecDimension.SOCIAL])
    
    def _get_default_context(
        self,
        question_type: QuestionType,
        age_group: AgeGroup
    ) -> str:
        """Get default context for question generation.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            
        Returns:
            str: Default context
        """
        age_contexts = {
            AgeGroup.TEEN: "Focus on school activities, hobbies, and future interests rather than specific careers",
            AgeGroup.YOUNG_ADULT: "Consider college experiences, early career exploration, and skill development",
            AgeGroup.ADULT: "Focus on professional development, career advancement, and leadership opportunities"
        }
        
        type_contexts = {
            QuestionType.MCQ: "Create clear, distinct options that represent different personality types",
            QuestionType.STATEMENT_SET: "Use first-person statements about preferences and interests",
            QuestionType.SCENARIO_MCQ: "Present realistic workplace or academic scenarios",
            QuestionType.THIS_OR_THAT: "Create compelling choices between contrasting activities",
            QuestionType.SCALE_RATING: "Focus on intensity of preference or enjoyment",
            QuestionType.PLOT_DAY: "Include diverse activities covering all RAISEC dimensions"
        }
        
        return f"{age_contexts.get(age_group, '')}. {type_contexts.get(question_type, '')}"
    
    def _get_default_constraints(
        self,
        question_type: QuestionType,
        age_group: AgeGroup
    ) -> Dict[str, Any]:
        """Get default constraints for question generation.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            
        Returns:
            Dict[str, Any]: Default constraints
        """
        base_constraints = {
            "cultural_context": "Indian",
            "language_level": "accessible",
            "avoid_stereotypes": True,
            "include_modern_scenarios": True
        }
        
        # Age-specific constraints
        if age_group == AgeGroup.TEEN:
            base_constraints.update({
                "avoid_work_jargon": True,
                "focus_on_activities": True,
                "use_simple_language": True
            })
        elif age_group == AgeGroup.YOUNG_ADULT:
            base_constraints.update({
                "include_college_context": True,
                "balance_theory_practice": True
            })
        elif age_group == AgeGroup.ADULT:
            base_constraints.update({
                "include_leadership_scenarios": True,
                "consider_work_life_balance": True
            })
        
        # Question type specific constraints
        if question_type == QuestionType.MCQ:
            base_constraints.update({
                "option_count": 4,
                "balanced_options": True
            })
        elif question_type == QuestionType.STATEMENT_SET:
            base_constraints.update({
                "statement_count": 6,
                "varied_dimensions": True
            })
        elif question_type == QuestionType.PLOT_DAY:
            base_constraints.update({
                "task_count": 8,
                "realistic_activities": True,
                "time_balanced": True
            })
        
        return base_constraints


# Convenience functions for direct usage

async def generate_mcq_question(
    age_group: AgeGroup,
    question_number: int = 1,
    dimensions_focus: Optional[List[RaisecDimension]] = None
) -> Optional[Dict[str, Any]]:
    """Generate a multiple choice question.
    
    Args:
        age_group: Target age group
        question_number: Question number
        dimensions_focus: RAISEC dimensions to focus on
        
    Returns:
        Optional[Dict[str, Any]]: Generated question or None
    """
    generator = QuestionGenerator()
    return await generator.generate_question(
        question_type=QuestionType.MCQ,
        age_group=age_group,
        question_number=question_number,
        dimensions_focus=dimensions_focus
    )


async def generate_statement_set_question(
    age_group: AgeGroup,
    question_number: int = 1,
    dimensions_focus: Optional[List[RaisecDimension]] = None
) -> Optional[Dict[str, Any]]:
    """Generate a statement set question.
    
    Args:
        age_group: Target age group
        question_number: Question number
        dimensions_focus: RAISEC dimensions to focus on
        
    Returns:
        Optional[Dict[str, Any]]: Generated question or None
    """
    generator = QuestionGenerator()
    return await generator.generate_question(
        question_type=QuestionType.STATEMENT_SET,
        age_group=age_group,
        question_number=question_number,
        dimensions_focus=dimensions_focus
    )


async def generate_plot_day_question(
    age_group: AgeGroup,
    question_number: int = 1
) -> Optional[Dict[str, Any]]:
    """Generate a plot day question.
    
    Args:
        age_group: Target age group
        question_number: Question number
        
    Returns:
        Optional[Dict[str, Any]]: Generated question or None
    """
    generator = QuestionGenerator()
    return await generator.generate_question(
        question_type=QuestionType.PLOT_DAY,
        age_group=age_group,
        question_number=question_number
    )


# Export main classes and functions
__all__ = [
    "QuestionGenerator",
    "QuestionGenerationError",
    "generate_mcq_question",
    "generate_statement_set_question", 
    "generate_plot_day_question"
]