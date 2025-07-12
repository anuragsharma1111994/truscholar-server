"""Question generation chain for RAISEC assessment.

This module provides LangChain-based question generation for the TruScholar
RAISEC assessment, supporting various question types and age groups.
"""

import asyncio
from typing import Dict, List, Any, Optional, Type
from datetime import datetime

from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

from src.llm.base_llm import BaseLLM, LLMRequest, LLMMessage, LLMRole
from src.llm.llm_factory import LLMFactory
from src.utils.constants import QuestionType, RaisecDimension, AgeGroup
from src.utils.logger import get_logger
from ..prompts.question_prompts import QuestionPrompts
from ..parsers.question_parser import QuestionParser

logger = get_logger(__name__)


class QuestionChain(Chain):
    """LangChain for generating RAISEC assessment questions.
    
    This chain generates contextually appropriate questions for different
    age groups and question types in the RAISEC assessment.
    """
    
    llm: BaseLLM
    prompt_template: BasePromptTemplate
    output_parser: BaseOutputParser
    question_type: QuestionType
    age_group: AgeGroup
    verbose: bool = False
    
    # Chain configuration
    input_keys: List[str] = [
        "question_number",
        "dimensions_focus", 
        "context",
        "constraints"
    ]
    output_keys: List[str] = [
        "question_data",
        "generation_metadata"
    ]
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        question_type: QuestionType = QuestionType.MCQ,
        age_group: AgeGroup = AgeGroup.YOUNG_ADULT,
        **kwargs
    ):
        """Initialize question generation chain.
        
        Args:
            llm: Language model to use (creates default if None)
            question_type: Type of question to generate
            age_group: Target age group for questions
            **kwargs: Additional chain arguments
        """
        # Set up LLM
        if llm is None:
            llm = LLMFactory.create_from_settings()
        
        # Get appropriate prompt template and parser
        prompt_template = QuestionPrompts.get_template(question_type, age_group)
        output_parser = QuestionParser(question_type=question_type)
        
        super().__init__(
            llm=llm,
            prompt_template=prompt_template,
            output_parser=output_parser,
            question_type=question_type,
            age_group=age_group,
            **kwargs
        )
    
    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "question_generation"
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        required_keys = {"question_number", "dimensions_focus"}
        missing_keys = required_keys - set(inputs.keys())
        
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        
        # Validate question number
        question_number = inputs.get("question_number")
        if not isinstance(question_number, int) or not (1 <= question_number <= 12):
            raise ValueError("question_number must be an integer between 1 and 12")
        
        # Validate dimensions focus
        dimensions_focus = inputs.get("dimensions_focus", [])
        if not isinstance(dimensions_focus, list) or len(dimensions_focus) == 0:
            raise ValueError("dimensions_focus must be a non-empty list")
        
        # Validate dimension values
        valid_dimensions = [dim.value for dim in RaisecDimension]
        for dim in dimensions_focus:
            if dim not in valid_dimensions:
                raise ValueError(f"Invalid dimension: {dim}")
    
    def _prepare_llm_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the language model.
        
        Args:
            inputs: Raw input dictionary
            
        Returns:
            Dict[str, Any]: Formatted input for prompt template
        """
        # Extract and format dimensions
        dimensions_focus = inputs["dimensions_focus"]
        dimension_details = []
        
        for dim_code in dimensions_focus:
            dimension = RaisecDimension(dim_code)
            dimension_details.append({
                "code": dimension.value,
                "name": dimension.name_full,
                "description": dimension.description,
                "keywords": []  # Could be enhanced with keywords
            })
        
        return {
            "question_number": inputs["question_number"],
            "question_type": self.question_type.value,
            "question_type_description": self.question_type.description,
            "age_group": self.age_group.value,
            "age_range": f"{self.age_group.get_age_range()[0]}-{self.age_group.get_age_range()[1]}",
            "dimensions_focus": dimension_details,
            "primary_dimension": dimension_details[0] if dimension_details else None,
            "context": inputs.get("context", ""),
            "constraints": inputs.get("constraints", {}),
            "current_timestamp": datetime.utcnow().isoformat(),
        }
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the question generation chain synchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Callback manager for the run
            
        Returns:
            Dict[str, Any]: Generated question data and metadata
        """
        # Run async version in sync context
        return asyncio.run(self._acall(inputs, run_manager))
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the question generation chain asynchronously.
        
        Args:
            inputs: Input dictionary
            run_manager: Async callback manager for the run
            
        Returns:
            Dict[str, Any]: Generated question data and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate inputs
            self._validate_inputs(inputs)
            
            # Prepare input for LLM
            formatted_input = self._prepare_llm_input(inputs)
            
            if run_manager:
                run_manager.on_text(f"Generating {self.question_type.value} question for {self.age_group.value}")
            
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
                max_tokens=2000,
                temperature=0.7,
                metadata={
                    "question_type": self.question_type.value,
                    "age_group": self.age_group.value,
                    "question_number": inputs["question_number"]
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
                "question_data": parsed_result,
                "generation_metadata": {
                    "question_type": self.question_type.value,
                    "age_group": self.age_group.value,
                    "question_number": inputs["question_number"],
                    "dimensions_focus": inputs["dimensions_focus"],
                    "generation_time_seconds": generation_time,
                    "model_used": self.llm.model.value,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "cost_estimated": response.usage.estimated_cost if response.usage else 0.0,
                    "generated_at": start_time.isoformat(),
                    "parser_version": self.output_parser.get_version(),
                }
            }
            
            logger.info(
                f"Generated {self.question_type.value} question #{inputs['question_number']} "
                f"for {self.age_group.value} in {generation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return error result
            return {
                "question_data": None,
                "generation_metadata": {
                    "error": error_msg,
                    "question_type": self.question_type.value,
                    "age_group": self.age_group.value,
                    "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                    "generated_at": start_time.isoformat(),
                }
            }
    
    def generate_question(
        self,
        question_number: int,
        dimensions_focus: List[str],
        context: str = "",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a single question with the specified parameters.
        
        Args:
            question_number: Question number (1-12)
            dimensions_focus: List of RAISEC dimension codes to focus on
            context: Additional context for question generation
            constraints: Optional constraints for generation
            
        Returns:
            Dict[str, Any]: Generated question data and metadata
        """
        inputs = {
            "question_number": question_number,
            "dimensions_focus": dimensions_focus,
            "context": context,
            "constraints": constraints or {}
        }
        
        return self(inputs)
    
    async def agenerate_question(
        self,
        question_number: int,
        dimensions_focus: List[str],
        context: str = "",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Asynchronously generate a single question with the specified parameters.
        
        Args:
            question_number: Question number (1-12)
            dimensions_focus: List of RAISEC dimension codes to focus on
            context: Additional context for question generation
            constraints: Optional constraints for generation
            
        Returns:
            Dict[str, Any]: Generated question data and metadata
        """
        inputs = {
            "question_number": question_number,
            "dimensions_focus": dimensions_focus,
            "context": context,
            "constraints": constraints or {}
        }
        
        return await self.acall(inputs)
    
    def batch_generate_questions(
        self,
        questions_config: List[Dict[str, Any]],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate multiple questions concurrently.
        
        Args:
            questions_config: List of question configurations
            max_concurrent: Maximum concurrent generations
            
        Returns:
            List[Dict[str, Any]]: List of generated questions
        """
        return asyncio.run(self.abatch_generate_questions(questions_config, max_concurrent))
    
    async def abatch_generate_questions(
        self,
        questions_config: List[Dict[str, Any]],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Asynchronously generate multiple questions concurrently.
        
        Args:
            questions_config: List of question configurations
            max_concurrent: Maximum concurrent generations
            
        Returns:
            List[Dict[str, Any]]: List of generated questions
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(config: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.agenerate_question(**config)
        
        # Start all generation tasks
        tasks = [generate_with_semaphore(config) for config in questions_config]
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate question {i+1}: {str(result)}")
                processed_results.append({
                    "question_data": None,
                    "generation_metadata": {
                        "error": str(result),
                        "question_config": questions_config[i]
                    }
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_supported_question_types(self) -> List[QuestionType]:
        """Get list of supported question types.
        
        Returns:
            List[QuestionType]: Supported question types
        """
        return [
            QuestionType.MCQ,
            QuestionType.STATEMENT_SET,
            QuestionType.SCENARIO_MCQ,
            QuestionType.SCENARIO_MULTI_SELECT,
            QuestionType.THIS_OR_THAT,
            QuestionType.SCALE_RATING,
            QuestionType.PLOT_DAY,
        ]
    
    def get_supported_age_groups(self) -> List[AgeGroup]:
        """Get list of supported age groups.
        
        Returns:
            List[AgeGroup]: Supported age groups
        """
        return [
            AgeGroup.TEEN,
            AgeGroup.YOUNG_ADULT,
            AgeGroup.ADULT,
        ]
    
    @classmethod
    def create_for_type_and_age(
        cls,
        question_type: QuestionType,
        age_group: AgeGroup,
        llm: Optional[BaseLLM] = None
    ) -> "QuestionChain":
        """Create a chain for specific question type and age group.
        
        Args:
            question_type: Type of questions to generate
            age_group: Target age group
            llm: Optional LLM instance
            
        Returns:
            QuestionChain: Configured chain instance
        """
        return cls(
            llm=llm,
            question_type=question_type,
            age_group=age_group,
            verbose=True
        )


# Export the chain
__all__ = ["QuestionChain"]