"""Question generation service for TruScholar application.

This service handles question generation using LLM providers with fallback
to static questions when needed.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from bson import ObjectId

from src.core.config import get_settings
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.cache.cache_manager import CacheManager
from src.cache.cache_keys import CacheKeys
from src.models.question import Question, QuestionOption, LikertStatement, PlotDayTask, ScoringRule
from src.models.test import Test
from src.utils.constants import AgeGroup, QuestionType, RaisecDimension
from src.utils.exceptions import TruScholarError, ValidationError, ResourceNotFoundError
from src.utils.logger import get_logger
from src.schemas.question_schemas import (
    QuestionResponse,
    QuestionContentResponse,
    MCQOption,
    StatementRating,
    ScenarioOption,
    ThisOrThatOption,
    ScaleEndpoint,
    PlotDayTask as PlotDayTaskSchema,
    PlotDayTimeSlot,
    QuestionGenerationStatusResponse,
    QuestionValidationResponse
)

settings = get_settings()
logger = get_logger(__name__)


class QuestionService:
    """Service for generating and managing test questions."""

    def __init__(self, db=None, cache=None):
        """Initialize question service."""
        self.db = db or MongoDB
        self.cache = cache or RedisClient
        self.cache_manager = CacheManager(self.cache)
        self.static_questions_path = Path("src/static_questions")

    async def generate_test_questions(
        self,
        test_id: ObjectId,
        age_group: AgeGroup,
        distribution: Dict[QuestionType, int]
    ) -> List[Question]:
        """Generate complete set of questions for a test.

        Args:
            test_id: ID of the test
            age_group: Age group for question targeting
            distribution: Distribution of question types

        Returns:
            List[Question]: Generated questions

        Raises:
            TruScholarError: If question generation fails
        """
        logger.info(f"Generating questions for test {test_id}",
                   extra={"test_id": str(test_id), "age_group": age_group.value})

        questions = []
        question_number = 1

        try:
            # Generate questions for each type
            for question_type, count in distribution.items():
                for i in range(count):
                    question = await self._generate_single_question(
                        test_id=test_id,
                        question_number=question_number,
                        question_type=question_type,
                        age_group=age_group
                    )
                    questions.append(question)
                    question_number += 1

            # Shuffle questions to randomize order
            random.shuffle(questions)

            # Update question numbers after shuffle
            for i, question in enumerate(questions, 1):
                question.question_number = i

            # Save all questions to database
            await self._save_questions(questions)

            logger.info(f"Generated {len(questions)} questions successfully",
                       extra={"test_id": str(test_id), "question_count": len(questions)})

            return questions

        except Exception as e:
            logger.error(f"Failed to generate questions: {str(e)}",
                        extra={"test_id": str(test_id), "error": str(e)})
            raise TruScholarError(f"Question generation failed: {str(e)}")

    async def _generate_single_question(
        self,
        test_id: ObjectId,
        question_number: int,
        question_type: QuestionType,
        age_group: AgeGroup
    ) -> Question:
        """Generate a single question.

        Args:
            test_id: ID of the test
            question_number: Number of the question
            question_type: Type of question to generate
            age_group: Age group for targeting

        Returns:
            Question: Generated question
        """
        logger.debug(f"Generating {question_type.value} question {question_number}",
                    extra={"question_type": question_type.value, "question_number": question_number})

        # Try LLM generation first if enabled
        if getattr(settings, 'ENABLE_DYNAMIC_QUESTIONS', True):
            try:
                question = await self._generate_with_llm(
                    test_id, question_number, question_type, age_group
                )
                if question:
                    return question
            except Exception as e:
                logger.warning(f"LLM generation failed, falling back to static: {str(e)}")

        # Fallback to static questions
        return await self._generate_static_question(
            test_id, question_number, question_type, age_group
        )

    async def _generate_with_llm(
        self,
        test_id: ObjectId,
        question_number: int,
        question_type: QuestionType,
        age_group: AgeGroup
    ) -> Optional[Question]:
        """Generate question using LLM.

        Args:
            test_id: ID of the test
            question_number: Number of the question
            question_type: Type of question to generate
            age_group: Age group for targeting

        Returns:
            Optional[Question]: Generated question or None if failed
        """
        try:
            # Import LLM services dynamically to avoid circular imports
            from src.langchain_handlers.question_generator import QuestionGenerator

            generator = QuestionGenerator(
                enable_caching=True,  # Enable question caching
                max_retries=3
            )
            
            # Prepare enhanced generation parameters
            dimensions_focus = self._get_balanced_dimensions_for_test(test_id, question_type)
            context = self._get_test_context(test_id, age_group)
            constraints = self._get_generation_constraints(age_group, question_type)
            
            question_data = await generator.generate_question(
                question_type=question_type,
                age_group=age_group,
                question_number=question_number,
                dimensions_focus=dimensions_focus,
                context=context,
                constraints=constraints
            )

            if question_data:
                question = self._create_question_from_llm_data(
                    test_id, question_number, question_type, age_group, question_data
                )
                
                # Cache successful generation for analytics
                await self._record_generation_success(
                    question_type, age_group, question_data.get("generation_metadata", {})
                )
                
                return question

        except ImportError:
            logger.warning("LLM question generator not available")
        except Exception as e:
            logger.error(f"LLM question generation error: {str(e)}")
            # Record failure for analytics
            await self._record_generation_failure(question_type, age_group, str(e))

        return None

    async def _generate_static_question(
        self,
        test_id: ObjectId,
        question_number: int,
        question_type: QuestionType,
        age_group: AgeGroup
    ) -> Question:
        """Generate question from static templates.

        Args:
            test_id: ID of the test
            question_number: Number of the question
            question_type: Type of question to generate
            age_group: Age group for targeting

        Returns:
            Question: Generated question
        """
        # Load static questions for age group and type
        static_questions = await self._load_static_questions(age_group, question_type)

        if not static_questions:
            raise TruScholarError(f"No static questions available for {question_type.value}")

        # Select random question
        question_template = random.choice(static_questions)

        # Create question from template
        return self._create_question_from_template(
            test_id, question_number, question_type, age_group, question_template
        )

    async def _load_static_questions(
        self,
        age_group: AgeGroup,
        question_type: QuestionType
    ) -> List[Dict[str, Any]]:
        """Load static questions from JSON files.

        Args:
            age_group: Age group for questions
            question_type: Type of questions to load

        Returns:
            List[Dict[str, Any]]: List of question templates
        """
        # Map age group to directory
        age_dir_map = {
            AgeGroup.TEEN: "age_13_17",
            AgeGroup.YOUNG_ADULT: "age_18_25",
            AgeGroup.ADULT: "age_26_35"
        }

        age_dir = age_dir_map.get(age_group)
        if not age_dir:
            raise ValidationError(f"Unsupported age group: {age_group}")

        # Build file path
        file_path = self.static_questions_path / age_dir / f"{question_type.value}.json"

        if not file_path.exists():
            # Create fallback static questions if file doesn't exist
            return self._create_fallback_questions(question_type)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load static questions: {str(e)}")
            return self._create_fallback_questions(question_type)

    def _create_fallback_questions(self, question_type: QuestionType) -> List[Dict[str, Any]]:
        """Create basic fallback questions when static files are missing.

        Args:
            question_type: Type of question to create

        Returns:
            List[Dict[str, Any]]: List of fallback question templates
        """
        if question_type == QuestionType.MCQ:
            return [{
                "question_text": "Which type of work environment appeals to you most?",
                "options": [
                    {"id": "a", "text": "Hands-on workshop or outdoors", "dimensions": {"R": 1.0}},
                    {"id": "b", "text": "Creative studio or gallery", "dimensions": {"A": 1.0}},
                    {"id": "c", "text": "Research laboratory", "dimensions": {"I": 1.0}},
                    {"id": "d", "text": "Community center helping others", "dimensions": {"S": 1.0}}
                ],
                "dimensions": ["R", "A", "I", "S"]
            }]

        elif question_type == QuestionType.STATEMENT_SET:
            return [{
                "question_text": "Rate how much you agree with these statements:",
                "statements": [
                    {"id": 1, "text": "I enjoy working with tools and machinery", "dimension": "R"},
                    {"id": 2, "text": "I like expressing myself through art", "dimension": "A"},
                    {"id": 3, "text": "I enjoy solving complex problems", "dimension": "I"},
                    {"id": 4, "text": "I feel fulfilled when helping others", "dimension": "S"},
                    {"id": 5, "text": "I like leading teams and projects", "dimension": "E"}
                ],
                "dimensions": ["R", "A", "I", "S", "E"]
            }]

        elif question_type == QuestionType.PLOT_DAY:
            return [{
                "question_text": "Arrange these activities in your ideal daily schedule:",
                "tasks": [
                    {"id": "task1", "title": "Team Meeting", "description": "Collaborate with colleagues", "primary_dimension": "S"},
                    {"id": "task2", "title": "Data Analysis", "description": "Analyze complex datasets", "primary_dimension": "I"},
                    {"id": "task3", "title": "Creative Design", "description": "Design visual materials", "primary_dimension": "A"},
                    {"id": "task4", "title": "Equipment Maintenance", "description": "Fix and maintain tools", "primary_dimension": "R"},
                    {"id": "task5", "title": "Project Planning", "description": "Plan and organize projects", "primary_dimension": "E"},
                    {"id": "task6", "title": "Documentation", "description": "Create detailed records", "primary_dimension": "C"},
                    {"id": "task7", "title": "Client Presentation", "description": "Present to stakeholders", "primary_dimension": "E"},
                    {"id": "task8", "title": "Quality Control", "description": "Review and validate work", "primary_dimension": "C"}
                ],
                "dimensions": ["R", "A", "I", "S", "E", "C"]
            }]

        # Add more fallback patterns for other question types
        return [{"question_text": f"Default {question_type.value} question", "dimensions": ["R", "A", "I"]}]

    def _create_question_from_template(
        self,
        test_id: ObjectId,
        question_number: int,
        question_type: QuestionType,
        age_group: AgeGroup,
        template: Dict[str, Any]
    ) -> Question:
        """Create Question object from template data.

        Args:
            test_id: ID of the test
            question_number: Number of the question
            question_type: Type of question
            age_group: Age group for targeting
            template: Question template data

        Returns:
            Question: Created question object
        """
        # Base question data
        question_data = {
            "test_id": test_id,
            "question_number": question_number,
            "question_type": question_type,
            "question_text": template["question_text"],
            "age_group": f"{age_group.value[0]}-{age_group.value[1]}",
            "dimensions_evaluated": [RaisecDimension(d) for d in template.get("dimensions", [])],
            "is_static": True,
            "scoring_rule": ScoringRule()
        }

        # Add type-specific content
        if question_type == QuestionType.MCQ:
            options = []
            for opt_data in template.get("options", []):
                option = QuestionOption(
                    id=opt_data["id"],
                    text=opt_data["text"],
                    dimension_weights={
                        RaisecDimension(dim): weight
                        for dim, weight in opt_data.get("dimensions", {}).items()
                    }
                )
                options.append(option)
            question_data["options"] = options

        elif question_type == QuestionType.STATEMENT_SET:
            statements = []
            for stmt_data in template.get("statements", []):
                statement = LikertStatement(
                    id=stmt_data["id"],
                    text=stmt_data["text"],
                    dimension=RaisecDimension(stmt_data["dimension"]),
                    reverse_scored=stmt_data.get("reverse_scored", False)
                )
                statements.append(statement)
            question_data["statements"] = statements

        elif question_type == QuestionType.THIS_OR_THAT:
            option_a_data = template.get("option_a", {})
            option_b_data = template.get("option_b", {})

            question_data["option_a"] = QuestionOption(
                id="a",
                text=option_a_data["text"],
                dimension_weights={
                    RaisecDimension(dim): weight
                    for dim, weight in option_a_data.get("dimensions", {}).items()
                }
            )
            question_data["option_b"] = QuestionOption(
                id="b",
                text=option_b_data["text"],
                dimension_weights={
                    RaisecDimension(dim): weight
                    for dim, weight in option_b_data.get("dimensions", {}).items()
                }
            )

        elif question_type == QuestionType.PLOT_DAY:
            tasks = []
            for task_data in template.get("tasks", []):
                task = PlotDayTask(
                    id=task_data["id"],
                    title=task_data["title"],
                    description=task_data.get("description", ""),
                    primary_dimension=RaisecDimension(task_data["primary_dimension"]),
                    secondary_dimensions=[
                        RaisecDimension(d) for d in task_data.get("secondary_dimensions", [])
                    ]
                )
                tasks.append(task)
            question_data["tasks"] = tasks

        elif question_type == QuestionType.SCALE_RATING:
            question_data.update({
                "scale_min": template.get("scale_min", 1),
                "scale_max": template.get("scale_max", 10),
                "scale_labels": template.get("scale_labels", {
                    "1": "Not at all",
                    "10": "Extremely"
                })
            })

        return Question(**question_data)

    def _create_question_from_llm_data(
        self,
        test_id: ObjectId,
        question_number: int,
        question_type: QuestionType,
        age_group: AgeGroup,
        llm_data: Dict[str, Any]
    ) -> Question:
        """Create Question object from LLM-generated data.

        Args:
            test_id: ID of the test
            question_number: Number of the question
            question_type: Type of question
            age_group: Age group for targeting
            llm_data: LLM-generated question data

        Returns:
            Question: Created question object
        """
        from src.utils.formatters import QuestionFormatter
        
        # Use formatter to structure LLM data
        formatter = QuestionFormatter()
        formatted_data = formatter.format_llm_response(llm_data, question_type)
        
        # Base question data
        question_data = {
            "test_id": test_id,
            "question_number": question_number,
            "question_type": question_type,
            "question_text": formatted_data.get("question_text", ""),
            "instructions": formatted_data.get("instructions"),
            "age_group": f"{age_group.value[0]}-{age_group.value[1]}",
            "dimensions_evaluated": [
                RaisecDimension(d) for d in formatted_data.get("dimensions_evaluated", [])
            ],
            "is_static": False,
            "llm_metadata": {
                "model": formatted_data.get("generation_metadata", {}).get("model", "unknown"),
                "prompt_version": formatted_data.get("generation_metadata", {}).get("prompt_version", "1.0"),
                "generation_attempt": formatted_data.get("generation_metadata", {}).get("generation_attempt", 1),
                "generated_at": formatted_data.get("generation_metadata", {}).get("generated_at"),
                "validation_passed": True
            },
            "estimated_time_seconds": self._estimate_question_time(question_type),
            "scoring_rule": ScoringRule()
        }

        # Add type-specific content with enhanced parsing
        if question_type == QuestionType.MCQ:
            options = []
            for opt_data in formatted_data.get("options", []):
                option = QuestionOption(
                    id=opt_data.get("id", ""),
                    text=opt_data.get("text", ""),
                    dimension_weights=self._parse_dimension_weights(
                        opt_data.get("scoring_guide", {})
                    )
                )
                options.append(option)
            question_data["options"] = options

        elif question_type == QuestionType.STATEMENT_SET:
            statements = []
            for stmt_data in formatted_data.get("statements", []):
                statement = LikertStatement(
                    id=stmt_data.get("id", ""),
                    text=stmt_data.get("text", ""),
                    dimension=RaisecDimension(stmt_data.get("dimension", "R")),
                    reverse_scored=stmt_data.get("reverse_scored", False)
                )
                statements.append(statement)
            question_data["statements"] = statements

        elif question_type == QuestionType.THIS_OR_THAT:
            if "option_a" in formatted_data and "option_b" in formatted_data:
                question_data["option_a"] = QuestionOption(
                    id="a",
                    text=formatted_data["option_a"].get("text", ""),
                    dimension_weights=self._parse_dimension_weights(
                        formatted_data["option_a"].get("scoring_guide", {})
                    )
                )
                question_data["option_b"] = QuestionOption(
                    id="b", 
                    text=formatted_data["option_b"].get("text", ""),
                    dimension_weights=self._parse_dimension_weights(
                        formatted_data["option_b"].get("scoring_guide", {})
                    )
                )

        elif question_type == QuestionType.PLOT_DAY:
            tasks = []
            for task_data in formatted_data.get("tasks", []):
                task = PlotDayTask(
                    id=task_data.get("id", ""),
                    title=task_data.get("title", ""),
                    description=task_data.get("description", ""),
                    category=task_data.get("category", "general"),
                    duration=task_data.get("duration", "1-2 hours"),
                    primary_dimension=RaisecDimension(
                        task_data.get("primary_dimension", "R")
                    ),
                    secondary_dimensions=[
                        RaisecDimension(d) for d in task_data.get("secondary_dimensions", [])
                    ]
                )
                tasks.append(task)
            question_data["tasks"] = tasks

        elif question_type == QuestionType.SCALE_RATING:
            question_data.update({
                "scale_min": formatted_data.get("scale_min", 1),
                "scale_max": formatted_data.get("scale_max", 10),
                "scale_labels": formatted_data.get("scale_labels", {
                    "min": "Not at all",
                    "max": "Extremely"
                }),
                "scale_item": formatted_data.get("scale_item", question_data["question_text"])
            })

        return Question(**question_data)

    async def _save_questions(self, questions: List[Question]) -> None:
        """Save questions to database.

        Args:
            questions: List of questions to save
        """
        if not questions:
            return

        # Convert to dictionaries for insertion
        question_docs = []
        for question in questions:
            question_dict = question.to_dict(exclude={"id"})
            question_docs.append(question_dict)

        # Bulk insert
        result = await self.db.insert_many("questions", question_docs)

        # Update question IDs
        for question, inserted_id in zip(questions, result.inserted_ids):
            question.id = inserted_id

        logger.info(f"Saved {len(questions)} questions to database")

    async def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get question by ID.

        Args:
            question_id: ID of the question

        Returns:
            Optional[Question]: Question if found
        """
        question_data = await self.db.find_one("questions", {"_id": ObjectId(question_id)})
        return Question.from_dict(question_data) if question_data else None

    async def validate_question_distribution(
        self,
        distribution: Dict[QuestionType, int]
    ) -> bool:
        """Validate question type distribution.

        Args:
            distribution: Question type distribution

        Returns:
            bool: True if valid distribution
        """
        total_questions = sum(distribution.values())

        # Should total 12 questions
        if total_questions != 12:
            return False

        # Each type should have at least 1 question
        if any(count < 1 for count in distribution.values()):
            return False

        # PLOT_DAY should only have 1 question (resource intensive)
        if distribution.get(QuestionType.PLOT_DAY, 0) > 1:
            return False

        return True

    def get_question_type_weights(self) -> Dict[QuestionType, float]:
        """Get relative weights for question types in scoring.

        Returns:
            Dict[QuestionType, float]: Question type weights
        """
        return {
            QuestionType.MCQ: 1.0,
            QuestionType.STATEMENT_SET: 0.8,
            QuestionType.SCENARIO_MCQ: 1.0,
            QuestionType.SCENARIO_MULTI_SELECT: 0.6,
            QuestionType.THIS_OR_THAT: 1.2,
            QuestionType.SCALE_RATING: 0.9,
            QuestionType.PLOT_DAY: 1.5,
        }

    # New methods for API endpoints

    async def validate_test_ownership(self, test_id: str, user_id: str) -> bool:
        """Validate that user owns the test.
        
        Args:
            test_id: Test ID
            user_id: User ID
            
        Returns:
            bool: True if user owns test
            
        Raises:
            PermissionError: If user doesn't own test
            ResourceNotFoundError: If test not found
        """
        test_data = await self.db.find_one("tests", {"_id": ObjectId(test_id)})
        if not test_data:
            raise ResourceNotFoundError(f"Test {test_id} not found")
            
        if str(test_data["user_id"]) != user_id:
            raise PermissionError("User does not have access to this test")
            
        return True

    async def generate_questions(
        self,
        test_id: str,
        age_group: AgeGroup,
        question_distribution: Optional[Dict[QuestionType, int]] = None,
        include_static: bool = False
    ) -> QuestionGenerationStatusResponse:
        """Generate questions for a test and return status.
        
        Args:
            test_id: Test ID
            age_group: Age group for questions
            question_distribution: Optional custom distribution
            include_static: Whether to include static questions
            
        Returns:
            QuestionGenerationStatusResponse: Generation status
        """
        # Use default distribution if not provided
        if not question_distribution:
            question_distribution = {"standard": 70, "scenario": 20, "interactive": 10}
            
        # Validate distribution
        if not await self.validate_question_distribution(question_distribution):
            raise ValidationError("Invalid question distribution")
            
        # Check if questions already exist
        existing_count = await self.db.count_documents(
            "questions", {"test_id": ObjectId(test_id)}
        )
        
        if existing_count >= 12:
            return QuestionGenerationStatusResponse(
                test_id=test_id,
                status="completed",
                questions_generated=12,
                total_questions=12
            )
            
        # Start generation (in production this would be async with Celery)
        try:
            questions = await self.generate_test_questions(
                test_id=ObjectId(test_id),
                age_group=age_group,
                distribution=question_distribution
            )
            
            return QuestionGenerationStatusResponse(
                test_id=test_id,
                status="completed",
                questions_generated=len(questions),
                total_questions=12
            )
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return QuestionGenerationStatusResponse(
                test_id=test_id,
                status="failed",
                questions_generated=existing_count,
                total_questions=12,
                errors=[str(e)]
            )

    async def get_test_questions(
        self,
        test_id: str,
        include_content: bool = True
    ) -> List[QuestionResponse]:
        """Get all questions for a test.
        
        Args:
            test_id: Test ID
            include_content: Whether to include full content
            
        Returns:
            List[QuestionResponse]: List of questions
        """
        # Try cache first
        cache_key = CacheKeys.test_questions(test_id)
        cached = await self.cache_manager.get(cache_key)
        if cached and include_content:
            return [self._question_to_response(Question.from_dict(q)) for q in cached]
            
        # Get from database
        question_docs = await self.db.find(
            "questions",
            {"test_id": ObjectId(test_id)},
            sort=[("question_number", 1)]
        )
        
        questions = [Question.from_dict(doc) for doc in question_docs]
        
        # Cache if we got full content
        if include_content and questions:
            await self.cache_manager.cache_questions(
                test_id, [q.to_dict() for q in questions]
            )
            
        return [self._question_to_response(q, include_content) for q in questions]

    async def get_question(self, question_id: str, user_id: str) -> Optional[QuestionResponse]:
        """Get a specific question with ownership validation.
        
        Args:
            question_id: Question ID
            user_id: User ID for validation
            
        Returns:
            Optional[QuestionResponse]: Question if found and authorized
        """
        # Get question
        question_data = await self.db.find_one(
            "questions", {"_id": ObjectId(question_id)}
        )
        
        if not question_data:
            return None
            
        question = Question.from_dict(question_data)
        
        # Validate ownership through test
        await self.validate_test_ownership(str(question.test_id), user_id)
        
        return self._question_to_response(question)

    async def regenerate_question(
        self,
        test_id: str,
        question_number: int,
        user_id: str,
        reason: Optional[str] = None
    ) -> Optional[QuestionResponse]:
        """Regenerate a specific question.
        
        Args:
            test_id: Test ID
            question_number: Question number to regenerate
            user_id: User ID for validation
            reason: Optional reason for regeneration
            
        Returns:
            Optional[QuestionResponse]: Regenerated question
        """
        # Validate ownership
        await self.validate_test_ownership(test_id, user_id)
        
        # Get existing question
        question_data = await self.db.find_one(
            "questions",
            {
                "test_id": ObjectId(test_id),
                "question_number": question_number
            }
        )
        
        if not question_data:
            return None
            
        old_question = Question.from_dict(question_data)
        
        # Generate new question
        new_question = await self._generate_single_question(
            test_id=ObjectId(test_id),
            question_number=question_number,
            question_type=old_question.question_type,
            age_group=self._get_age_group_from_string(old_question.age_group)
        )
        
        # Update in database
        new_question.id = old_question.id
        new_question.regenerated_at = datetime.utcnow()
        
        await self.db.update_one(
            "questions",
            {"_id": old_question.id},
            {"$set": new_question.to_dict(exclude={"id", "_id"})}
        )
        
        # Clear cache
        await self.cache_manager.delete(CacheKeys.test_questions(test_id))
        
        # Log regeneration
        logger.info(
            f"Question regenerated",
            extra={
                "test_id": test_id,
                "question_number": question_number,
                "reason": reason
            }
        )
        
        return self._question_to_response(new_question)

    async def validate_question(
        self,
        question_id: str,
        user_id: str
    ) -> Optional[QuestionValidationResponse]:
        """Validate a question for quality and correctness.
        
        Args:
            question_id: Question ID
            user_id: User ID for validation
            
        Returns:
            Optional[QuestionValidationResponse]: Validation result
        """
        # Get question
        question_data = await self.db.find_one(
            "questions", {"_id": ObjectId(question_id)}
        )
        
        if not question_data:
            return None
            
        question = Question.from_dict(question_data)
        
        # Validate ownership
        await self.validate_test_ownership(str(question.test_id), user_id)
        
        # Perform validation
        errors = []
        warnings = []
        suggestions = []
        
        # Check question text
        if len(question.question_text) < 10:
            errors.append("Question text is too short")
        elif len(question.question_text) > 500:
            warnings.append("Question text might be too long")
            
        # Check RAISEC dimensions
        if not question.dimensions_evaluated:
            errors.append("No RAISEC dimensions specified")
        elif len(question.dimensions_evaluated) > 3:
            warnings.append("Too many dimensions for a single question")
            
        # Type-specific validation
        if question.question_type == QuestionType.MCQ:
            if not question.options or len(question.options) < 2:
                errors.append("MCQ must have at least 2 options")
            elif len(question.options) > 6:
                warnings.append("Consider reducing number of options")
                
        elif question.question_type == QuestionType.PLOT_DAY:
            if not question.tasks or len(question.tasks) < 5:
                errors.append("Plot day needs at least 5 tasks")
            elif len(question.tasks) > 12:
                warnings.append("Too many tasks for plot day")
                
        # Age appropriateness
        if question.age_group == "13-17" and any(
            word in question.question_text.lower()
            for word in ["career", "job", "salary", "workplace"]
        ):
            suggestions.append("Consider using age-appropriate terms like 'future interests' instead of 'career'")
            
        return QuestionValidationResponse(
            question_id=question_id,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    async def get_generation_status(
        self,
        test_id: str,
        user_id: str
    ) -> Optional[QuestionGenerationStatusResponse]:
        """Get question generation status for a test.
        
        Args:
            test_id: Test ID
            user_id: User ID for validation
            
        Returns:
            Optional[QuestionGenerationStatusResponse]: Generation status
        """
        # Validate ownership
        await self.validate_test_ownership(test_id, user_id)
        
        # Check current question count
        question_count = await self.db.count_documents(
            "questions", {"test_id": ObjectId(test_id)}
        )
        
        # Check if generation is in progress (would check Celery task in production)
        status = "completed" if question_count >= 12 else "in_progress"
        
        return QuestionGenerationStatusResponse(
            test_id=test_id,
            status=status,
            questions_generated=question_count,
            total_questions=12,
            estimated_completion_seconds=None if status == "completed" else (12 - question_count) * 2
        )

    async def reload_static_questions(
        self,
        age_group: Optional[str] = None,
        question_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reload static questions from files.
        
        Args:
            age_group: Optional specific age group
            question_type: Optional specific question type
            
        Returns:
            Dict[str, Any]: Reload summary
        """
        reloaded = {
            "age_groups": [],
            "question_types": [],
            "total_files": 0,
            "errors": []
        }
        
        # Determine what to reload
        age_groups = [AgeGroup(age_group)] if age_group else list(AgeGroup)
        question_types = [QuestionType(question_type)] if question_type else list(QuestionType)
        
        for ag in age_groups:
            for qt in question_types:
                try:
                    questions = await self._load_static_questions(ag, qt)
                    if questions:
                        reloaded["age_groups"].append(ag.value)
                        reloaded["question_types"].append(qt.value)
                        reloaded["total_files"] += 1
                except Exception as e:
                    reloaded["errors"].append(f"{ag.value}/{qt.value}: {str(e)}")
                    
        return reloaded

    async def get_question_distribution(
        self,
        age_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get recommended question type distribution.
        
        Args:
            age_group: Optional age group filter
            
        Returns:
            Dict[str, Any]: Distribution information
        """
        base_distribution = {"standard": 70, "scenario": 20, "interactive": 10}
        
        # Age-specific adjustments could be added here
        distribution = {
            "standard": {qt.value: count for qt, count in base_distribution.items()},
            "total_questions": sum(base_distribution.values()),
            "weights": {qt.value: weight for qt, weight in self.get_question_type_weights().items()}
        }
        
        if age_group:
            # Could add age-specific recommendations
            distribution["age_group"] = age_group
            distribution["notes"] = self._get_age_specific_notes(age_group)
            
        return distribution

    def _question_to_response(
        self,
        question: Question,
        include_content: bool = True
    ) -> QuestionResponse:
        """Convert Question model to response schema.
        
        Args:
            question: Question model
            include_content: Whether to include full content
            
        Returns:
            QuestionResponse: Response schema
        """
        # Base response
        response_data = {
            "id": str(question.id),
            "test_id": str(question.test_id),
            "question_number": question.question_number,
            "question_type": question.question_type,
            "question_text": question.question_text,
            "instructions": question.instructions,
            "raisec_dimensions": question.dimensions_evaluated,
            "time_estimate_seconds": question.estimated_time_seconds,
            "is_required": question.is_required,
            "generated_by": "static" if question.is_static else "ai",
            "prompt_version": question.llm_metadata.get("prompt_version") if question.llm_metadata else None,
            "created_at": question.created_at,
            "regenerated_at": question.regenerated_at
        }
        
        # Add content if requested
        if include_content:
            response_data["content"] = self._create_content_response(question)
            
        return QuestionResponse(**response_data)

    def _create_content_response(self, question: Question) -> QuestionContentResponse:
        """Create question content response based on question type.
        
        Args:
            question: Question model
            
        Returns:
            QuestionContentResponse: Content response
        """
        content = {}
        
        if question.question_type == QuestionType.MCQ:
            if question.options:
                content["options"] = [
                    MCQOption(
                        id=opt.id,
                        text=opt.text,
                        raisec_dimension=list(opt.dimension_weights.keys())[0] if opt.dimension_weights else RaisecDimension.REALISTIC
                    )
                    for opt in question.options
                ]
                
        elif question.question_type == QuestionType.STATEMENT_SET:
            if question.statements:
                content["statements"] = [
                    StatementRating(
                        id=str(stmt.id),
                        text=stmt.text,
                        raisec_dimension=stmt.dimension
                    )
                    for stmt in question.statements
                ]
                content["scale_labels"] = {
                    "1": "Strongly Disagree",
                    "2": "Disagree",
                    "3": "Neutral",
                    "4": "Agree",
                    "5": "Strongly Agree"
                }
                
        elif question.question_type == QuestionType.SCENARIO_MCQ:
            content["scenario"] = question.scenario_description
            if question.options:
                content["scenario_options"] = [
                    ScenarioOption(
                        id=opt.id,
                        text=opt.text,
                        raisec_dimensions=list(opt.dimension_weights.keys())
                    )
                    for opt in question.options
                ]
                
        elif question.question_type == QuestionType.THIS_OR_THAT:
            if question.option_a and question.option_b:
                content["option_a"] = ThisOrThatOption(
                    id="A",
                    text=question.option_a.text,
                    raisec_dimension=list(question.option_a.dimension_weights.keys())[0] if question.option_a.dimension_weights else RaisecDimension.REALISTIC
                )
                content["option_b"] = ThisOrThatOption(
                    id="B",
                    text=question.option_b.text,
                    raisec_dimension=list(question.option_b.dimension_weights.keys())[0] if question.option_b.dimension_weights else RaisecDimension.ARTISTIC
                )
                
        elif question.question_type == QuestionType.SCALE_RATING:
            content["scale_min"] = ScaleEndpoint(
                value=question.scale_min or 1,
                label=question.scale_labels.get("min", "Not at all")
            )
            content["scale_max"] = ScaleEndpoint(
                value=question.scale_max or 10,
                label=question.scale_labels.get("max", "Extremely")
            )
            content["scale_item"] = question.question_text
            
        elif question.question_type == QuestionType.PLOT_DAY:
            if question.tasks:
                content["tasks"] = [
                    PlotDayTaskSchema(
                        id=task.id,
                        text=task.title,
                        category=task.category or "general",
                        typical_duration=task.duration or "1-2 hours",
                        raisec_dimensions=[task.primary_dimension] + (task.secondary_dimensions or [])
                    )
                    for task in question.tasks
                ]
            content["time_slots"] = [
                PlotDayTimeSlot(id="morning", time_range="9:00-12:00", label="Morning"),
                PlotDayTimeSlot(id="afternoon", time_range="12:00-15:00", label="Afternoon"),
                PlotDayTimeSlot(id="late_afternoon", time_range="15:00-18:00", label="Late Afternoon"),
                PlotDayTimeSlot(id="evening", time_range="18:00-21:00", label="Evening")
            ]
            content["special_slot"] = {"id": "not_interested", "label": "Not Interested"}
            
        return QuestionContentResponse(**content)

    def _get_age_specific_notes(self, age_group: str) -> List[str]:
        """Get age-specific notes for question distribution.
        
        Args:
            age_group: Age group
            
        Returns:
            List[str]: Notes for the age group
        """
        notes = {
            "13-17": [
                "Focus on interests and hobbies rather than careers",
                "Use simpler language and relatable scenarios",
                "Include more visual and interactive questions"
            ],
            "18-25": [
                "Balance between education and career exploration",
                "Include modern job roles and gig economy options",
                "Consider student and early career scenarios"
            ],
            "26-35": [
                "Focus on career progression and specialization",
                "Include leadership and management scenarios",
                "Consider work-life balance aspects"
            ]
        }
        
        return notes.get(age_group, [])
    
    def _get_balanced_dimensions_for_test(
        self,
        test_id: ObjectId,
        question_type: QuestionType
    ) -> List[RaisecDimension]:
        """Get balanced RAISEC dimensions for test questions.
        
        Args:
            test_id: Test ID
            question_type: Question type
            
        Returns:
            List[RaisecDimension]: Balanced dimensions
        """
        # In production, this would analyze existing questions in the test
        # For now, return balanced distribution based on question type
        from random import sample
        
        all_dimensions = list(RaisecDimension)
        
        # Different question types focus on different numbers of dimensions
        focus_counts = {
            QuestionType.MCQ: 3,
            QuestionType.STATEMENT_SET: 4,
            QuestionType.SCENARIO_MCQ: 2,
            QuestionType.SCENARIO_MULTI_SELECT: 3,
            QuestionType.THIS_OR_THAT: 2,
            QuestionType.SCALE_RATING: 2,
            QuestionType.PLOT_DAY: 6  # All dimensions
        }
        
        count = focus_counts.get(question_type, 3)
        return sample(all_dimensions, min(count, len(all_dimensions)))
    
    def _get_test_context(self, test_id: ObjectId, age_group: AgeGroup) -> str:
        """Get context for test-specific question generation.
        
        Args:
            test_id: Test ID
            age_group: Age group
            
        Returns:
            str: Test context
        """
        # In production, this would consider test metadata, user profile, etc.
        base_context = {
            AgeGroup.TEEN: "Focus on school activities, interests, and future aspirations",
            AgeGroup.YOUNG_ADULT: "Consider academic choices, early career exploration, and skill development",
            AgeGroup.ADULT: "Focus on career advancement, leadership roles, and professional fulfillment"
        }
        
        return base_context.get(age_group, "General career interest assessment")
    
    def _get_generation_constraints(
        self,
        age_group: AgeGroup,
        question_type: QuestionType
    ) -> Dict[str, Any]:
        """Get generation constraints for LLM.
        
        Args:
            age_group: Age group
            question_type: Question type
            
        Returns:
            Dict[str, Any]: Generation constraints
        """
        constraints = {
            "cultural_context": "Indian",
            "language_complexity": "moderate",
            "avoid_bias": True,
            "modern_scenarios": True
        }
        
        # Age-specific constraints
        if age_group == AgeGroup.TEEN:
            constraints.update({
                "language_complexity": "simple",
                "focus_on_activities": True,
                "avoid_career_jargon": True
            })
        elif age_group == AgeGroup.ADULT:
            constraints.update({
                "include_leadership": True,
                "professional_context": True
            })
        
        return constraints
    
    async def _record_generation_success(
        self,
        question_type: QuestionType,
        age_group: AgeGroup,
        metadata: Dict[str, Any]
    ) -> None:
        """Record successful question generation for analytics.
        
        Args:
            question_type: Question type
            age_group: Age group
            metadata: Generation metadata
        """
        # In production, this would record to analytics/metrics system
        logger.info(
            "Question generation successful",
            extra={
                "question_type": question_type.value,
                "age_group": age_group.value,
                "generation_time": metadata.get("generation_time"),
                "attempt": metadata.get("generation_attempt", 1)
            }
        )
    
    async def _record_generation_failure(
        self,
        question_type: QuestionType,
        age_group: AgeGroup,
        error: str
    ) -> None:
        """Record failed question generation for analytics.
        
        Args:
            question_type: Question type
            age_group: Age group
            error: Error message
        """
        # In production, this would record to analytics/metrics system
        logger.warning(
            "Question generation failed",
            extra={
                "question_type": question_type.value,
                "age_group": age_group.value,
                "error": error
            }
        )
    
    def _parse_dimension_weights(self, scoring_guide: Dict[str, Any]) -> Dict[RaisecDimension, float]:
        """Parse dimension weights from LLM scoring guide.
        
        Args:
            scoring_guide: Scoring guide from LLM
            
        Returns:
            Dict[RaisecDimension, float]: Dimension weights
        """
        weights = {}
        
        for key, value in scoring_guide.items():
            if isinstance(value, dict):
                dimension_str = value.get("dimension", "R")
                score = value.get("score", 1.0)
                
                try:
                    dimension = RaisecDimension(dimension_str)
                    weights[dimension] = float(score)
                except (ValueError, TypeError):
                    # Default fallback
                    weights[RaisecDimension.REALISTIC] = 1.0
        
        return weights
    
    def _estimate_question_time(self, question_type: QuestionType) -> int:
        """Estimate time to complete question in seconds.
        
        Args:
            question_type: Question type
            
        Returns:
            int: Estimated seconds
        """
        time_estimates = {
            QuestionType.MCQ: 30,
            QuestionType.STATEMENT_SET: 90,
            QuestionType.SCENARIO_MCQ: 45,
            QuestionType.SCENARIO_MULTI_SELECT: 60,
            QuestionType.THIS_OR_THAT: 20,
            QuestionType.SCALE_RATING: 25,
            QuestionType.PLOT_DAY: 180
        }
        
        return time_estimates.get(question_type, 45)
    
    def _get_age_group_from_string(self, age_group_str: str) -> AgeGroup:
        """Convert age group string to AgeGroup enum.
        
        Args:
            age_group_str: Age group string (e.g., "18-25")
            
        Returns:
            AgeGroup: Age group enum
        """
        for ag in AgeGroup:
            if ag.value == age_group_str:
                return ag
        
        # Try to parse age range
        if "-" in age_group_str:
            try:
                min_age = int(age_group_str.split("-")[0])
                return AgeGroup.from_age(min_age)
            except:
                pass
                
        raise ValueError(f"Invalid age group: {age_group_str}")
