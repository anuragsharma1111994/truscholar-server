"""Question generation service for TruScholar application.

This service handles question generation using LLM providers with fallback
to static questions when needed.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId

from src.core.config import get_settings
from src.core.settings import business_settings, feature_flags
from src.database.mongodb import MongoDB
from src.models.question import Question, QuestionOption, LikertStatement, PlotDayTask, ScoringRule
from src.utils.constants import AgeGroup, QuestionType, RaisecDimension
from src.utils.exceptions import TruScholarError, ValidationError
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class QuestionService:
    """Service for generating and managing test questions."""

    def __init__(self):
        """Initialize question service."""
        self.db = MongoDB
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
        if feature_flags.ENABLE_DYNAMIC_QUESTIONS:
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

            generator = QuestionGenerator()
            question_data = await generator.generate_question(
                question_type=question_type,
                age_group=age_group,
                question_number=question_number
            )

            if question_data:
                return self._create_question_from_llm_data(
                    test_id, question_number, question_type, age_group, question_data
                )

        except ImportError:
            logger.warning("LLM question generator not available")
        except Exception as e:
            logger.error(f"LLM question generation error: {str(e)}")

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
        # Similar to template creation but with LLM-specific handling
        # This would parse the structured output from the LLM
        # For now, using similar logic to template creation
        return self._create_question_from_template(
            test_id, question_number, question_type, age_group, llm_data
        )

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
