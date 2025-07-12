"""Scoring service for RAISEC test calculations.

This service handles the calculation of RAISEC dimension scores and
generates the final RAISEC code based on user responses.
"""

from typing import Dict, List, Tuple

from bson import ObjectId

from src.core.settings import business_settings
from src.database.mongodb import MongoDB
from src.models.answer import Answer
from src.models.question import Question
from src.models.test import DimensionScore, TestScores
from src.utils.constants import RaisecDimension
from src.utils.exceptions import TruScholarError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScoringService:
    """Service for calculating RAISEC test scores."""

    def __init__(self):
        """Initialize scoring service."""
        self.db = MongoDB

    async def calculate_test_scores(self, test_id: ObjectId) -> TestScores:
        """Calculate complete test scores and generate RAISEC code.

        Args:
            test_id: ID of the test to score

        Returns:
            TestScores: Calculated test scores

        Raises:
            TruScholarError: If scoring fails
            ValidationError: If test data is invalid
        """
        logger.info(f"Calculating scores for test {test_id}", extra={"test_id": str(test_id)})

        try:
            # Get all answers for the test
            answers = await self._get_test_answers(test_id)

            if len(answers) < 12:
                raise ValidationError(f"Incomplete test: only {len(answers)} answers found")

            # Get questions for context
            questions = await self._get_test_questions(test_id)
            question_map = {str(q.id): q for q in questions}

            # Calculate dimension scores
            raw_scores = await self._calculate_dimension_scores(answers, question_map)

            # Normalize and rank scores
            normalized_scores = self._normalize_scores(raw_scores)
            ranked_scores = self._rank_dimensions(normalized_scores)

            # Calculate quality metrics
            consistency_score = self._calculate_consistency_score(answers, question_map)
            differentiation_index = self._calculate_differentiation_index(normalized_scores)

            # Generate RAISEC codes
            primary_code = self._generate_raisec_code(ranked_scores, count=3)
            secondary_code = self._generate_raisec_code(ranked_scores, count=6)

            # Create dimension score objects
            dimension_scores = {}
            total_score = sum(normalized_scores.values())

            for dimension, score in normalized_scores.items():
                dimension_scores[dimension.value] = DimensionScore(
                    dimension=dimension,
                    raw_score=raw_scores[dimension],
                    percentage=score,
                    rank=ranked_scores[dimension],
                    confidence=self._calculate_dimension_confidence(
                        dimension, answers, question_map
                    )
                )

            # Create test scores object
            test_scores = TestScores(
                realistic=dimension_scores["R"],
                artistic=dimension_scores["A"],
                investigative=dimension_scores["I"],
                social=dimension_scores["S"],
                enterprising=dimension_scores["E"],
                conventional=dimension_scores["C"],
                raisec_code=primary_code,
                secondary_code=secondary_code,
                total_score=total_score,
                consistency_score=consistency_score,
                differentiation_index=differentiation_index
            )

            logger.info(f"Scores calculated: RAISEC {primary_code}",
                       extra={"test_id": str(test_id), "raisec_code": primary_code})

            return test_scores

        except Exception as e:
            logger.error(f"Scoring failed for test {test_id}: {str(e)}",
                        extra={"test_id": str(test_id), "error": str(e)})
            raise TruScholarError(f"Failed to calculate test scores: {str(e)}")

    async def _get_test_answers(self, test_id: ObjectId) -> List[Answer]:
        """Get all answers for a test.

        Args:
            test_id: ID of the test

        Returns:
            List[Answer]: List of answers
        """
        answer_docs = await self.db.find(
            "answers",
            {"test_id": test_id},
            sort=[("question_number", 1)]
        )

        return [Answer.from_dict(doc) for doc in answer_docs]

    async def _get_test_questions(self, test_id: ObjectId) -> List[Question]:
        """Get all questions for a test.

        Args:
            test_id: ID of the test

        Returns:
            List[Question]: List of questions
        """
        question_docs = await self.db.find(
            "questions",
            {"test_id": test_id},
            sort=[("question_number", 1)]
        )

        return [Question.from_dict(doc) for doc in question_docs]

    async def _calculate_dimension_scores(
        self,
        answers: List[Answer],
        question_map: Dict[str, Question]
    ) -> Dict[RaisecDimension, float]:
        """Calculate raw dimension scores from answers.

        Args:
            answers: List of user answers
            question_map: Map of question ID to Question object

        Returns:
            Dict[RaisecDimension, float]: Raw dimension scores
        """
        dimension_scores = {dim: 0.0 for dim in RaisecDimension}

        for answer in answers:
            if answer.is_skipped:
                continue

            question = question_map.get(str(answer.question_id))
            if not question:
                logger.warning(f"Question {answer.question_id} not found for answer")
                continue

            # Calculate scores based on question type
            answer_scores = self._score_individual_answer(answer, question)

            # Add to dimension totals
            for dimension, score in answer_scores.items():
                dimension_scores[dimension] += score

        return dimension_scores

    def _score_individual_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score an individual answer.

        Args:
            answer: User's answer
            question: The question being answered

        Returns:
            Dict[RaisecDimension, float]: Dimension scores for this answer
        """
        scores = {dim: 0.0 for dim in RaisecDimension}

        if answer.question_type.value == "mcq":
            scores.update(self._score_mcq_answer(answer, question))
        elif answer.question_type.value == "statement_set":
            scores.update(self._score_statement_set_answer(answer, question))
        elif answer.question_type.value == "scenario_mcq":
            scores.update(self._score_scenario_mcq_answer(answer, question))
        elif answer.question_type.value == "scenario_multi_select":
            scores.update(self._score_multi_select_answer(answer, question))
        elif answer.question_type.value == "this_or_that":
            scores.update(self._score_this_or_that_answer(answer, question))
        elif answer.question_type.value == "scale_rating":
            scores.update(self._score_scale_rating_answer(answer, question))
        elif answer.question_type.value == "plot_day":
            scores.update(self._score_plot_day_answer(answer, question))

        return scores

    def _score_mcq_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score MCQ answer.

        Args:
            answer: User's answer
            question: MCQ question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        scores = {dim: 0.0 for dim in RaisecDimension}
        selected_option = answer.selected_option

        if not selected_option:
            return scores

        # Find selected option
        for option in question.options:
            if option.id == selected_option:
                # Apply dimension weights
                for dimension, weight in option.dimension_weights.items():
                    scores[dimension] = weight * business_settings.SINGLE_DIMENSION_POINTS
                break

        return scores

    def _score_statement_set_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score statement set (Likert scale) answer.

        Args:
            answer: User's answer
            question: Statement set question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        scores = {dim: 0.0 for dim in RaisecDimension}
        ratings = answer.ratings

        if not ratings:
            return scores

        for statement in question.statements:
            rating = ratings.get(str(statement.id))
            if rating is None:
                continue

            # Apply reverse scoring if needed
            if statement.reverse_scored:
                rating = 6 - rating

            # Convert rating to score
            score = business_settings.LIKERT_SCALE_MAP.get(rating, 5.0)
            scores[statement.dimension] += score

        return scores

    def _score_scenario_mcq_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score scenario MCQ answer.

        Args:
            answer: User's answer
            question: Scenario MCQ question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        # Similar to regular MCQ but with scenario context
        return self._score_mcq_answer(answer, question)

    def _score_multi_select_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score multi-select answer.

        Args:
            answer: User's answer
            question: Multi-select question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        scores = {dim: 0.0 for dim in RaisecDimension}
        selected_options = answer.selected_options

        if not selected_options:
            return scores

        # Score each selected option
        for option in question.options:
            if option.id in selected_options:
                for dimension, weight in option.dimension_weights.items():
                    scores[dimension] += weight * business_settings.MULTI_DIMENSION_SECONDARY_POINTS

        return scores

    def _score_this_or_that_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score this-or-that binary choice answer.

        Args:
            answer: User's answer
            question: This-or-that question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        scores = {dim: 0.0 for dim in RaisecDimension}
        selected = answer.selected_option

        if selected == "a" and question.option_a:
            for dimension, weight in question.option_a.dimension_weights.items():
                scores[dimension] = weight * business_settings.SINGLE_DIMENSION_POINTS
        elif selected == "b" and question.option_b:
            for dimension, weight in question.option_b.dimension_weights.items():
                scores[dimension] = weight * business_settings.SINGLE_DIMENSION_POINTS

        return scores

    def _score_scale_rating_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score scale rating answer.

        Args:
            answer: User's answer
            question: Scale rating question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        scores = {dim: 0.0 for dim in RaisecDimension}
        rating = answer.scale_rating

        if rating is None:
            return scores

        # Normalize rating to 0-1 scale
        normalized_rating = rating / question.scale_max
        base_score = normalized_rating * business_settings.SINGLE_DIMENSION_POINTS

        # Apply to all evaluated dimensions equally
        for dimension in question.dimensions_evaluated:
            scores[dimension] = base_score

        return scores

    def _score_plot_day_answer(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score plot day answer.

        Args:
            answer: User's answer
            question: Plot day question

        Returns:
            Dict[RaisecDimension, float]: Dimension scores
        """
        scores = {dim: 0.0 for dim in RaisecDimension}
        placements = answer.task_placements

        if not placements:
            return scores

        # Create task lookup
        task_map = {task.id: task for task in question.tasks}

        # Score each time slot
        for time_slot, task_ids in placements.items():
            if time_slot == "not_interested":
                continue

            # Get time slot weight
            slot_weight = business_settings.PLOT_DAY_TIME_WEIGHTS.get(time_slot, 1.0)

            # Score each task in this slot
            for task_id in task_ids:
                task = task_map.get(task_id)
                if not task:
                    continue

                # Primary dimension gets full points
                primary_score = business_settings.PLOT_DAY_TASK_POINTS * slot_weight
                scores[task.primary_dimension] += primary_score

                # Secondary dimensions get partial points
                for secondary_dim in task.secondary_dimensions:
                    secondary_score = primary_score * 0.5
                    scores[secondary_dim] += secondary_score

        return scores

    def _normalize_scores(
        self,
        raw_scores: Dict[RaisecDimension, float]
    ) -> Dict[RaisecDimension, float]:
        """Normalize scores to percentages.

        Args:
            raw_scores: Raw dimension scores

        Returns:
            Dict[RaisecDimension, float]: Normalized scores (0-100)
        """
        # Find maximum possible score (theoretical maximum)
        max_possible = self._calculate_theoretical_maximum()

        # Normalize each dimension
        normalized = {}
        for dimension, score in raw_scores.items():
            # Convert to percentage of theoretical maximum
            percentage = (score / max_possible) * 100
            normalized[dimension] = min(100.0, max(0.0, percentage))

        return normalized

    def _calculate_theoretical_maximum(self) -> float:
        """Calculate theoretical maximum score per dimension.

        Returns:
            float: Theoretical maximum score
        """
        # This is based on the question distribution and max points per type
        max_score = 0.0

        distribution = business_settings.QUESTION_DISTRIBUTION

        # MCQ questions: max single dimension points
        max_score += distribution.get("mcq", 0) * business_settings.SINGLE_DIMENSION_POINTS

        # Statement sets: max likert points per statement (assume 5 statements)
        max_score += distribution.get("statement_set", 0) * 5 * 10.0

        # Scenario MCQ: similar to MCQ
        max_score += distribution.get("scenario_mcq", 0) * business_settings.SINGLE_DIMENSION_POINTS

        # Multi-select: assume 3 selections
        max_score += distribution.get("scenario_multi_select", 0) * 3 * business_settings.MULTI_DIMENSION_SECONDARY_POINTS

        # This or that: single dimension points
        max_score += distribution.get("this_or_that", 0) * business_settings.SINGLE_DIMENSION_POINTS

        # Scale rating: max rating normalized
        max_score += distribution.get("scale_rating", 0) * business_settings.SINGLE_DIMENSION_POINTS

        # Plot day: assume 8 tasks in high-weight slots
        max_score += distribution.get("plot_day", 0) * 8 * business_settings.PLOT_DAY_TASK_POINTS * 1.2

        return max_score

    def _rank_dimensions(
        self,
        normalized_scores: Dict[RaisecDimension, float]
    ) -> Dict[RaisecDimension, int]:
        """Rank dimensions by score.

        Args:
            normalized_scores: Normalized dimension scores

        Returns:
            Dict[RaisecDimension, int]: Dimension rankings (1-6)
        """
        # Sort dimensions by score (descending)
        sorted_dims = sorted(
            normalized_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Assign ranks
        rankings = {}
        for rank, (dimension, _) in enumerate(sorted_dims, 1):
            rankings[dimension] = rank

        return rankings

    def _generate_raisec_code(
        self,
        rankings: Dict[RaisecDimension, int],
        count: int = 3
    ) -> str:
        """Generate RAISEC code from rankings.

        Args:
            rankings: Dimension rankings
            count: Number of dimensions to include

        Returns:
            str: RAISEC code
        """
        # Sort by rank to get top dimensions
        sorted_by_rank = sorted(rankings.items(), key=lambda x: x[1])

        # Take top N dimensions
        top_dimensions = sorted_by_rank[:count]

        # Create code string
        code = "".join([dim.value for dim, _ in top_dimensions])

        return code

    def _calculate_consistency_score(
        self,
        answers: List[Answer],
        question_map: Dict[str, Question]
    ) -> float:
        """Calculate response consistency score.

        Args:
            answers: List of answers
            question_map: Question mapping

        Returns:
            float: Consistency score (0-100)
        """
        # This would analyze consistency across similar questions
        # For now, return a basic metric based on answer patterns

        valid_answers = [a for a in answers if not a.is_skipped]
        if not valid_answers:
            return 0.0

        # Simple consistency metric: percentage of valid answers
        consistency = (len(valid_answers) / len(answers)) * 100

        # Adjust for answer changes (more changes = less consistent)
        avg_changes = sum(a.metrics.revision_count for a in valid_answers) / len(valid_answers)
        consistency -= min(avg_changes * 5, 20)  # Penalty for excessive changes

        return max(0.0, min(100.0, consistency))

    def _calculate_differentiation_index(
        self,
        normalized_scores: Dict[RaisecDimension, float]
    ) -> float:
        """Calculate score differentiation index.

        Args:
            normalized_scores: Normalized scores

        Returns:
            float: Differentiation index (0-100)
        """
        scores = list(normalized_scores.values())

        if not scores:
            return 0.0

        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5

        # Convert to 0-100 scale (higher std_dev = better differentiation)
        # Maximum possible std_dev is when one score is 100 and others are 0
        max_std_dev = (100 ** 2 * 5 / 6) ** 0.5  # Theoretical maximum
        differentiation = (std_dev / max_std_dev) * 100

        return min(100.0, max(0.0, differentiation))

    def _calculate_dimension_confidence(
        self,
        dimension: RaisecDimension,
        answers: List[Answer],
        question_map: Dict[str, Question]
    ) -> float:
        """Calculate confidence level for a dimension score.

        Args:
            dimension: RAISEC dimension
            answers: List of answers
            question_map: Question mapping

        Returns:
            float: Confidence level (0-100)
        """
        relevant_answers = []

        # Find answers that contribute to this dimension
        for answer in answers:
            if answer.is_skipped:
                continue

            question = question_map.get(str(answer.question_id))
            if not question:
                continue

            if dimension in question.dimensions_evaluated:
                relevant_answers.append(answer)

        if not relevant_answers:
            return 0.0

        # Base confidence from number of relevant questions
        base_confidence = min(100.0, (len(relevant_answers) / 3) * 100)

        # Adjust for answer quality metrics
        avg_hesitation = sum(a.metrics.hesitation_score for a in relevant_answers) / len(relevant_answers)
        confidence_penalty = avg_hesitation * 0.3

        final_confidence = base_confidence - confidence_penalty
        return max(0.0, min(100.0, final_confidence))

    def get_score_interpretation(self, score: float) -> str:
        """Get text interpretation of a score.

        Args:
            score: Dimension score (0-100)

        Returns:
            str: Score interpretation
        """
        if score >= 80:
            return "Very High"
        elif score >= 60:
            return "High"
        elif score >= 40:
            return "Moderate"
        elif score >= 20:
            return "Low"
        else:
            return "Very Low"

    def get_raisec_interpretation(self, raisec_code: str) -> Dict[str, str]:
        """Get interpretation of RAISEC code.

        Args:
            raisec_code: 3-letter RAISEC code

        Returns:
            Dict[str, str]: Code interpretation
        """
        dimension_names = {
            "R": "Realistic",
            "A": "Artistic",
            "I": "Investigative",
            "S": "Social",
            "E": "Enterprising",
            "C": "Conventional"
        }

        if len(raisec_code) != 3:
            return {"error": "Invalid RAISEC code"}

        return {
            "primary": dimension_names.get(raisec_code[0], "Unknown"),
            "secondary": dimension_names.get(raisec_code[1], "Unknown"),
            "tertiary": dimension_names.get(raisec_code[2], "Unknown"),
            "full_code": raisec_code
        }
