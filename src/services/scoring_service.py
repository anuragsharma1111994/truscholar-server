"""Scoring service for RAISEC assessment.

This service handles the complete scoring algorithm for RAISEC personality assessments,
including individual answer scoring, test-level aggregation, consistency validation,
and final score calculation with confidence metrics.
"""

import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from bson import ObjectId

from src.core.config import get_settings
from src.database.mongodb import MongoDB
from src.database.redis_client import RedisClient
from src.cache.cache_manager import CacheManager
from src.cache.cache_keys import CacheKeys
from src.models.answer import Answer, DimensionScore
from src.models.question import Question
from src.models.test import Test, TestScores
from src.utils.constants import QuestionType, RaisecDimension, TestStatus
from src.utils.exceptions import TruScholarError, ValidationError
from src.utils.logger import get_logger
from src.utils.datetime_utils import (
    calculate_duration_minutes,
    is_within_business_hours,
    get_time_zone_for_location,
    normalize_datetime_to_utc
)

settings = get_settings()
logger = get_logger(__name__)


class ScoringService:
    """Service for comprehensive RAISEC assessment scoring."""
    
    def __init__(self, db=None, cache=None):
        """Initialize scoring service.
        
        Args:
            db: Database instance (defaults to MongoDB)
            cache: Cache instance (defaults to RedisClient)
        """
        self.db = db or MongoDB
        self.cache = cache or RedisClient
        self.cache_manager = CacheManager(self.cache)
        
        # Scoring configuration
        self.scoring_config = self._load_scoring_config()
        
        # Question type weights for balanced scoring
        self.question_weights = {
            QuestionType.MCQ: 1.0,
            QuestionType.STATEMENT_SET: 0.8,
            QuestionType.SCENARIO_MCQ: 1.0,
            QuestionType.SCENARIO_MULTI_SELECT: 0.6,
            QuestionType.THIS_OR_THAT: 1.2,
            QuestionType.SCALE_RATING: 0.9,
            QuestionType.PLOT_DAY: 1.5
        }
        
        # Time-based adjustment factors
        self.time_adjustment_factors = {
            "very_fast": 0.85,    # < 50% of expected time
            "fast": 0.92,         # 50-75% of expected time
            "normal": 1.0,        # 75-125% of expected time
            "slow": 1.05,         # 125-150% of expected time
            "very_slow": 0.95     # > 150% of expected time
        }
    
    async def score_test(
        self,
        test_id: ObjectId,
        force_rescore: bool = False,
        include_analytics: bool = True
    ) -> TestScores:
        """Score a complete test and generate RAISEC profile.
        
        Args:
            test_id: Test ID to score
            force_rescore: Whether to force rescoring even if already scored
            include_analytics: Whether to include detailed analytics
            
        Returns:
            TestScores: Complete test scoring results
            
        Raises:
            TruScholarError: If scoring fails
            ValidationError: If test data is invalid
        """
        logger.info(f"Starting test scoring for test_id: {test_id}")
        
        try:
            # Get test and validate status
            test = await self._get_test_for_scoring(test_id, force_rescore)
            
            # Get all answers for the test
            answers = await self._get_test_answers(test_id)
            
            # Get all questions for the test
            questions = await self._get_test_questions(test_id)
            
            # Validate completeness
            self._validate_test_completeness(test, answers, questions)
            
            # Score individual answers
            scored_answers = await self._score_all_answers(answers, questions)
            
            # Calculate dimension aggregates
            dimension_scores = self._calculate_dimension_scores(scored_answers, questions)
            
            # Generate RAISEC profile
            raisec_profile = self._generate_raisec_profile(dimension_scores, test)
            
            # Calculate consistency metrics
            consistency_analysis = self._analyze_consistency(scored_answers, questions)
            
            # Generate final score analysis
            score_analysis = self._generate_score_analysis(
                dimension_scores, consistency_analysis, test, scored_answers
            )
            
            # Include detailed analytics if requested
            analytics = None
            if include_analytics:
                analytics = await self._generate_analytics(
                    scored_answers, questions, test, dimension_scores
                )
            
            # Create TestScores object
            test_scores = TestScores(
                test_id=test_id,
                user_id=test.user_id,
                raisec_profile=raisec_profile,
                dimension_scores=dimension_scores,
                score_analysis=score_analysis,
                consistency_score=consistency_analysis["overall_consistency"],
                completion_metrics={
                    "questions_answered": len(scored_answers),
                    "questions_skipped": len([a for a in scored_answers if a.is_skipped]),
                    "total_time_minutes": calculate_duration_minutes(
                        test.created_at, 
                        max((a.metrics.final_submission_timestamp for a in scored_answers 
                            if a.metrics.final_submission_timestamp), default=datetime.utcnow())
                    ),
                    "average_time_per_question": sum(
                        a.metrics.total_time_seconds for a in scored_answers
                    ) / len(scored_answers) if scored_answers else 0,
                    "confidence_score": statistics.mean([
                        statistics.mean([ds.confidence for ds in a.dimension_scores]) 
                        for a in scored_answers if a.dimension_scores
                    ]) if any(a.dimension_scores for a in scored_answers) else 100.0
                },
                analytics=analytics,
                scoring_metadata={
                    "scoring_version": "v2.0",
                    "scoring_algorithm": "weighted_dimensional_analysis",
                    "scored_at": datetime.utcnow(),
                    "scorer": "automated",
                    "force_rescore": force_rescore,
                    "question_weights": self.question_weights,
                    "consistency_threshold": self.scoring_config["consistency_threshold"]
                }
            )
            
            # Save scores to database
            await self._save_test_scores(test_scores)
            
            # Update test status
            await self._update_test_status(test_id, TestStatus.COMPLETED, test_scores)
            
            # Cache results
            await self._cache_test_scores(test_scores)
            
            logger.info(
                f"Test scoring completed successfully",
                extra={
                    "test_id": str(test_id),
                    "raisec_code": raisec_profile.code,
                    "total_score": raisec_profile.total_score,
                    "consistency": consistency_analysis["overall_consistency"]
                }
            )
            
            return test_scores
            
        except Exception as e:
            logger.error(
                f"Test scoring failed for test_id: {test_id}",
                extra={"test_id": str(test_id), "error": str(e)}
            )
            raise TruScholarError(f"Test scoring failed: {str(e)}")
    
    # Legacy method for backwards compatibility
    async def calculate_test_scores(self, test_id: ObjectId) -> TestScores:
        """Legacy method - delegates to score_test."""
        return await self.score_test(test_id, force_rescore=False, include_analytics=False)
    
    async def score_single_answer(
        self,
        answer: Answer,
        question: Question,
        test_context: Optional[Dict[str, Any]] = None
    ) -> Answer:
        """Score a single answer and update dimension scores.
        
        Args:
            answer: Answer to score
            question: Associated question
            test_context: Optional test context for enhanced scoring
            
        Returns:
            Answer: Updated answer with scores
        """
        logger.debug(f"Scoring answer for question {answer.question_number}")
        
        try:
            # Skip scoring if already scored and not forced
            if answer.is_scored and not test_context.get("force_rescore", False):
                return answer
            
            # Validate answer completeness
            if not answer.validate_answer_content():
                logger.warning(f"Answer validation failed for question {answer.question_number}")
                return answer
            
            # Calculate raw dimension scores based on question type
            raw_scores = self._calculate_raw_dimension_scores(answer, question)
            
            # Apply question type weighting
            question_weight = self.question_weights.get(question.question_type, 1.0)
            
            # Apply time-based adjustments
            time_factor = self._calculate_time_adjustment_factor(answer, question)
            
            # Apply consistency bonuses/penalties if available
            consistency_factor = 1.0
            if test_context and "consistency_patterns" in test_context:
                consistency_factor = self._calculate_consistency_factor(
                    answer, test_context["consistency_patterns"]
                )
            
            # Create dimension score objects
            dimension_scores = []
            total_weighted_score = 0.0
            
            for dimension, raw_score in raw_scores.items():
                # Apply all adjustment factors
                adjusted_score = raw_score * question_weight * time_factor * consistency_factor
                
                # Calculate confidence based on answer behavior
                confidence = self._calculate_answer_confidence(answer, question)
                
                # Final weighted score
                weighted_score = adjusted_score * (confidence / 100.0)
                total_weighted_score += weighted_score
                
                dimension_score = DimensionScore(
                    dimension=dimension,
                    raw_score=raw_score,
                    weighted_score=weighted_score,
                    confidence=confidence,
                    contribution_percentage=0.0  # Will be calculated after all scores
                )
                dimension_scores.append(dimension_score)
            
            # Calculate contribution percentages
            if total_weighted_score > 0:
                for score in dimension_scores:
                    score.contribution_percentage = (score.weighted_score / total_weighted_score) * 100
            
            # Update answer with scores
            answer.dimension_scores = dimension_scores
            answer.total_points = total_weighted_score
            answer.is_scored = True
            answer.scored_at = datetime.utcnow()
            answer.scoring_version = "v2.0"
            
            return answer
            
        except Exception as e:
            logger.error(
                f"Failed to score answer for question {answer.question_number}: {str(e)}"
            )
            raise TruScholarError(f"Answer scoring failed: {str(e)}")
    
    async def recalculate_test_scores(
        self,
        test_id: ObjectId,
        scoring_adjustments: Optional[Dict[str, Any]] = None
    ) -> TestScores:
        """Recalculate test scores with optional adjustments.
        
        Args:
            test_id: Test ID to recalculate
            scoring_adjustments: Optional scoring parameter adjustments
            
        Returns:
            TestScores: Recalculated scores
        """
        logger.info(f"Recalculating scores for test_id: {test_id}")
        
        # Apply scoring adjustments if provided
        if scoring_adjustments:
            self._apply_scoring_adjustments(scoring_adjustments)
        
        try:
            # Force rescore with current configuration
            return await self.score_test(test_id, force_rescore=True)
        finally:
            # Reset to original configuration
            if scoring_adjustments:
                self.scoring_config = self._load_scoring_config()
    
    async def get_scoring_explanation(
        self,
        test_id: ObjectId,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """Get detailed explanation of how scores were calculated.
        
        Args:
            test_id: Test ID to explain
            detailed: Whether to include detailed breakdowns
            
        Returns:
            Dict containing scoring explanation
        """
        # Get test scores
        test_scores = await self._get_cached_test_scores(test_id)
        if not test_scores:
            test_scores = await self.score_test(test_id)
        
        # Get answers and questions for detailed analysis
        answers = await self._get_test_answers(test_id)
        questions = await self._get_test_questions(test_id)
        
        explanation = {
            "test_id": str(test_id),
            "raisec_code": test_scores.raisec_profile.code,
            "total_score": test_scores.raisec_profile.total_score,
            "scoring_methodology": {
                "algorithm": "Weighted Dimensional Analysis",
                "version": "v2.0",
                "description": "Multi-factor scoring considering answer accuracy, consistency, timing, and confidence"
            },
            "dimension_breakdown": {},
            "scoring_factors": {
                "question_type_weights": self.question_weights,
                "time_adjustment_factors": self.time_adjustment_factors,
                "consistency_bonus_range": "±10%",
                "confidence_impact": "50-100% of raw score"
            }
        }
        
        # Add dimension-by-dimension breakdown
        for dimension, scores in test_scores.dimension_scores.items():
            explanation["dimension_breakdown"][dimension.value] = {
                "final_score": scores["total_score"],
                "percentile": scores.get("percentile", 0),
                "contributing_questions": scores.get("question_count", 0),
                "consistency": scores.get("consistency", 100),
                "confidence": scores.get("average_confidence", 100)
            }
        
        if detailed:
            # Add question-by-question breakdown
            explanation["question_details"] = []
            
            for answer in answers:
                question = next((q for q in questions if q.id == answer.question_id), None)
                if question and answer.dimension_scores:
                    question_detail = {
                        "question_number": answer.question_number,
                        "question_type": answer.question_type.value,
                        "time_spent": answer.metrics.total_time_seconds,
                        "confidence_factors": {
                            "revision_count": answer.metrics.revision_count,
                            "hesitation_score": answer.metrics.hesitation_score,
                            "time_factor": self._calculate_time_adjustment_factor(answer, question)
                        },
                        "dimension_contributions": [
                            {
                                "dimension": ds.dimension.value,
                                "raw_score": ds.raw_score,
                                "weighted_score": ds.weighted_score,
                                "confidence": ds.confidence,
                                "contribution_percent": ds.contribution_percentage
                            }
                            for ds in answer.dimension_scores
                        ]
                    }
                    explanation["question_details"].append(question_detail)
        
        return explanation
    
    def _load_scoring_config(self) -> Dict[str, Any]:
        """Load scoring configuration from business settings."""
        return {
            "base_points_per_question": 10.0,
            "consistency_threshold": 0.7,
            "minimum_confidence": 50.0,
            "maximum_time_bonus": 1.15,
            "minimum_time_penalty": 0.85,
            "dimension_balance_bonus": 0.05,
            "plot_day_time_slot_weights": {
                "9:00-12:00": 1.2,    # Morning peak
                "12:00-15:00": 1.0,   # Standard
                "15:00-18:00": 1.1,   # Afternoon focus
                "18:00-21:00": 0.9,   # Evening lower
                "not_interested": 0.0
            },
            "likert_scale_mapping": {
                1: 0.0,   # Strongly disagree
                2: 2.5,   # Disagree
                3: 5.0,   # Neutral
                4: 7.5,   # Agree
                5: 10.0   # Strongly agree
            },
            "scale_rating_normalization": {
                "min_scale": 1,
                "max_scale": 10,
                "base_points": 10.0
            }
        }
    
    async def _get_test_for_scoring(self, test_id: ObjectId, force_rescore: bool) -> Test:
        """Get and validate test for scoring."""
        test_data = await self.db.find_one("tests", {"_id": test_id})
        if not test_data:
            raise ValidationError(f"Test {test_id} not found")
        
        test = Test.from_dict(test_data)
        
        # Check if already scored
        if test.scores and not force_rescore:
            raise ValidationError(f"Test {test_id} already scored. Use force_rescore=True to override.")
        
        # Check if test is ready for scoring
        if test.status not in [TestStatus.IN_PROGRESS, TestStatus.SUBMITTED]:
            raise ValidationError(f"Test {test_id} status {test.status} is not ready for scoring")
        
        return test
    
    async def _get_test_answers(self, test_id: ObjectId) -> List[Answer]:
        """Get all answers for a test."""
        answer_docs = await self.db.find(
            "answers",
            {"test_id": test_id},
            sort=[("question_number", 1)]
        )
        
        return [Answer.from_dict(doc) for doc in answer_docs]
    
    async def _get_test_questions(self, test_id: ObjectId) -> List[Question]:
        """Get all questions for a test."""
        question_docs = await self.db.find(
            "questions",
            {"test_id": test_id},
            sort=[("question_number", 1)]
        )
        
        return [Question.from_dict(doc) for doc in question_docs]
    
    def _validate_test_completeness(
        self,
        test: Test,
        answers: List[Answer],
        questions: List[Question]
    ) -> None:
        """Validate test completeness for scoring."""
        # Check minimum completion requirements
        answered_count = len([a for a in answers if not a.is_skipped])
        skipped_count = len([a for a in answers if a.is_skipped])
        
        minimum_required = 10  # Minimum questions required for scoring
        
        if answered_count < minimum_required:
            raise ValidationError(
                f"Insufficient answers for scoring. "
                f"Answered: {answered_count}, Required: {minimum_required}"
            )
        
        # Check that we have questions for all answers
        question_ids = {q.id for q in questions}
        answer_question_ids = {a.question_id for a in answers}
        
        missing_questions = answer_question_ids - question_ids
        if missing_questions:
            raise ValidationError(f"Missing questions for answers: {missing_questions}")
        
        logger.info(
            f"Test completeness validated",
            extra={
                "test_id": str(test.id),
                "answered": answered_count,
                "skipped": skipped_count,
                "total_questions": len(questions)
            }
        )
    
    async def _score_all_answers(
        self,
        answers: List[Answer],
        questions: List[Question]
    ) -> List[Answer]:
        """Score all answers in a test."""
        # Create question lookup
        question_map = {q.id: q for q in questions}
        
        # Analyze consistency patterns for enhanced scoring
        consistency_patterns = self._analyze_answer_patterns(answers)
        
        scored_answers = []
        
        for answer in answers:
            question = question_map.get(answer.question_id)
            if not question:
                logger.warning(f"Question not found for answer {answer.id}")
                continue
            
            # Score with test context
            test_context = {
                "consistency_patterns": consistency_patterns,
                "force_rescore": True
            }
            
            scored_answer = await self.score_single_answer(answer, question, test_context)
            scored_answers.append(scored_answer)
        
        return scored_answers
    
    def _calculate_raw_dimension_scores(
        self,
        answer: Answer,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Calculate raw dimension scores for an answer."""
        scores = {}
        base_points = self.scoring_config["base_points_per_question"]
        
        if answer.is_skipped:
            # Skipped questions get 0 points
            return {dim: 0.0 for dim in question.dimensions_evaluated}
        
        if question.question_type == QuestionType.MCQ:
            # Single option selection
            if answer.selected_option and question.options:
                for option in question.options:
                    if option.id == answer.selected_option:
                        for dimension, weight in option.dimension_weights.items():
                            scores[dimension] = weight * base_points
                        break
        
        elif question.question_type == QuestionType.STATEMENT_SET:
            # Likert scale ratings
            likert_map = self.scoring_config["likert_scale_mapping"]
            
            if answer.ratings and question.statements:
                for statement in question.statements:
                    statement_id = str(statement.id)
                    if statement_id in answer.ratings:
                        rating = answer.ratings[statement_id]
                        points = likert_map.get(rating, 5.0)
                        
                        # Handle reverse scoring
                        if statement.reverse_scored:
                            points = 10.0 - points
                        
                        if statement.dimension in scores:
                            scores[statement.dimension] += points
                        else:
                            scores[statement.dimension] = points
        
        elif question.question_type == QuestionType.SCENARIO_MCQ:
            # Similar to MCQ but with scenario context
            if answer.selected_option and question.options:
                for option in question.options:
                    if option.id == answer.selected_option:
                        for dimension, weight in option.dimension_weights.items():
                            scores[dimension] = weight * base_points
                        break
        
        elif question.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            # Multiple selections with partial scoring
            if answer.selected_options and question.options:
                total_points = base_points / len(answer.selected_options)
                
                for option in question.options:
                    if option.id in answer.selected_options:
                        for dimension, weight in option.dimension_weights.items():
                            points = weight * total_points
                            if dimension in scores:
                                scores[dimension] += points
                            else:
                                scores[dimension] = points
        
        elif question.question_type == QuestionType.THIS_OR_THAT:
            # Binary choice
            if answer.selected_option == "a" and question.option_a:
                for dimension, weight in question.option_a.dimension_weights.items():
                    scores[dimension] = weight * base_points
            elif answer.selected_option == "b" and question.option_b:
                for dimension, weight in question.option_b.dimension_weights.items():
                    scores[dimension] = weight * base_points
        
        elif question.question_type == QuestionType.SCALE_RATING:
            # Scale rating (1-10)
            if answer.scale_rating:
                config = self.scoring_config["scale_rating_normalization"]
                normalized = (answer.scale_rating - config["min_scale"]) / (config["max_scale"] - config["min_scale"])
                points = normalized * config["base_points"]
                
                # Apply to all evaluated dimensions equally
                for dimension in question.dimensions_evaluated:
                    scores[dimension] = points
        
        elif question.question_type == QuestionType.PLOT_DAY:
            # Task placement with time slot weighting
            if answer.task_placements and question.tasks:
                slot_weights = self.scoring_config["plot_day_time_slot_weights"]
                task_points = base_points / len(question.tasks)
                
                # Create task lookup
                task_map = {task.id: task for task in question.tasks}
                
                for slot, task_ids in answer.task_placements.items():
                    slot_weight = slot_weights.get(slot, 1.0)
                    
                    for task_id in task_ids:
                        task = task_map.get(task_id)
                        if task:
                            points = task_points * slot_weight
                            
                            # Primary dimension gets full points
                            if task.primary_dimension in scores:
                                scores[task.primary_dimension] += points
                            else:
                                scores[task.primary_dimension] = points
                            
                            # Secondary dimensions get partial points
                            if task.secondary_dimensions:
                                secondary_points = points * 0.3
                                for dim in task.secondary_dimensions:
                                    if dim in scores:
                                        scores[dim] += secondary_points
                                    else:
                                        scores[dim] = secondary_points
        
        return scores
    
    def _calculate_time_adjustment_factor(self, answer: Answer, question: Question) -> float:
        """Calculate time-based adjustment factor for answer scoring."""
        if not answer.metrics.total_time_seconds:
            return 1.0
        
        # Get expected time for question type
        expected_times = {
            QuestionType.MCQ: 30,
            QuestionType.STATEMENT_SET: 90,
            QuestionType.SCENARIO_MCQ: 45,
            QuestionType.SCENARIO_MULTI_SELECT: 60,
            QuestionType.THIS_OR_THAT: 20,
            QuestionType.SCALE_RATING: 25,
            QuestionType.PLOT_DAY: 180
        }
        
        expected_time = expected_times.get(question.question_type, 45)
        actual_time = answer.metrics.total_time_seconds
        time_ratio = actual_time / expected_time
        
        # Determine time category and apply factor
        if time_ratio < 0.5:
            return self.time_adjustment_factors["very_fast"]
        elif time_ratio < 0.75:
            return self.time_adjustment_factors["fast"]
        elif time_ratio <= 1.25:
            return self.time_adjustment_factors["normal"]
        elif time_ratio <= 1.5:
            return self.time_adjustment_factors["slow"]
        else:
            return self.time_adjustment_factors["very_slow"]
    
    def _calculate_answer_confidence(self, answer: Answer, question: Question) -> float:
        """Calculate confidence score for an answer."""
        base_confidence = 100.0
        
        # Reduce confidence for excessive revisions
        if answer.metrics.revision_count > 2:
            base_confidence -= min(20, (answer.metrics.revision_count - 2) * 5)
        
        # Reduce confidence for high hesitation
        if answer.metrics.hesitation_score > 50:
            base_confidence -= (answer.metrics.hesitation_score - 50) * 0.3
        
        # Reduce confidence for validation errors
        if not answer.validation.is_valid:
            base_confidence -= 30
        elif answer.validation.warnings:
            base_confidence -= len(answer.validation.warnings) * 5
        
        # Adjust for incomplete responses
        if answer.validation.completeness_score < 100:
            base_confidence *= (answer.validation.completeness_score / 100)
        
        return max(base_confidence, self.scoring_config["minimum_confidence"])
    
    def _calculate_consistency_factor(
        self,
        answer: Answer,
        consistency_patterns: Dict[str, Any]
    ) -> float:
        """Calculate consistency-based adjustment factor."""
        base_factor = 1.0
        
        # Get patterns for this question type
        type_patterns = consistency_patterns.get(answer.question_type.value, {})
        
        # Check time consistency
        time_consistency = type_patterns.get("time_consistency", 1.0)
        if time_consistency < 0.7:  # Highly inconsistent timing
            base_factor *= 0.95
        elif time_consistency > 0.9:  # Very consistent timing
            base_factor *= 1.05
        
        # Check answer pattern consistency
        answer_consistency = type_patterns.get("answer_consistency", 1.0)
        if answer_consistency > 0.9:  # Very consistent answers
            base_factor *= 1.03
        
        return min(max(base_factor, 0.9), 1.1)  # Cap between 0.9 and 1.1
    
    def _analyze_answer_patterns(self, answers: List[Answer]) -> Dict[str, Any]:
        """Analyze patterns in answers for consistency scoring."""
        patterns = {}
        
        # Group answers by type
        by_type = defaultdict(list)
        for answer in answers:
            by_type[answer.question_type.value].append(answer)
        
        for q_type, type_answers in by_type.items():
            if len(type_answers) < 2:
                continue
            
            # Analyze time consistency
            times = [a.metrics.total_time_seconds for a in type_answers if a.metrics.total_time_seconds > 0]
            if times:
                time_cv = statistics.stdev(times) / statistics.mean(times) if len(times) > 1 else 0
                time_consistency = max(0, 1 - time_cv)  # Lower CV = higher consistency
            else:
                time_consistency = 1.0
            
            # Analyze revision consistency
            revisions = [a.metrics.revision_count for a in type_answers]
            revision_consistency = 1.0 - (statistics.stdev(revisions) / (statistics.mean(revisions) + 1))
            
            patterns[q_type] = {
                "time_consistency": time_consistency,
                "answer_consistency": revision_consistency,
                "count": len(type_answers)
            }
        
        return patterns
    
    def _calculate_dimension_scores(
        self,
        scored_answers: List[Answer],
        questions: List[Question]
    ) -> Dict[RaisecDimension, Dict[str, Any]]:
        """Calculate final dimension scores from all answers."""
        dimension_totals = defaultdict(list)
        dimension_details = defaultdict(lambda: {
            "raw_scores": [],
            "weighted_scores": [],
            "confidences": [],
            "question_count": 0,
            "question_types": []
        })
        
        # Collect all dimension scores
        for answer in scored_answers:
            for dim_score in answer.dimension_scores:
                dimension = dim_score.dimension
                
                dimension_totals[dimension].append(dim_score.weighted_score)
                dimension_details[dimension]["raw_scores"].append(dim_score.raw_score)
                dimension_details[dimension]["weighted_scores"].append(dim_score.weighted_score)
                dimension_details[dimension]["confidences"].append(dim_score.confidence)
                dimension_details[dimension]["question_count"] += 1
                dimension_details[dimension]["question_types"].append(answer.question_type.value)
        
        # Calculate final scores for each dimension
        final_scores = {}
        all_totals = []
        
        for dimension in RaisecDimension:
            if dimension in dimension_totals:
                scores = dimension_totals[dimension]
                details = dimension_details[dimension]
                
                total_score = sum(scores)
                average_score = statistics.mean(scores)
                confidence = statistics.mean(details["confidences"])
                
                # Calculate consistency within dimension
                consistency = 100.0
                if len(scores) > 1:
                    score_cv = statistics.stdev(scores) / statistics.mean(scores)
                    consistency = max(0, 100 - (score_cv * 50))
                
                final_scores[dimension] = {
                    "total_score": total_score,
                    "average_score": average_score,
                    "question_count": details["question_count"],
                    "consistency": consistency,
                    "confidence": confidence,
                    "percentile": 0,  # Will be calculated after all dimensions
                    "question_types": list(set(details["question_types"]))
                }
                
                all_totals.append(total_score)
            else:
                # No scores for this dimension
                final_scores[dimension] = {
                    "total_score": 0.0,
                    "average_score": 0.0,
                    "question_count": 0,
                    "consistency": 0.0,
                    "confidence": 0.0,
                    "percentile": 0,
                    "question_types": []
                }
        
        # Calculate percentiles
        if all_totals:
            max_score = max(all_totals)
            for dimension in final_scores:
                if max_score > 0:
                    final_scores[dimension]["percentile"] = (
                        final_scores[dimension]["total_score"] / max_score
                    ) * 100
        
        return final_scores
    
    def _generate_raisec_profile(
        self,
        dimension_scores: Dict[RaisecDimension, Dict[str, Any]],
        test: Test
    ) -> Dict[str, Any]:
        """Generate RAISEC profile from dimension scores."""
        # Sort dimensions by total score
        sorted_dimensions = sorted(
            dimension_scores.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )
        
        # Generate RAISEC code (top 3 dimensions)
        top_three = [dim.value for dim, _ in sorted_dimensions[:3]]
        raisec_code = "".join(top_three)
        
        # Calculate total score
        total_score = sum(scores["total_score"] for scores in dimension_scores.values())
        
        # Determine profile type based on score distribution
        score_values = [scores["total_score"] for scores in dimension_scores.values()]
        score_std = statistics.stdev(score_values) if len(score_values) > 1 else 0
        score_mean = statistics.mean(score_values)
        
        if score_std / score_mean < 0.3:  # Low variation
            profile_type = "balanced"
        elif sorted_dimensions[0][1]["total_score"] > score_mean * 1.5:  # One dominant
            profile_type = "specialist"
        else:
            profile_type = "moderate"
        
        # Create primary, secondary, tertiary dimension info
        primary_dim = sorted_dimensions[0] if sorted_dimensions else (RaisecDimension.REALISTIC, {"total_score": 0})
        secondary_dim = sorted_dimensions[1] if len(sorted_dimensions) > 1 else (RaisecDimension.ARTISTIC, {"total_score": 0})
        tertiary_dim = sorted_dimensions[2] if len(sorted_dimensions) > 2 else (RaisecDimension.INVESTIGATIVE, {"total_score": 0})
        
        return {
            "code": raisec_code,
            "primary_dimension": {
                "dimension": primary_dim[0],
                "score": primary_dim[1]["total_score"],
                "percentage": primary_dim[1]["percentile"],
                "description": self._get_dimension_description(primary_dim[0])
            },
            "secondary_dimension": {
                "dimension": secondary_dim[0],
                "score": secondary_dim[1]["total_score"],
                "percentage": secondary_dim[1]["percentile"],
                "description": self._get_dimension_description(secondary_dim[0])
            },
            "tertiary_dimension": {
                "dimension": tertiary_dim[0],
                "score": tertiary_dim[1]["total_score"],
                "percentage": tertiary_dim[1]["percentile"],
                "description": self._get_dimension_description(tertiary_dim[0])
            },
            "total_score": total_score,
            "profile_type": profile_type,
            "score_distribution": {dim.value: scores["total_score"] for dim, scores in dimension_scores.items()},
            "confidence_level": statistics.mean([
                scores["confidence"] for scores in dimension_scores.values()
                if scores["confidence"] > 0
            ]) if any(scores["confidence"] > 0 for scores in dimension_scores.values()) else 100.0
        }
    
    def _analyze_consistency(
        self,
        scored_answers: List[Answer],
        questions: List[Question]
    ) -> Dict[str, Any]:
        """Analyze answer consistency across the test."""
        consistency_metrics = {
            "overall_consistency": 100.0,
            "time_consistency": 100.0,
            "response_consistency": 100.0,
            "dimension_consistency": 100.0,
            "internal_consistency": 100.0,
            "consistency_flags": []
        }
        
        if len(scored_answers) < 3:
            return consistency_metrics
        
        # Time consistency analysis
        times = [a.metrics.total_time_seconds for a in scored_answers if a.metrics.total_time_seconds > 0]
        if len(times) > 1:
            time_cv = statistics.stdev(times) / statistics.mean(times)
            consistency_metrics["time_consistency"] = max(0, 100 - (time_cv * 100))
            
            if time_cv > 1.0:  # High time variation
                consistency_metrics["consistency_flags"].append("high_time_variation")
        
        # Response pattern consistency
        revision_counts = [a.metrics.revision_count for a in scored_answers]
        if len(revision_counts) > 1:
            revision_cv = statistics.stdev(revision_counts) / (statistics.mean(revision_counts) + 1)
            consistency_metrics["response_consistency"] = max(0, 100 - (revision_cv * 50))
            
            if statistics.mean(revision_counts) > 3:
                consistency_metrics["consistency_flags"].append("high_revision_pattern")
        
        # Dimension score consistency
        dimension_variations = []
        for dimension in RaisecDimension:
            dim_scores = []
            for answer in scored_answers:
                for ds in answer.dimension_scores:
                    if ds.dimension == dimension:
                        dim_scores.append(ds.weighted_score)
            
            if len(dim_scores) > 1:
                dim_cv = statistics.stdev(dim_scores) / statistics.mean(dim_scores)
                dimension_variations.append(dim_cv)
        
        if dimension_variations:
            avg_dim_cv = statistics.mean(dimension_variations)
            consistency_metrics["dimension_consistency"] = max(0, 100 - (avg_dim_cv * 100))
        
        # Calculate overall consistency
        consistency_scores = [
            consistency_metrics["time_consistency"],
            consistency_metrics["response_consistency"], 
            consistency_metrics["dimension_consistency"]
        ]
        consistency_metrics["overall_consistency"] = statistics.mean(consistency_scores)
        
        # Add flags for low consistency
        if consistency_metrics["overall_consistency"] < 70:
            consistency_metrics["consistency_flags"].append("low_overall_consistency")
        
        return consistency_metrics
    
    def _generate_score_analysis(
        self,
        dimension_scores: Dict[RaisecDimension, Dict[str, Any]],
        consistency_analysis: Dict[str, Any],
        test: Test,
        scored_answers: List[Answer]
    ) -> Dict[str, Any]:
        """Generate comprehensive score analysis."""
        # Calculate score statistics
        all_scores = [scores["total_score"] for scores in dimension_scores.values()]
        score_mean = statistics.mean(all_scores)
        score_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
        
        # Determine score pattern
        if score_std / score_mean < 0.2:
            score_pattern = "uniform"
        elif score_std / score_mean > 0.8:
            score_pattern = "polarized"
        else:
            score_pattern = "moderate"
        
        # Calculate completion quality
        answered_count = len([a for a in scored_answers if not a.is_skipped])
        total_count = len(scored_answers)
        completion_rate = (answered_count / total_count) * 100 if total_count > 0 else 0
        
        # Average confidence
        all_confidences = []
        for answer in scored_answers:
            if answer.dimension_scores:
                all_confidences.extend([ds.confidence for ds in answer.dimension_scores])
        
        avg_confidence = statistics.mean(all_confidences) if all_confidences else 100.0
        
        return {
            "score_pattern": score_pattern,
            "score_range": {
                "min": min(all_scores),
                "max": max(all_scores),
                "mean": score_mean,
                "std": score_std
            },
            "completion_quality": {
                "completion_rate": completion_rate,
                "average_confidence": avg_confidence,
                "consistency_score": consistency_analysis["overall_consistency"],
                "quality_flags": self._identify_quality_flags(scored_answers, consistency_analysis)
            },
            "reliability_indicators": {
                "internal_consistency": consistency_analysis["overall_consistency"],
                "response_stability": consistency_analysis["response_consistency"],
                "time_stability": consistency_analysis["time_consistency"],
                "cross_validation_score": self._calculate_cross_validation_score(dimension_scores)
            },
            "interpretation_notes": [
                f"Profile shows {score_pattern} score distribution",
                f"Completion rate: {completion_rate:.1f}%",
                f"Overall consistency: {consistency_analysis['overall_consistency']:.1f}%",
                f"Average confidence: {avg_confidence:.1f}%"
            ]
        }
    
    def _get_dimension_description(self, dimension: RaisecDimension) -> str:
        """Get description for a RAISEC dimension."""
        descriptions = {
            RaisecDimension.REALISTIC: "Practical, hands-on, mechanical, and outdoor-oriented",
            RaisecDimension.INVESTIGATIVE: "Analytical, intellectual, and research-oriented",
            RaisecDimension.ARTISTIC: "Creative, expressive, and aesthetic-oriented",
            RaisecDimension.SOCIAL: "People-focused, helping, and interpersonal-oriented",
            RaisecDimension.ENTERPRISING: "Leadership, persuasive, and business-oriented",
            RaisecDimension.CONVENTIONAL: "Organized, systematic, and detail-oriented"
        }
        return descriptions.get(dimension, "")
    
    def _identify_quality_flags(
        self,
        scored_answers: List[Answer],
        consistency_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify quality issues with the assessment."""
        flags = []
        
        # Check for excessive skipping
        skip_rate = len([a for a in scored_answers if a.is_skipped]) / len(scored_answers)
        if skip_rate > 0.3:
            flags.append("high_skip_rate")
        
        # Check for very fast completion
        avg_time = statistics.mean([
            a.metrics.total_time_seconds for a in scored_answers 
            if a.metrics.total_time_seconds > 0
        ]) if any(a.metrics.total_time_seconds > 0 for a in scored_answers) else 0
        
        if avg_time < 15:  # Less than 15 seconds per question on average
            flags.append("very_fast_completion")
        
        # Check for excessive revisions
        avg_revisions = statistics.mean([a.metrics.revision_count for a in scored_answers])
        if avg_revisions > 5:
            flags.append("high_revision_pattern")
        
        # Add consistency flags
        flags.extend(consistency_analysis.get("consistency_flags", []))
        
        return flags
    
    def _calculate_cross_validation_score(
        self,
        dimension_scores: Dict[RaisecDimension, Dict[str, Any]]
    ) -> float:
        """Calculate cross-validation score for reliability."""
        # This would typically involve comparing scores across different question types
        # For now, using a simplified approach based on internal consistency
        
        consistencies = [scores["consistency"] for scores in dimension_scores.values()]
        return statistics.mean(consistencies) if consistencies else 100.0
    
    async def _generate_analytics(
        self,
        scored_answers: List[Answer],
        questions: List[Question],
        test: Test,
        dimension_scores: Dict[RaisecDimension, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate detailed analytics for the test."""
        return {
            "answer_distribution": self._analyze_answer_distribution(scored_answers),
            "time_analysis": self._analyze_time_patterns(scored_answers),
            "behavior_patterns": self._analyze_behavior_patterns(scored_answers),
            "question_performance": self._analyze_question_performance(scored_answers, questions),
            "comparative_metrics": await self._get_comparative_metrics(test, dimension_scores)
        }
    
    def _analyze_answer_distribution(self, scored_answers: List[Answer]) -> Dict[str, Any]:
        """Analyze distribution of answer types and patterns."""
        distribution = {
            "by_question_type": defaultdict(int),
            "by_confidence_level": defaultdict(int),
            "skip_patterns": [],
            "revision_patterns": []
        }
        
        for answer in scored_answers:
            distribution["by_question_type"][answer.question_type.value] += 1
            
            if answer.dimension_scores:
                avg_confidence = statistics.mean([ds.confidence for ds in answer.dimension_scores])
                if avg_confidence >= 90:
                    distribution["by_confidence_level"]["high"] += 1
                elif avg_confidence >= 70:
                    distribution["by_confidence_level"]["medium"] += 1
                else:
                    distribution["by_confidence_level"]["low"] += 1
            
            if answer.is_skipped:
                distribution["skip_patterns"].append({
                    "question_number": answer.question_number,
                    "question_type": answer.question_type.value,
                    "reason": answer.skip_reason
                })
            
            if answer.metrics.revision_count > 2:
                distribution["revision_patterns"].append({
                    "question_number": answer.question_number,
                    "revisions": answer.metrics.revision_count,
                    "question_type": answer.question_type.value
                })
        
        return dict(distribution)
    
    def _analyze_time_patterns(self, scored_answers: List[Answer]) -> Dict[str, Any]:
        """Analyze time usage patterns."""
        times = [a.metrics.total_time_seconds for a in scored_answers if a.metrics.total_time_seconds > 0]
        
        if not times:
            return {"total_time": 0, "average_time": 0, "time_distribution": {}}
        
        return {
            "total_time_seconds": sum(times),
            "average_time_seconds": statistics.mean(times),
            "median_time_seconds": statistics.median(times),
            "time_distribution": {
                "fast": len([t for t in times if t < 30]),
                "normal": len([t for t in times if 30 <= t <= 120]),
                "slow": len([t for t in times if t > 120])
            },
            "time_consistency": {
                "coefficient_of_variation": statistics.stdev(times) / statistics.mean(times),
                "is_consistent": statistics.stdev(times) / statistics.mean(times) < 0.5
            }
        }
    
    def _analyze_behavior_patterns(self, scored_answers: List[Answer]) -> Dict[str, Any]:
        """Analyze user behavior patterns during assessment."""
        return {
            "engagement_metrics": {
                "total_focus_events": sum(a.metrics.focus_events for a in scored_answers),
                "total_blur_events": sum(a.metrics.blur_events for a in scored_answers),
                "total_changes": sum(a.metrics.change_events for a in scored_answers),
                "average_hesitation": statistics.mean([
                    a.metrics.hesitation_score for a in scored_answers
                ])
            },
            "response_patterns": {
                "immediate_responses": len([
                    a for a in scored_answers if a.metrics.revision_count == 0
                ]),
                "revised_responses": len([
                    a for a in scored_answers if a.metrics.revision_count > 0
                ]),
                "highly_revised": len([
                    a for a in scored_answers if a.metrics.revision_count > 3
                ])
            }
        }
    
    def _analyze_question_performance(
        self,
        scored_answers: List[Answer],
        questions: List[Question]
    ) -> Dict[str, Any]:
        """Analyze performance on different question types."""
        performance = {}
        
        # Group by question type
        by_type = defaultdict(list)
        for answer in scored_answers:
            by_type[answer.question_type.value].append(answer)
        
        for q_type, answers in by_type.items():
            avg_score = statistics.mean([a.total_points for a in answers])
            avg_time = statistics.mean([
                a.metrics.total_time_seconds for a in answers 
                if a.metrics.total_time_seconds > 0
            ]) if any(a.metrics.total_time_seconds > 0 for a in answers) else 0
            
            performance[q_type] = {
                "average_score": avg_score,
                "average_time_seconds": avg_time,
                "completion_rate": len([a for a in answers if not a.is_skipped]) / len(answers),
                "confidence": statistics.mean([
                    statistics.mean([ds.confidence for ds in a.dimension_scores])
                    for a in answers if a.dimension_scores
                ]) if any(a.dimension_scores for a in answers) else 100.0
            }
        
        return performance
    
    async def _get_comparative_metrics(
        self,
        test: Test,
        dimension_scores: Dict[RaisecDimension, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get comparative metrics against other users."""
        # This would query aggregate statistics from other tests
        # For now, returning placeholder data
        
        return {
            "percentile_rankings": {
                dim.value: scores.get("percentile", 50)
                for dim, scores in dimension_scores.items()
            },
            "age_group_comparison": "average",  # above_average, average, below_average
            "completion_time_percentile": 50,
            "consistency_percentile": 75
        }
    
    async def _save_test_scores(self, test_scores: TestScores) -> None:
        """Save test scores to database."""
        scores_dict = test_scores.to_dict(exclude={"id"})
        await self.db.insert_one("test_scores", scores_dict)
        logger.info(f"Test scores saved for test_id: {test_scores.test_id}")
    
    async def _update_test_status(
        self,
        test_id: ObjectId,
        status: TestStatus,
        test_scores: TestScores
    ) -> None:
        """Update test status and add scores reference."""
        await self.db.update_one(
            "tests",
            {"_id": test_id},
            {
                "$set": {
                    "status": status.value,
                    "scores": {
                        "raisec_code": test_scores.raisec_profile.code,
                        "total_score": test_scores.raisec_profile.total_score,
                        "consistency_score": test_scores.consistency_score,
                        "scored_at": test_scores.scoring_metadata["scored_at"]
                    },
                    "completed_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
    
    async def _cache_test_scores(self, test_scores: TestScores) -> None:
        """Cache test scores for quick retrieval."""
        cache_key = CacheKeys.test_scores(str(test_scores.test_id))
        await self.cache_manager.set(
            cache_key,
            test_scores.to_dict(),
            expire_seconds=3600  # 1 hour
        )
    
    async def _get_cached_test_scores(self, test_id: ObjectId) -> Optional[TestScores]:
        """Get cached test scores."""
        cache_key = CacheKeys.test_scores(str(test_id))
        cached_data = await self.cache_manager.get(cache_key)
        
        if cached_data:
            return TestScores.from_dict(cached_data)
        
        return None
    
    def _apply_scoring_adjustments(self, adjustments: Dict[str, Any]) -> None:
        """Apply temporary scoring adjustments."""
        if "question_weights" in adjustments:
            self.question_weights.update(adjustments["question_weights"])
        
        if "time_factors" in adjustments:
            self.time_adjustment_factors.update(adjustments["time_factors"])
        
        if "scoring_config" in adjustments:
            self.scoring_config.update(adjustments["scoring_config"])
    
    # Legacy compatibility methods
    def get_score_interpretation(self, score: float) -> str:
        """Get text interpretation of a score."""
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
        """Get interpretation of RAISEC code."""
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


# Export main class
__all__ = ["ScoringService"]