"""Answer service for managing test answers.

This module handles answer submission, validation, scoring, and retrieval
for the TruScholar assessment system.
"""

import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from bson import ObjectId

from src.cache import Cache
from src.db import Database
from src.models.answer import Answer
from src.models.question import Question
from src.models.test import Test
from src.models.user import User
from src.schemas.answer_schemas import (
    AnswerSubmitRequest,
    BulkAnswerSubmitRequest,
    AnswerResponse,
    AnswerListResponse,
    AnswerValidationResult,
    AnswerScoreResult,
    AnswerStatistics,
    AnswerContext,
    AnswerProcessingResult,
    MCQAnswerData,
    StatementSetAnswerData,
    ScenarioMCQAnswerData,
    ScenarioMultiSelectAnswerData,
    ThisOrThatAnswerData,
    ScaleRatingAnswerData,
    PlotDayAnswerData
)
from src.utils.enums import QuestionType, RaisecDimension, TestStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerService:
    """Service for managing test answers."""
    
    def __init__(self, db: Database, cache: Cache):
        """Initialize answer service.
        
        Args:
            db: Database instance
            cache: Cache instance
        """
        self.db = db
        self.cache = cache
        self.answer_collection = db.get_collection("answers")
        self.question_collection = db.get_collection("questions")
        self.test_collection = db.get_collection("tests")
        
    async def submit_answer(
        self,
        user_id: str,
        request: AnswerSubmitRequest
    ) -> AnswerResponse:
        """Submit an answer to a question.
        
        Args:
            user_id: User ID submitting the answer
            request: Answer submission request
            
        Returns:
            AnswerResponse: Submitted answer details
            
        Raises:
            ValueError: If question doesn't exist or test is completed
        """
        # Get question and validate
        question = await self._get_question(request.question_id)
        if not question:
            raise ValueError(f"Question {request.question_id} not found")
            
        # Get test and validate ownership
        test = await self._get_test(str(question.test_id))
        if not test or str(test.user_id) != user_id:
            raise ValueError("Invalid test or unauthorized access")
            
        if test.status == TestStatus.COMPLETED:
            raise ValueError("Cannot submit answers to completed test")
            
        # Validate answer format
        validation_result = await self._validate_answer(request, question)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid answer: {', '.join(validation_result.errors)}")
            
        # Calculate score
        score_result = await self._calculate_score(request, question)
        
        # Check if answer already exists
        existing_answer = await self.answer_collection.find_one({
            "user_id": ObjectId(user_id),
            "test_id": ObjectId(test.id),
            "question_id": ObjectId(request.question_id)
        })
        
        if existing_answer:
            # Update existing answer
            answer_doc = {
                "answer_data": request.answer_data.model_dump(),
                "validation": validation_result.model_dump(),
                "score": score_result.model_dump() if score_result else None,
                "time_spent_seconds": request.time_spent_seconds,
                "changed_count": existing_answer.get("changed_count", 0) + 1,
                "device_type": request.device_type,
                "updated_at": datetime.utcnow()
            }
            
            await self.answer_collection.update_one(
                {"_id": existing_answer["_id"]},
                {"$set": answer_doc}
            )
            
            answer_id = str(existing_answer["_id"])
            submitted_at = existing_answer["submitted_at"]
            question_number = existing_answer["question_number"]
            
        else:
            # Create new answer
            # Get question number
            question_number = await self._get_question_number(test.id, request.question_id)
            
            answer_doc = {
                "user_id": ObjectId(user_id),
                "test_id": ObjectId(test.id),
                "question_id": ObjectId(request.question_id),
                "question_number": question_number,
                "question_type": request.question_type,
                "answer_data": request.answer_data.model_dump(),
                "validation": validation_result.model_dump(),
                "score": score_result.model_dump() if score_result else None,
                "time_spent_seconds": request.time_spent_seconds,
                "changed_count": request.changed_count,
                "device_type": request.device_type,
                "submitted_at": datetime.utcnow(),
                "updated_at": None
            }
            
            result = await self.answer_collection.insert_one(answer_doc)
            answer_id = str(result.inserted_id)
            submitted_at = answer_doc["submitted_at"]
            
        # Clear cache
        await self._clear_answer_cache(user_id, str(test.id))
        
        # Update test progress
        await self._update_test_progress(str(test.id))
        
        return AnswerResponse(
            id=answer_id,
            test_id=str(test.id),
            question_id=request.question_id,
            question_number=question_number,
            question_type=request.question_type,
            answer_data=request.answer_data.model_dump(),
            validation=validation_result,
            score=score_result,
            time_spent_seconds=request.time_spent_seconds,
            changed_count=request.changed_count if existing_answer else 0,
            device_type=request.device_type,
            submitted_at=submitted_at,
            updated_at=answer_doc.get("updated_at")
        )
        
    async def submit_bulk_answers(
        self,
        user_id: str,
        request: BulkAnswerSubmitRequest
    ) -> List[AnswerResponse]:
        """Submit multiple answers at once.
        
        Args:
            user_id: User ID submitting answers
            request: Bulk answer submission request
            
        Returns:
            List[AnswerResponse]: List of submitted answers
        """
        # Validate test ownership
        test = await self._get_test(request.test_id)
        if not test or str(test.user_id) != user_id:
            raise ValueError("Invalid test or unauthorized access")
            
        # Submit answers concurrently
        tasks = [
            self.submit_answer(user_id, answer)
            for answer in request.answers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        answers = []
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Answer {i}: {str(result)}")
            else:
                answers.append(result)
                
        if errors:
            logger.warning(f"Bulk submit errors: {errors}")
            
        return answers
        
    async def get_answer(
        self,
        user_id: str,
        answer_id: str
    ) -> Optional[AnswerResponse]:
        """Get a specific answer.
        
        Args:
            user_id: User ID
            answer_id: Answer ID
            
        Returns:
            Optional[AnswerResponse]: Answer details if found
        """
        answer = await self.answer_collection.find_one({
            "_id": ObjectId(answer_id),
            "user_id": ObjectId(user_id)
        })
        
        if not answer:
            return None
            
        return self._answer_to_response(answer)
        
    async def get_test_answers(
        self,
        user_id: str,
        test_id: str
    ) -> AnswerListResponse:
        """Get all answers for a test.
        
        Args:
            user_id: User ID
            test_id: Test ID
            
        Returns:
            AnswerListResponse: List of test answers
        """
        # Check cache first
        cache_key = f"test_answers:{user_id}:{test_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return AnswerListResponse(**cached)
            
        # Validate test ownership
        test = await self._get_test(test_id)
        if not test or str(test.user_id) != user_id:
            raise ValueError("Invalid test or unauthorized access")
            
        # Get answers
        cursor = self.answer_collection.find({
            "user_id": ObjectId(user_id),
            "test_id": ObjectId(test_id)
        }).sort("question_number", 1)
        
        answers = []
        async for answer in cursor:
            answers.append(self._answer_to_response(answer))
            
        # Get total questions
        total_questions = await self.question_collection.count_documents({
            "test_id": ObjectId(test_id)
        })
        
        response = AnswerListResponse(
            answers=answers,
            total=len(answers),
            test_id=test_id,
            questions_answered=len(answers),
            questions_remaining=total_questions - len(answers)
        )
        
        # Cache result
        await self.cache.set(cache_key, response.model_dump(), expire=300)
        
        return response
        
    async def get_answer_statistics(
        self,
        user_id: str,
        test_id: str
    ) -> AnswerStatistics:
        """Get statistics about test answers.
        
        Args:
            user_id: User ID
            test_id: Test ID
            
        Returns:
            AnswerStatistics: Answer statistics
        """
        # Validate test ownership
        test = await self._get_test(test_id)
        if not test or str(test.user_id) != user_id:
            raise ValueError("Invalid test or unauthorized access")
            
        # Get all answers
        cursor = self.answer_collection.find({
            "user_id": ObjectId(user_id),
            "test_id": ObjectId(test_id)
        })
        
        total_time = 0
        total_changes = 0
        fastest_time = float('inf')
        slowest_time = 0
        questions_by_type = {}
        
        answer_count = 0
        async for answer in cursor:
            answer_count += 1
            time_spent = answer["time_spent_seconds"]
            total_time += time_spent
            total_changes += answer.get("changed_count", 0)
            
            if time_spent < fastest_time:
                fastest_time = time_spent
            if time_spent > slowest_time:
                slowest_time = time_spent
                
            q_type = answer["question_type"]
            questions_by_type[q_type] = questions_by_type.get(q_type, 0) + 1
            
        # Get total questions
        total_questions = await self.question_collection.count_documents({
            "test_id": ObjectId(test_id)
        })
        
        if answer_count == 0:
            return AnswerStatistics(
                total_time_seconds=0,
                average_time_per_question=0,
                fastest_answer_seconds=0,
                slowest_answer_seconds=0,
                total_changes=0,
                questions_by_type={},
                completion_rate=0
            )
            
        return AnswerStatistics(
            total_time_seconds=total_time,
            average_time_per_question=total_time / answer_count,
            fastest_answer_seconds=int(fastest_time),
            slowest_answer_seconds=int(slowest_time),
            total_changes=total_changes,
            questions_by_type=questions_by_type,
            completion_rate=answer_count / total_questions if total_questions > 0 else 0
        )
        
    async def delete_answer(
        self,
        user_id: str,
        answer_id: str
    ) -> bool:
        """Delete an answer.
        
        Args:
            user_id: User ID
            answer_id: Answer ID
            
        Returns:
            bool: True if deleted successfully
        """
        result = await self.answer_collection.delete_one({
            "_id": ObjectId(answer_id),
            "user_id": ObjectId(user_id)
        })
        
        if result.deleted_count > 0:
            # Clear cache
            answer = await self.answer_collection.find_one({"_id": ObjectId(answer_id)})
            if answer:
                await self._clear_answer_cache(user_id, str(answer["test_id"]))
            return True
            
        return False
        
    # Private helper methods
    
    async def _get_question(self, question_id: str) -> Optional[Question]:
        """Get question by ID."""
        doc = await self.question_collection.find_one({
            "_id": ObjectId(question_id)
        })
        return Question.from_db(doc) if doc else None
        
    async def _get_test(self, test_id: str) -> Optional[Test]:
        """Get test by ID."""
        doc = await self.test_collection.find_one({
            "_id": ObjectId(test_id)
        })
        return Test.from_db(doc) if doc else None
        
    async def _get_question_number(self, test_id: str, question_id: str) -> int:
        """Get question number within test."""
        # Get all questions for test sorted by creation
        cursor = self.question_collection.find({
            "test_id": ObjectId(test_id)
        }).sort("created_at", 1)
        
        question_number = 1
        async for question in cursor:
            if str(question["_id"]) == question_id:
                return question_number
            question_number += 1
            
        return 1  # Default if not found
        
    async def _validate_answer(
        self,
        request: AnswerSubmitRequest,
        question: Question
    ) -> AnswerValidationResult:
        """Validate answer against question requirements."""
        errors = []
        warnings = []
        
        # Type-specific validation
        if request.question_type == QuestionType.MCQ:
            errors.extend(await self._validate_mcq_answer(request.answer_data, question))
            
        elif request.question_type == QuestionType.STATEMENT_SET:
            errors.extend(await self._validate_statement_set_answer(request.answer_data, question))
            
        elif request.question_type == QuestionType.SCENARIO_MCQ:
            errors.extend(await self._validate_scenario_mcq_answer(request.answer_data, question))
            
        elif request.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            errors.extend(await self._validate_scenario_multi_select_answer(request.answer_data, question))
            
        elif request.question_type == QuestionType.THIS_OR_THAT:
            errors.extend(await self._validate_this_or_that_answer(request.answer_data, question))
            
        elif request.question_type == QuestionType.SCALE_RATING:
            errors.extend(await self._validate_scale_rating_answer(request.answer_data, question))
            
        elif request.question_type == QuestionType.PLOT_DAY:
            errors.extend(await self._validate_plot_day_answer(request.answer_data, question))
            
        # Add warnings
        if request.changed_count > 5:
            warnings.append("Answer changed many times")
            
        if request.time_spent_seconds < 5:
            warnings.append("Answer submitted very quickly")
            
        if request.time_spent_seconds > 600:
            warnings.append("Unusually long time spent on question")
            
        return AnswerValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    async def _validate_mcq_answer(
        self,
        answer_data: MCQAnswerData,
        question: Question
    ) -> List[str]:
        """Validate MCQ answer."""
        errors = []
        
        # Check if option exists
        valid_options = ['a', 'b', 'c', 'd']
        if answer_data.selected_option not in valid_options:
            errors.append(f"Invalid option: {answer_data.selected_option}")
            
        # Check if option exists in question
        if hasattr(question.content, 'options'):
            option_ids = [opt['id'] for opt in question.content.options]
            if answer_data.selected_option not in option_ids:
                errors.append(f"Option {answer_data.selected_option} not found in question")
                
        return errors
        
    async def _validate_statement_set_answer(
        self,
        answer_data: StatementSetAnswerData,
        question: Question
    ) -> List[str]:
        """Validate statement set answer."""
        errors = []
        
        # Check if all statements are answered
        if hasattr(question.content, 'statements'):
            statement_ids = [str(i+1) for i in range(len(question.content.statements))]
            answered_ids = list(answer_data.ratings.keys())
            
            missing = set(statement_ids) - set(answered_ids)
            if missing:
                errors.append(f"Missing ratings for statements: {', '.join(missing)}")
                
            extra = set(answered_ids) - set(statement_ids)
            if extra:
                errors.append(f"Extra ratings for non-existent statements: {', '.join(extra)}")
                
        return errors
        
    async def _validate_scenario_mcq_answer(
        self,
        answer_data: ScenarioMCQAnswerData,
        question: Question
    ) -> List[str]:
        """Validate scenario MCQ answer."""
        errors = []
        
        # Similar to MCQ validation
        valid_options = ['a', 'b', 'c', 'd']
        if answer_data.selected_option not in valid_options:
            errors.append(f"Invalid option: {answer_data.selected_option}")
            
        return errors
        
    async def _validate_scenario_multi_select_answer(
        self,
        answer_data: ScenarioMultiSelectAnswerData,
        question: Question
    ) -> List[str]:
        """Validate scenario multi-select answer."""
        errors = []
        
        # Check number of selections
        if len(answer_data.selected_options) == 0:
            errors.append("No options selected")
            
        # Check if options exist in question
        if hasattr(question.content, 'options'):
            valid_options = [opt['id'] for opt in question.content.options]
            for option in answer_data.selected_options:
                if option not in valid_options:
                    errors.append(f"Invalid option: {option}")
                    
        return errors
        
    async def _validate_this_or_that_answer(
        self,
        answer_data: ThisOrThatAnswerData,
        question: Question
    ) -> List[str]:
        """Validate this or that answer."""
        errors = []
        
        if answer_data.selected not in ['A', 'B']:
            errors.append(f"Invalid selection: {answer_data.selected}")
            
        return errors
        
    async def _validate_scale_rating_answer(
        self,
        answer_data: ScaleRatingAnswerData,
        question: Question
    ) -> List[str]:
        """Validate scale rating answer."""
        errors = []
        
        # Check rating range
        if hasattr(question.content, 'scale_min') and hasattr(question.content, 'scale_max'):
            if not (question.content.scale_min <= answer_data.rating <= question.content.scale_max):
                errors.append(f"Rating {answer_data.rating} outside valid range")
                
        return errors
        
    async def _validate_plot_day_answer(
        self,
        answer_data: PlotDayAnswerData,
        question: Question
    ) -> List[str]:
        """Validate plot day answer."""
        errors = []
        
        # Check if all tasks are placed
        if hasattr(question.content, 'tasks'):
            task_ids = [task['id'] for task in question.content.tasks]
            
            placed_tasks = []
            for tasks in answer_data.placements.values():
                placed_tasks.extend(tasks)
            placed_tasks.extend(answer_data.not_interested)
            
            missing = set(task_ids) - set(placed_tasks)
            if missing:
                errors.append(f"Missing task placements: {', '.join(missing)}")
                
            extra = set(placed_tasks) - set(task_ids)
            if extra:
                errors.append(f"Extra task placements: {', '.join(extra)}")
                
        return errors
        
    async def _calculate_score(
        self,
        request: AnswerSubmitRequest,
        question: Question
    ) -> Optional[AnswerScoreResult]:
        """Calculate score for an answer."""
        dimension_scores = {}
        total_score = 0
        
        # Type-specific scoring
        if request.question_type == QuestionType.MCQ:
            dimension_scores = await self._score_mcq_answer(
                request.answer_data,
                question
            )
            
        elif request.question_type == QuestionType.STATEMENT_SET:
            dimension_scores = await self._score_statement_set_answer(
                request.answer_data,
                question
            )
            
        elif request.question_type == QuestionType.SCENARIO_MCQ:
            dimension_scores = await self._score_scenario_mcq_answer(
                request.answer_data,
                question
            )
            
        elif request.question_type == QuestionType.SCENARIO_MULTI_SELECT:
            dimension_scores = await self._score_scenario_multi_select_answer(
                request.answer_data,
                question
            )
            
        elif request.question_type == QuestionType.THIS_OR_THAT:
            dimension_scores = await self._score_this_or_that_answer(
                request.answer_data,
                question
            )
            
        elif request.question_type == QuestionType.SCALE_RATING:
            dimension_scores = await self._score_scale_rating_answer(
                request.answer_data,
                question
            )
            
        elif request.question_type == QuestionType.PLOT_DAY:
            dimension_scores = await self._score_plot_day_answer(
                request.answer_data,
                question
            )
            
        # Calculate total score
        total_score = sum(dimension_scores.values())
        
        # Calculate weighted score based on question type
        weight_map = {
            QuestionType.MCQ: 1.0,
            QuestionType.STATEMENT_SET: 1.2,
            QuestionType.SCENARIO_MCQ: 1.1,
            QuestionType.SCENARIO_MULTI_SELECT: 1.15,
            QuestionType.THIS_OR_THAT: 0.9,
            QuestionType.SCALE_RATING: 1.0,
            QuestionType.PLOT_DAY: 1.3
        }
        
        weighted_score = total_score * weight_map.get(request.question_type, 1.0)
        
        # Calculate confidence based on answer patterns
        confidence = 0.9  # Base confidence
        
        if request.changed_count > 3:
            confidence -= 0.1
        if request.time_spent_seconds < 10:
            confidence -= 0.1
        if request.time_spent_seconds > 300:
            confidence -= 0.05
            
        confidence = max(0.5, confidence)  # Minimum confidence
        
        return AnswerScoreResult(
            dimension_scores=dimension_scores,
            total_score=total_score,
            weighted_score=weighted_score,
            confidence=confidence
        )
        
    async def _score_mcq_answer(
        self,
        answer_data: MCQAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score MCQ answer."""
        scores = {}
        
        # Get option dimensions
        if hasattr(question.content, 'options'):
            for option in question.content.options:
                if option['id'] == answer_data.selected_option:
                    dimensions = option.get('dimensions', {})
                    for dim, score in dimensions.items():
                        scores[dim] = score
                        
        return scores
        
    async def _score_statement_set_answer(
        self,
        answer_data: StatementSetAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score statement set answer."""
        scores = {}
        
        if hasattr(question.content, 'statements'):
            for i, statement in enumerate(question.content.statements):
                statement_id = str(i + 1)
                if statement_id in answer_data.ratings:
                    rating = answer_data.ratings[statement_id]
                    dimensions = statement.get('dimensions', {})
                    
                    # Scale rating to 0-1
                    normalized_rating = (rating - 1) / 4.0
                    
                    for dim, weight in dimensions.items():
                        if dim not in scores:
                            scores[dim] = 0
                        scores[dim] += normalized_rating * weight
                        
        return scores
        
    async def _score_scenario_mcq_answer(
        self,
        answer_data: ScenarioMCQAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score scenario MCQ answer."""
        scores = {}
        
        # Similar to MCQ scoring
        if hasattr(question.content, 'options'):
            for option in question.content.options:
                if option['id'] == answer_data.selected_option:
                    dimensions = option.get('dimensions', {})
                    
                    # Apply confidence if provided
                    confidence_factor = answer_data.confidence_level or 1.0
                    
                    for dim, score in dimensions.items():
                        scores[dim] = score * confidence_factor
                        
        return scores
        
    async def _score_scenario_multi_select_answer(
        self,
        answer_data: ScenarioMultiSelectAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score scenario multi-select answer."""
        scores = {}
        
        if hasattr(question.content, 'options'):
            selected_count = len(answer_data.selected_options)
            
            for option in question.content.options:
                if option['id'] in answer_data.selected_options:
                    dimensions = option.get('dimensions', {})
                    
                    # Normalize by number of selections
                    for dim, score in dimensions.items():
                        if dim not in scores:
                            scores[dim] = 0
                        scores[dim] += score / selected_count
                        
        return scores
        
    async def _score_this_or_that_answer(
        self,
        answer_data: ThisOrThatAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score this or that answer."""
        scores = {}
        
        if hasattr(question.content, 'option_a') and hasattr(question.content, 'option_b'):
            selected_option = question.content.option_a if answer_data.selected == 'A' else question.content.option_b
            dimensions = selected_option.get('dimensions', {})
            
            for dim, score in dimensions.items():
                scores[dim] = score
                
        return scores
        
    async def _score_scale_rating_answer(
        self,
        answer_data: ScaleRatingAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score scale rating answer."""
        scores = {}
        
        # Normalize rating to 0-1
        if hasattr(question.content, 'scale_min') and hasattr(question.content, 'scale_max'):
            scale_range = question.content.scale_max - question.content.scale_min
            normalized_rating = (answer_data.rating - question.content.scale_min) / scale_range
            
            # Apply to question dimensions
            if hasattr(question, 'dimensions'):
                for dim in question.dimensions:
                    scores[dim] = normalized_rating
                    
        return scores
        
    async def _score_plot_day_answer(
        self,
        answer_data: PlotDayAnswerData,
        question: Question
    ) -> Dict[RaisecDimension, float]:
        """Score plot day answer."""
        scores = {}
        
        if hasattr(question.content, 'tasks'):
            # Time slot weights
            slot_weights = {
                'morning': 1.0,
                'afternoon': 0.9,
                'late_afternoon': 0.8,
                'evening': 0.7
            }
            
            # Score based on task placements
            for slot, tasks in answer_data.placements.items():
                weight = slot_weights.get(slot, 0.5)
                
                for task_id in tasks:
                    # Find task
                    for task in question.content.tasks:
                        if task['id'] == task_id:
                            primary_dim = task.get('primary_dimension')
                            if primary_dim:
                                if primary_dim not in scores:
                                    scores[primary_dim] = 0
                                scores[primary_dim] += weight
                                
                            # Secondary dimensions with lower weight
                            for sec_dim in task.get('secondary_dimensions', []):
                                if sec_dim not in scores:
                                    scores[sec_dim] = 0
                                scores[sec_dim] += weight * 0.5
                                
            # Normalize scores
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    for dim in scores:
                        scores[dim] = scores[dim] / max_score
                        
        return scores
        
    async def _clear_answer_cache(self, user_id: str, test_id: str):
        """Clear answer-related cache entries."""
        cache_keys = [
            f"test_answers:{user_id}:{test_id}",
            f"test_progress:{test_id}",
            f"user_tests:{user_id}"
        ]
        
        for key in cache_keys:
            await self.cache.delete(key)
            
    async def _update_test_progress(self, test_id: str):
        """Update test progress based on answers."""
        # Count answered questions
        answered_count = await self.answer_collection.count_documents({
            "test_id": ObjectId(test_id)
        })
        
        # Count total questions
        total_count = await self.question_collection.count_documents({
            "test_id": ObjectId(test_id)
        })
        
        # Update test
        update_doc = {
            "questions_answered": answered_count,
            "progress": answered_count / total_count if total_count > 0 else 0,
            "updated_at": datetime.utcnow()
        }
        
        # If all questions answered, mark as completed
        if answered_count >= total_count and total_count > 0:
            update_doc["status"] = TestStatus.COMPLETED
            update_doc["completed_at"] = datetime.utcnow()
            
        await self.test_collection.update_one(
            {"_id": ObjectId(test_id)},
            {"$set": update_doc}
        )
        
    def _answer_to_response(self, answer_doc: dict) -> AnswerResponse:
        """Convert answer document to response schema."""
        return AnswerResponse(
            id=str(answer_doc["_id"]),
            test_id=str(answer_doc["test_id"]),
            question_id=str(answer_doc["question_id"]),
            question_number=answer_doc["question_number"],
            question_type=answer_doc["question_type"],
            answer_data=answer_doc["answer_data"],
            validation=AnswerValidationResult(**answer_doc["validation"]),
            score=AnswerScoreResult(**answer_doc["score"]) if answer_doc.get("score") else None,
            time_spent_seconds=answer_doc["time_spent_seconds"],
            changed_count=answer_doc.get("changed_count", 0),
            device_type=answer_doc.get("device_type"),
            submitted_at=answer_doc["submitted_at"],
            updated_at=answer_doc.get("updated_at")
        )


# Export the service
__all__ = ["AnswerService"]