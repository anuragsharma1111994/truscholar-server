"""Test management API endpoints for TruScholar application.

This module provides FastAPI routes for test lifecycle management including
creation, question handling, answer submission, and completion.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.routers.auth import get_current_user, TokenData
from src.schemas.base import (
    SuccessResponse,
    create_success_response,
)
from src.schemas.test_schemas import (
    AnswerResponse,
    AnswerSubmissionRequest,
    CareerInterestsRequest,
    CareerInterestsResponse,
    QuestionResponse,
    TestListResponse,
    TestResponse,
    TestScoresResponse,
    TestStartRequest,
    TestSubmissionRequest,
    TestSummaryResponse,
    ValidationQuestionsRequest,
    ValidationQuestionsResponse,
)
from src.services.test_service import TestService
from src.utils.exceptions import (
    BusinessLogicError,
    ResourceNotFoundError,
    TruScholarError,
    ValidationError,
)
from src.utils.logger import get_api_logger, log_api_request, log_api_response

# Initialize router
router = APIRouter(
    prefix="/tests",
    tags=["Tests"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Permission denied"},
        404: {"description": "Test not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Initialize services
test_service = TestService()

# Logger
logger = get_api_logger()


@router.post(
    "/",
    response_model=SuccessResponse[TestResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create new test",
    description="Create a new RAISEC test for the authenticated user"
)
async def create_test(
    test_request: TestStartRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[TestResponse]:
    """Create a new test for the current user.

    Args:
        test_request: Test creation request
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[TestResponse]: Created test information

    Raises:
        HTTPException: If test creation fails
    """
    log_api_request("POST", "/tests", current_user.user_id, logger=logger)

    try:
        # Create test
        test = await test_service.create_test(current_user.user_id, test_request)

        # Convert to response format
        test_response = TestResponse(
            test_id=str(test.id),
            user_id=str(test.user_id),
            age_group=test.age_group.value,
            status=test.status.value,
            progress=test.get_progress_summary()["progress"],
            is_practice=test.is_practice,
            created_at=test.created_at.isoformat(),
            updated_at=test.updated_at.isoformat(),
            expires_at=test.expires_at.isoformat(),
            can_answer=test.can_answer_questions(),
            can_submit=test.progress.is_complete,
            can_extend=test.status.value in ["in_progress", "questions_ready"]
        )

        log_api_response("POST", "/tests", status.HTTP_201_CREATED, 0, logger=logger)

        return create_success_response(
            data=test_response,
            message="Test created successfully"
        )

    except ValidationError as e:
        logger.warning(f"Test creation validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except BusinessLogicError as e:
        logger.warning(f"Test creation business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Test creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test"
        )


@router.post(
    "/{test_id}/start",
    response_model=SuccessResponse[TestResponse],
    summary="Start test",
    description="Start a test by generating questions"
)
async def start_test(
    test_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[TestResponse]:
    """Start a test by generating questions.

    Args:
        test_id: ID of the test to start
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[TestResponse]: Started test with questions ready

    Raises:
        HTTPException: If test cannot be started
    """
    log_api_request("POST", f"/tests/{test_id}/start", current_user.user_id, logger=logger)

    try:
        # Start test
        test = await test_service.start_test(test_id, current_user.user_id)

        # Get questions for response
        questions = await test_service.get_questions(test_id, current_user.user_id)
        question_responses = [
            QuestionResponse(**q.to_display_dict())
            for q in questions
        ]

        # Create response
        test_response = TestResponse(
            test_id=str(test.id),
            user_id=str(test.user_id),
            age_group=test.age_group.value,
            status=test.status.value,
            progress=test.get_progress_summary()["progress"],
            questions=question_responses,
            is_practice=test.is_practice,
            created_at=test.created_at.isoformat(),
            updated_at=test.updated_at.isoformat(),
            expires_at=test.expires_at.isoformat(),
            can_answer=test.can_answer_questions(),
            can_submit=test.progress.is_complete,
            can_extend=True
        )

        log_api_response("POST", f"/tests/{test_id}/start", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=test_response,
            message="Test started successfully"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test not found"
        )
    except BusinessLogicError as e:
        logger.warning(f"Test start business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Test start error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start test"
        )


@router.get(
    "/{test_id}",
    response_model=SuccessResponse[TestResponse],
    summary="Get test details",
    description="Get test information and progress"
)
async def get_test(
    test_id: str,
    request: Request,
    include_questions: bool = Query(False, description="Include questions in response"),
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[TestResponse]:
    """Get test details and progress.

    Args:
        test_id: ID of the test
        request: FastAPI request object
        include_questions: Whether to include questions in response
        current_user: Current authenticated user

    Returns:
        SuccessResponse[TestResponse]: Test information

    Raises:
        HTTPException: If test not found or access denied
    """
    log_api_request("GET", f"/tests/{test_id}", current_user.user_id, logger=logger)

    try:
        # Get test
        test = await test_service.get_test(test_id, current_user.user_id)

        # Get questions if requested and available
        questions = []
        if include_questions and test.status.value in ["questions_ready", "in_progress", "scoring", "scored"]:
            question_objs = await test_service.get_questions(test_id, current_user.user_id)
            questions = [QuestionResponse(**q.to_display_dict()) for q in question_objs]

        # Create response
        test_response = TestResponse(
            test_id=str(test.id),
            user_id=str(test.user_id),
            age_group=test.age_group.value,
            status=test.status.value,
            progress=test.get_progress_summary()["progress"],
            questions=questions,
            is_practice=test.is_practice,
            created_at=test.created_at.isoformat(),
            updated_at=test.updated_at.isoformat(),
            expires_at=test.expires_at.isoformat(),
            can_answer=test.can_answer_questions(),
            can_submit=test.progress.is_complete,
            can_extend=test.status.value in ["in_progress", "questions_ready"]
        )

        log_api_response("GET", f"/tests/{test_id}", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=test_response,
            message="Test retrieved successfully"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test not found"
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Get test error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve test"
        )


@router.get(
    "/{test_id}/questions",
    response_model=SuccessResponse[List[QuestionResponse]],
    summary="Get test questions",
    description="Get all questions for a test"
)
async def get_test_questions(
    test_id: str,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[List[QuestionResponse]]:
    """Get all questions for a test.

    Args:
        test_id: ID of the test
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[List[QuestionResponse]]: List of questions

    Raises:
        HTTPException: If questions not found or not ready
    """
    log_api_request("GET", f"/tests/{test_id}/questions", current_user.user_id, logger=logger)

    try:
        # Get questions
        questions = await test_service.get_questions(test_id, current_user.user_id)

        # Convert to response format
        question_responses = [
            QuestionResponse(**q.to_display_dict())
            for q in questions
        ]

        log_api_response("GET", f"/tests/{test_id}/questions", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=question_responses,
            message=f"Retrieved {len(question_responses)} questions"
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test or questions not found"
        )
    except BusinessLogicError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Get questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve questions"
        )


@router.post(
    "/{test_id}/answers",
    response_model=SuccessResponse[AnswerResponse],
    summary="Submit answer",
    description="Submit an answer to a question"
)
async def submit_answer(
    test_id: str,
    answer_request: AnswerSubmissionRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[AnswerResponse]:
    """Submit an answer to a question.

    Args:
        test_id: ID of the test
        answer_request: Answer submission request
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[AnswerResponse]: Answer submission result

    Raises:
        HTTPException: If answer submission fails
    """
    log_api_request("POST", f"/tests/{test_id}/answers", current_user.user_id, logger=logger)

    try:
        # Submit answer
        answer = await test_service.submit_answer(test_id, current_user.user_id, answer_request)

        # Create response
        answer_response = AnswerResponse(
            answer_id=str(answer.id),
            question_id=str(answer.question_id),
            question_number=answer.question_number,
            is_valid=answer.validation.is_valid,
            validation_errors=answer.validation.errors,
            time_spent_seconds=answer.metrics.total_time_seconds,
            submitted_at=answer.created_at.isoformat()
        )

        log_api_response("POST", f"/tests/{test_id}/answers", status.HTTP_201_CREATED, 0, logger=logger)

        return create_success_response(
            data=answer_response,
            message="Answer submitted successfully"
        )

    except ValidationError as e:
        logger.warning(f"Answer validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except BusinessLogicError as e:
        logger.warning(f"Answer submission business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Answer submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit answer"
        )


@router.post(
    "/{test_id}/submit",
    response_model=SuccessResponse[TestScoresResponse],
    summary="Submit test for scoring",
    description="Submit completed test for scoring and get RAISEC results"
)
async def submit_test(
    test_id: str,
    submission_request: TestSubmissionRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[TestScoresResponse]:
    """Submit completed test for scoring.

    Args:
        test_id: ID of the test
        submission_request: Test submission request
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[TestScoresResponse]: Test scores and RAISEC code

    Raises:
        HTTPException: If test submission fails
    """
    log_api_request("POST", f"/tests/{test_id}/submit", current_user.user_id, logger=logger)

    try:
        # Submit test for scoring
        test = await test_service.submit_test(test_id, current_user.user_id)

        # Create scores response
        dimension_scores = {}
        for dim_name, dim_score in test.scores.get_dimension_scores().items():
            dimension_scores[dim_name] = {
                "score": dim_score.percentage,
                "rank": dim_score.rank,
                "confidence": dim_score.confidence
            }

        scores_response = TestScoresResponse(
            test_id=str(test.id),
            raisec_code=test.scores.raisec_code,
            dimension_scores=dimension_scores,
            total_score=test.scores.total_score,
            consistency_score=test.scores.consistency_score,
            scored_at=test.progress.scoring_started_at.isoformat() if test.progress.scoring_started_at else datetime.utcnow().isoformat()
        )

        log_api_response("POST", f"/tests/{test_id}/submit", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=scores_response,
            message="Test scored successfully"
        )

    except BusinessLogicError as e:
        logger.warning(f"Test submission business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Test submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit test"
        )


@router.post(
    "/{test_id}/career-interests",
    response_model=SuccessResponse[CareerInterestsResponse],
    summary="Submit career interests",
    description="Submit career interests and get validation questions"
)
async def submit_career_interests(
    test_id: str,
    interests_request: CareerInterestsRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[CareerInterestsResponse]:
    """Submit career interests and get validation questions.

    Args:
        test_id: ID of the test
        interests_request: Career interests request
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[CareerInterestsResponse]: Validation questions

    Raises:
        HTTPException: If interests submission fails
    """
    log_api_request("POST", f"/tests/{test_id}/career-interests", current_user.user_id, logger=logger)

    try:
        # Submit career interests
        test, validation_questions = await test_service.submit_career_interests(
            test_id, current_user.user_id, interests_request
        )

        # Create response
        interests_response = CareerInterestsResponse(
            interests_submitted=True,
            validation_questions=validation_questions,
            submitted_at=test.career_interests.submitted_at.isoformat()
        )

        log_api_response("POST", f"/tests/{test_id}/career-interests", status.HTTP_201_CREATED, 0, logger=logger)

        return create_success_response(
            data=interests_response,
            message="Career interests submitted successfully"
        )

    except BusinessLogicError as e:
        logger.warning(f"Career interests business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Career interests error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit career interests"
        )


@router.post(
    "/{test_id}/validation",
    response_model=SuccessResponse[dict],
    summary="Submit validation responses",
    description="Submit yes/no responses to validation questions"
)
async def submit_validation_responses(
    test_id: str,
    validation_request: ValidationQuestionsRequest,
    request: Request,
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Submit validation question responses.

    Args:
        test_id: ID of the test
        validation_request: Validation responses
        request: FastAPI request object
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Validation completion confirmation

    Raises:
        HTTPException: If validation submission fails
    """
    log_api_request("POST", f"/tests/{test_id}/validation", current_user.user_id, logger=logger)

    try:
        # Submit validation responses
        test = await test_service.submit_validation_responses(
            test_id, current_user.user_id, validation_request
        )

        log_api_response("POST", f"/tests/{test_id}/validation", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data={
                "validation_completed": True,
                "test_ready_for_recommendations": True,
                "completed_at": test.updated_at.isoformat()
            },
            message="Validation responses submitted successfully"
        )

    except BusinessLogicError as e:
        logger.warning(f"Validation business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Validation submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit validation responses"
        )


@router.post(
    "/{test_id}/extend",
    response_model=SuccessResponse[dict],
    summary="Extend test time",
    description="Extend test expiry time"
)
async def extend_test(
    test_id: str,
    request: Request,
    hours: int = Query(1, ge=1, le=4, description="Hours to extend"),
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[dict]:
    """Extend test expiry time.

    Args:
        test_id: ID of the test
        request: FastAPI request object
        hours: Number of hours to extend
        current_user: Current authenticated user

    Returns:
        SuccessResponse[dict]: Extension confirmation

    Raises:
        HTTPException: If extension fails
    """
    log_api_request("POST", f"/tests/{test_id}/extend", current_user.user_id, logger=logger)

    try:
        # Extend test
        test = await test_service.extend_test(test_id, current_user.user_id, hours)

        log_api_response("POST", f"/tests/{test_id}/extend", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data={
                "extended": True,
                "hours_added": hours,
                "new_expiry": test.expires_at.isoformat()
            },
            message=f"Test extended by {hours} hours"
        )

    except BusinessLogicError as e:
        logger.warning(f"Test extension business error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except TruScholarError as e:
        logger.error(f"Test extension error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extend test"
        )


@router.get(
    "/",
    response_model=SuccessResponse[TestListResponse],
    summary="Get user tests",
    description="Get list of user's tests"
)
async def get_user_tests(
    request: Request,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of tests to return"),
    current_user: TokenData = Depends(get_current_user)
) -> SuccessResponse[TestListResponse]:
    """Get user's test history.

    Args:
        request: FastAPI request object
        limit: Maximum number of tests to return
        current_user: Current authenticated user

    Returns:
        SuccessResponse[TestListResponse]: List of user tests

    Raises:
        HTTPException: If retrieval fails
    """
    log_api_request("GET", "/tests", current_user.user_id, logger=logger)

    try:
        # Get user tests
        tests = await test_service.get_user_tests(current_user.user_id, limit)

        # Convert to summary format
        test_summaries = []
        completed_count = 0
        in_progress_count = 0

        for test in tests:
            if test.is_complete:
                completed_count += 1
            elif test.status.value in ["in_progress", "questions_ready"]:
                in_progress_count += 1

            test_summary = TestSummaryResponse(
                test_id=str(test.id),
                status=test.status.value,
                completion_percentage=test.progress.completion_percentage,
                raisec_code=test.scores.raisec_code if test.scores else None,
                completed_at=test.progress.completed_at.isoformat() if test.progress.completed_at else None,
                duration_minutes=test.completion_time_minutes,
                can_view_results=test.is_complete
            )
            test_summaries.append(test_summary)

        # Create response
        test_list_response = TestListResponse(
            tests=test_summaries,
            total_tests=len(tests),
            completed_tests=completed_count,
            in_progress_tests=in_progress_count
        )

        log_api_response("GET", "/tests", status.HTTP_200_OK, 0, logger=logger)

        return create_success_response(
            data=test_list_response,
            message=f"Retrieved {len(tests)} tests"
        )

    except TruScholarError as e:
        logger.error(f"Get user tests error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve tests"
        )


# Health check endpoint
@router.get(
    "/health",
    response_model=SuccessResponse[dict],
    summary="Test service health check",
    description="Check if test service is healthy"
)
async def test_health_check() -> SuccessResponse[dict]:
    """Check test service health.

    Returns:
        SuccessResponse[dict]: Health status
    """
    return create_success_response(
        data={
            "service": "tests",
            "status": "healthy",
            "version": "1.0.0"
        },
        message="Test service is healthy"
    )


# Export router
__all__ = ["router"]
