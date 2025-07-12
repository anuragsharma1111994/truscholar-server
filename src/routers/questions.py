"""Question management routes for TruScholar API.

This module handles all question-related endpoints including question generation,
retrieval, regeneration, and validation.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from bson import ObjectId

from src.api.dependencies import (
    get_current_user,
    get_db,
    get_cache,
    get_request_id
)
from src.schemas.question_schemas import (
    QuestionGenerateRequest,
    QuestionRegenerateRequest,
    QuestionResponse,
    QuestionListResponse,
    QuestionGenerationStatusResponse,
    QuestionValidationResponse
)
from src.schemas.base import (
    SuccessResponse,
    ErrorResponse,
    create_success_response,
    create_error_response
)
from src.services.question_service import QuestionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/questions",
    tags=["questions"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Question not found"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    }
)


@router.post("/generate", response_model=SuccessResponse[QuestionGenerationStatusResponse])
async def generate_questions(
    request: QuestionGenerateRequest,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[QuestionGenerationStatusResponse]:
    """Generate AI-powered questions for a test.
    
    This endpoint triggers asynchronous question generation using LangChain and LLMs.
    Questions are generated based on the user's age group and RAISEC methodology.
    """
    try:
        logger.info(f"Generating questions for test {request.test_id}")
        
        question_service = QuestionService(db, cache)
        
        # Validate test ownership
        await question_service.validate_test_ownership(
            test_id=request.test_id,
            user_id=current_user["id"]
        )
        
        # Start question generation
        generation_status = await question_service.generate_questions(
            test_id=request.test_id,
            age_group=request.age_group,
            question_distribution=request.question_types,
            include_static=request.include_static
        )
        
        return create_success_response(
            data=generation_status,
            message="Question generation started successfully",
            request_id=request_id
        )
        
    except ValueError as e:
        logger.error(f"Validation error generating questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except PermissionError as e:
        logger.error(f"Permission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate questions"
        )


@router.get("/test/{test_id}", response_model=SuccessResponse[QuestionListResponse])
async def get_test_questions(
    test_id: str,
    include_content: bool = Query(True, description="Include full question content"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[QuestionListResponse]:
    """Get all questions for a specific test."""
    try:
        if not ObjectId.is_valid(test_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid test ID format"
            )
        
        question_service = QuestionService(db, cache)
        
        # Validate test ownership
        await question_service.validate_test_ownership(
            test_id=test_id,
            user_id=current_user["id"]
        )
        
        # Get questions
        questions = await question_service.get_test_questions(
            test_id=test_id,
            include_content=include_content
        )
        
        response = QuestionListResponse(
            questions=questions,
            total=len(questions),
            test_id=test_id,
            generation_complete=len(questions) == 12
        )
        
        return create_success_response(
            data=response,
            message=f"Retrieved {len(questions)} questions",
            request_id=request_id
        )
        
    except PermissionError as e:
        logger.error(f"Permission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve questions"
        )


@router.get("/{question_id}", response_model=SuccessResponse[QuestionResponse])
async def get_question(
    question_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[QuestionResponse]:
    """Get a specific question by ID."""
    try:
        if not ObjectId.is_valid(question_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid question ID format"
            )
        
        question_service = QuestionService(db, cache)
        
        # Get question with ownership validation
        question = await question_service.get_question(
            question_id=question_id,
            user_id=current_user["id"]
        )
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        return create_success_response(
            data=question,
            message="Question retrieved successfully",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve question"
        )


@router.post("/{question_id}/regenerate", response_model=SuccessResponse[QuestionResponse])
async def regenerate_question(
    question_id: str,
    request: QuestionRegenerateRequest,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[QuestionResponse]:
    """Regenerate a specific question.
    
    This endpoint allows regenerating a question if the user finds it unclear,
    inappropriate, or wants a different variation.
    """
    try:
        if not ObjectId.is_valid(question_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid question ID format"
            )
        
        question_service = QuestionService(db, cache)
        
        # Regenerate question
        regenerated_question = await question_service.regenerate_question(
            test_id=request.test_id,
            question_number=request.question_number,
            user_id=current_user["id"],
            reason=request.reason
        )
        
        if not regenerated_question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found or cannot be regenerated"
            )
        
        return create_success_response(
            data=regenerated_question,
            message="Question regenerated successfully",
            request_id=request_id
        )
        
    except ValueError as e:
        logger.error(f"Validation error regenerating question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate question"
        )


@router.get("/{question_id}/validate", response_model=SuccessResponse[QuestionValidationResponse])
async def validate_question(
    question_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[QuestionValidationResponse]:
    """Validate a question for correctness and quality.
    
    This endpoint checks if a question meets quality standards, has proper
    RAISEC dimension mapping, and follows age-appropriate guidelines.
    """
    try:
        if not ObjectId.is_valid(question_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid question ID format"
            )
        
        question_service = QuestionService(db, cache)
        
        # Validate question
        validation_result = await question_service.validate_question(
            question_id=question_id,
            user_id=current_user["id"]
        )
        
        if not validation_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        return create_success_response(
            data=validation_result,
            message="Question validation completed",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate question"
        )


@router.get("/generation/status/{test_id}", response_model=SuccessResponse[QuestionGenerationStatusResponse])
async def get_generation_status(
    test_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[QuestionGenerationStatusResponse]:
    """Get question generation status for a test.
    
    This endpoint provides real-time status of asynchronous question generation,
    including progress, errors, and estimated completion time.
    """
    try:
        if not ObjectId.is_valid(test_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid test ID format"
            )
        
        question_service = QuestionService(db, cache)
        
        # Get generation status
        status = await question_service.get_generation_status(
            test_id=test_id,
            user_id=current_user["id"]
        )
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test not found or no generation in progress"
            )
        
        return create_success_response(
            data=status,
            message="Generation status retrieved",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting generation status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get generation status"
        )


@router.post("/static/reload", response_model=SuccessResponse[dict])
async def reload_static_questions(
    age_group: Optional[str] = Query(None, description="Specific age group to reload"),
    question_type: Optional[str] = Query(None, description="Specific question type to reload"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[dict]:
    """Reload static question templates from files.
    
    This endpoint is typically used by administrators to refresh the static
    question bank after updates.
    """
    try:
        # TODO: Add admin role check
        # if current_user.get("role") != "admin":
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="Admin access required"
        #     )
        
        question_service = QuestionService(db, cache)
        
        # Reload static questions
        result = await question_service.reload_static_questions(
            age_group=age_group,
            question_type=question_type
        )
        
        return create_success_response(
            data=result,
            message="Static questions reloaded successfully",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error reloading static questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload static questions"
        )


@router.get("/types/distribution", response_model=SuccessResponse[dict])
async def get_question_type_distribution(
    age_group: Optional[str] = Query(None, description="Filter by age group"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    cache = Depends(get_cache),
    request_id: str = Depends(get_request_id)
) -> SuccessResponse[dict]:
    """Get the recommended question type distribution for tests.
    
    This endpoint returns the standard distribution of question types
    used in RAISEC assessments, optionally filtered by age group.
    """
    try:
        question_service = QuestionService(db, cache)
        
        distribution = await question_service.get_question_distribution(
            age_group=age_group
        )
        
        return create_success_response(
            data=distribution,
            message="Question distribution retrieved",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Error getting question distribution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get question distribution"
        )