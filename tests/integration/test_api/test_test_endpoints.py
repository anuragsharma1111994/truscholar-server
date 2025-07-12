"""Integration tests for test API endpoints.

This module contains integration tests for all test-related API endpoints,
testing the complete flow from test creation to completion.
"""

import pytest
from httpx import AsyncClient
from datetime import datetime
from unittest.mock import AsyncMock, patch
from bson import ObjectId

from src.api.main import app
from src.models.test import Test, TestStatus
from src.models.user import User
from src.models.question import Question
from src.models.answer import Answer
from src.utils.constants import AgeGroup, QuestionType


@pytest.fixture
async def authenticated_client():
    """Create authenticated test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Mock authentication
        with patch('src.routers.auth.get_current_user') as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=str(ObjectId()),
                phone="9876543210"
            )
            yield client


@pytest.fixture
def mock_test_id():
    """Generate mock test ID."""
    return str(ObjectId())


@pytest.fixture
def mock_question_id():
    """Generate mock question ID."""
    return str(ObjectId())


class TestTestEndpoints:
    """Integration tests for test endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_test_success(self, authenticated_client):
        """Test successful test creation."""
        # Arrange
        request_data = {
            "age": 20,
            "test_type": "raisec",
            "is_practice": False
        }
        
        with patch('src.services.test_service.TestService.create_test') as mock_create:
            mock_test = Test(
                user_id=ObjectId(),
                age_group=AgeGroup.AGE_18_25,
                status=TestStatus.CREATED
            )
            mock_test.id = ObjectId()
            mock_create.return_value = mock_test
            
            # Act
            response = await authenticated_client.post(
                "/api/v1/tests/",
                json=request_data
            )
            
            # Assert
            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Test created successfully"
            assert data["data"]["status"] == "created"
            assert data["data"]["age_group"] == "18-25"
    
    @pytest.mark.asyncio
    async def test_create_test_invalid_age(self, authenticated_client):
        """Test test creation with invalid age."""
        # Arrange
        request_data = {
            "age": 10,
            "test_type": "raisec",
            "is_practice": False
        }
        
        with patch('src.services.test_service.TestService.create_test') as mock_create:
            from src.utils.exceptions import ValidationError
            mock_create.side_effect = ValidationError("Age 10 is not supported")
            
            # Act
            response = await authenticated_client.post(
                "/api/v1/tests/",
                json=request_data
            )
            
            # Assert
            assert response.status_code == 422
            data = response.json()
            assert "Age 10 is not supported" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_start_test_success(self, authenticated_client, mock_test_id):
        """Test successful test start."""
        # Arrange
        with patch('src.services.test_service.TestService.start_test') as mock_start:
            with patch('src.services.test_service.TestService.get_questions') as mock_questions:
                mock_test = Test(
                    user_id=ObjectId(),
                    age_group=AgeGroup.AGE_18_25,
                    status=TestStatus.QUESTIONS_READY
                )
                mock_test.id = ObjectId(mock_test_id)
                mock_start.return_value = mock_test
                
                # Mock questions
                questions = []
                for i in range(12):
                    q = Question(
                        test_id=mock_test.id,
                        question_number=i + 1,
                        question_type=QuestionType.MCQ,
                        question_text=f"Question {i + 1}"
                    )
                    q.id = ObjectId()
                    questions.append(q)
                mock_questions.return_value = questions
                
                # Act
                response = await authenticated_client.post(
                    f"/api/v1/tests/{mock_test_id}/start"
                )
                
                # Assert
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["message"] == "Test started successfully"
                assert len(data["data"]["questions"]) == 12
    
    @pytest.mark.asyncio
    async def test_get_test_success(self, authenticated_client, mock_test_id):
        """Test successful test retrieval."""
        # Arrange
        with patch('src.services.test_service.TestService.get_test') as mock_get:
            mock_test = Test(
                user_id=ObjectId(),
                age_group=AgeGroup.AGE_18_25,
                status=TestStatus.IN_PROGRESS
            )
            mock_test.id = ObjectId(mock_test_id)
            mock_get.return_value = mock_test
            
            # Act
            response = await authenticated_client.get(
                f"/api/v1/tests/{mock_test_id}"
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["test_id"] == mock_test_id
            assert data["data"]["status"] == "in_progress"
    
    @pytest.mark.asyncio
    async def test_get_test_not_found(self, authenticated_client, mock_test_id):
        """Test test retrieval when not found."""
        # Arrange
        with patch('src.services.test_service.TestService.get_test') as mock_get:
            from src.utils.exceptions import ResourceNotFoundError
            mock_get.side_effect = ResourceNotFoundError("Test not found")
            
            # Act
            response = await authenticated_client.get(
                f"/api/v1/tests/{mock_test_id}"
            )
            
            # Assert
            assert response.status_code == 404
            data = response.json()
            assert data["detail"] == "Test not found"
    
    @pytest.mark.asyncio
    async def test_submit_answer_success(self, authenticated_client, mock_test_id, mock_question_id):
        """Test successful answer submission."""
        # Arrange
        request_data = {
            "question_id": mock_question_id,
            "answer_data": {"selected_option": "A"},
            "time_spent_seconds": 30
        }
        
        with patch('src.services.test_service.TestService.submit_answer') as mock_submit:
            mock_answer = Answer(
                test_id=ObjectId(mock_test_id),
                question_id=ObjectId(mock_question_id),
                user_id=ObjectId(),
                question_number=1,
                question_type=QuestionType.MCQ,
                answer_data=request_data["answer_data"]
            )
            mock_answer.id = ObjectId()
            mock_submit.return_value = mock_answer
            
            # Act
            response = await authenticated_client.post(
                f"/api/v1/tests/{mock_test_id}/answers",
                json=request_data
            )
            
            # Assert
            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Answer submitted successfully"
            assert data["data"]["question_id"] == mock_question_id
    
    @pytest.mark.asyncio
    async def test_submit_test_success(self, authenticated_client, mock_test_id):
        """Test successful test submission."""
        # Arrange
        request_data = {
            "test_id": mock_test_id,
            "final_review": True,
            "feedback": "Good test"
        }
        
        with patch('src.services.test_service.TestService.submit_test') as mock_submit:
            mock_test = Test(
                user_id=ObjectId(),
                age_group=AgeGroup.AGE_18_25,
                status=TestStatus.SCORED
            )
            mock_test.id = ObjectId(mock_test_id)
            mock_test.scores = MagicMock()
            mock_test.scores.raisec_code = "IRA"
            mock_test.scores.get_dimension_scores.return_value = {
                "realistic": MagicMock(percentage=70.0, rank=2, confidence=0.85),
                "investigative": MagicMock(percentage=85.0, rank=1, confidence=0.90),
                "artistic": MagicMock(percentage=75.0, rank=3, confidence=0.88),
                "social": MagicMock(percentage=50.0, rank=4, confidence=0.80),
                "enterprising": MagicMock(percentage=45.0, rank=5, confidence=0.78),
                "conventional": MagicMock(percentage=40.0, rank=6, confidence=0.75)
            }
            mock_test.scores.total_score = 365.0
            mock_test.scores.consistency_score = 0.85
            mock_test.progress = MagicMock()
            mock_test.progress.scoring_started_at = datetime.utcnow()
            
            mock_submit.return_value = mock_test
            
            # Act
            response = await authenticated_client.post(
                f"/api/v1/tests/{mock_test_id}/submit",
                json=request_data
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == "Test scored successfully"
            assert data["data"]["raisec_code"] == "IRA"
            assert "dimension_scores" in data["data"]
    
    @pytest.mark.asyncio
    async def test_submit_career_interests_success(self, authenticated_client, mock_test_id):
        """Test successful career interests submission."""
        # Arrange
        request_data = {
            "interests_text": "I want to work in technology and research",
            "current_status": "University student studying computer science"
        }
        
        with patch('src.services.test_service.TestService.submit_career_interests') as mock_submit:
            mock_test = Test(
                user_id=ObjectId(),
                age_group=AgeGroup.AGE_18_25,
                status=TestStatus.SCORED
            )
            mock_test.id = ObjectId(mock_test_id)
            mock_test.career_interests = MagicMock()
            mock_test.career_interests.submitted_at = datetime.utcnow()
            
            validation_questions = [
                "Do you enjoy solving complex problems?",
                "Would you be comfortable in research roles?",
                "Do you prefer analytical work?"
            ]
            
            mock_submit.return_value = (mock_test, validation_questions)
            
            # Act
            response = await authenticated_client.post(
                f"/api/v1/tests/{mock_test_id}/career-interests",
                json=request_data
            )
            
            # Assert
            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
            assert data["data"]["interests_submitted"] is True
            assert len(data["data"]["validation_questions"]) == 3
    
    @pytest.mark.asyncio
    async def test_submit_validation_responses_success(self, authenticated_client, mock_test_id):
        """Test successful validation responses submission."""
        # Arrange
        request_data = {
            "responses": [True, False, True]
        }
        
        with patch('src.services.test_service.TestService.submit_validation_responses') as mock_submit:
            mock_test = Test(
                user_id=ObjectId(),
                age_group=AgeGroup.AGE_18_25,
                status=TestStatus.VALIDATION_COMPLETED
            )
            mock_test.id = ObjectId(mock_test_id)
            mock_test.updated_at = datetime.utcnow()
            
            mock_submit.return_value = mock_test
            
            # Act
            response = await authenticated_client.post(
                f"/api/v1/tests/{mock_test_id}/validation",
                json=request_data
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["validation_completed"] is True
            assert data["data"]["test_ready_for_recommendations"] is True
    
    @pytest.mark.asyncio
    async def test_extend_test_success(self, authenticated_client, mock_test_id):
        """Test successful test extension."""
        # Arrange
        hours = 2
        
        with patch('src.services.test_service.TestService.extend_test') as mock_extend:
            mock_test = Test(
                user_id=ObjectId(),
                age_group=AgeGroup.AGE_18_25,
                status=TestStatus.IN_PROGRESS
            )
            mock_test.id = ObjectId(mock_test_id)
            mock_test.expires_at = datetime.utcnow()
            
            mock_extend.return_value = mock_test
            
            # Act
            response = await authenticated_client.post(
                f"/api/v1/tests/{mock_test_id}/extend?hours={hours}"
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["message"] == f"Test extended by {hours} hours"
            assert data["data"]["extended"] is True
            assert data["data"]["hours_added"] == hours
    
    @pytest.mark.asyncio
    async def test_get_user_tests_success(self, authenticated_client):
        """Test successful user tests retrieval."""
        # Arrange
        with patch('src.services.test_service.TestService.get_user_tests') as mock_get:
            mock_tests = []
            for i in range(3):
                test = Test(
                    user_id=ObjectId(),
                    age_group=AgeGroup.AGE_18_25,
                    status=TestStatus.SCORED if i < 2 else TestStatus.IN_PROGRESS
                )
                test.id = ObjectId()
                test.progress = MagicMock()
                test.progress.completion_percentage = 100.0 if i < 2 else 50.0
                test.progress.completed_at = datetime.utcnow() if i < 2 else None
                test.scores = MagicMock() if i < 2 else None
                if test.scores:
                    test.scores.raisec_code = "IRA"
                test.is_complete = i < 2
                test.completion_time_minutes = 30 if i < 2 else None
                mock_tests.append(test)
            
            mock_get.return_value = mock_tests
            
            # Act
            response = await authenticated_client.get(
                "/api/v1/tests/?limit=10"
            )
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["total_tests"] == 3
            assert data["data"]["completed_tests"] == 2
            assert data["data"]["in_progress_tests"] == 1
            assert len(data["data"]["tests"]) == 3
    
    @pytest.mark.asyncio
    async def test_test_health_check(self, authenticated_client):
        """Test health check endpoint."""
        # Act
        response = await authenticated_client.get("/api/v1/tests/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["service"] == "tests"
        assert data["data"]["status"] == "healthy"
        assert data["data"]["version"] == "1.0.0"