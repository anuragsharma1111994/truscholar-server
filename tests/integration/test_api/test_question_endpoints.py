"""Integration tests for question API endpoints."""

import pytest
from httpx import AsyncClient
from datetime import datetime
from bson import ObjectId

from src.utils.enums import QuestionType, TestStatus, AgeGroup
from tests.conftest import create_test_user, create_test_test, create_test_question


class TestQuestionGeneration:
    """Test question generation endpoints."""
    
    @pytest.mark.asyncio
    async def test_generate_questions_success(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test successful question generation."""
        # Create test for user
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        
        # Generate questions
        response = await async_client.post(
            "/api/v1/questions/generate",
            headers=test_auth_headers,
            json={
                "test_id": str(test["_id"]),
                "question_types": [
                    {"type": "mcq", "count": 2},
                    {"type": "statement_set", "count": 1}
                ],
                "use_ai": False  # Use static questions for testing
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "processing"
        assert "task_id" in data["data"]
        
    @pytest.mark.asyncio
    async def test_generate_questions_invalid_test(
        self,
        async_client: AsyncClient,
        test_auth_headers
    ):
        """Test question generation with invalid test ID."""
        response = await async_client.post(
            "/api/v1/questions/generate",
            headers=test_auth_headers,
            json={
                "test_id": str(ObjectId()),
                "question_types": [{"type": "mcq", "count": 2}],
                "use_ai": False
            }
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["error"]["message"]
        
    @pytest.mark.asyncio
    async def test_generate_questions_completed_test(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test question generation for completed test."""
        # Create completed test
        user = await create_test_user(test_db)
        test = await create_test_test(
            test_db,
            user["_id"],
            status=TestStatus.COMPLETED
        )
        
        response = await async_client.post(
            "/api/v1/questions/generate",
            headers=test_auth_headers,
            json={
                "test_id": str(test["_id"]),
                "question_types": [{"type": "mcq", "count": 2}],
                "use_ai": False
            }
        )
        
        assert response.status_code == 400
        assert "completed" in response.json()["error"]["message"].lower()
        

class TestQuestionRetrieval:
    """Test question retrieval endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_test_questions(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test retrieving questions for a test."""
        # Create test with questions
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        
        # Create questions
        questions = []
        for i in range(3):
            q = await create_test_question(
                test_db,
                test["_id"],
                question_number=i+1,
                question_type=QuestionType.MCQ
            )
            questions.append(q)
            
        # Get questions
        response = await async_client.get(
            f"/api/v1/questions/test/{test['_id']}",
            headers=test_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["questions"]) == 3
        assert data["data"]["total"] == 3
        
        # Verify question order
        for i, q in enumerate(data["data"]["questions"]):
            assert q["question_number"] == i + 1
            
    @pytest.mark.asyncio
    async def test_get_single_question(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test retrieving a single question."""
        # Create test and question
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        question = await create_test_question(test_db, test["_id"])
        
        # Get question
        response = await async_client.get(
            f"/api/v1/questions/{question['_id']}",
            headers=test_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == str(question["_id"])
        assert data["data"]["question_type"] == "mcq"
        

class TestQuestionWithAnswers:
    """Test question endpoints with answer functionality."""
    
    @pytest.mark.asyncio
    async def test_get_questions_with_answer_status(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test getting questions shows answer status."""
        # Create test with questions
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        
        # Create questions
        q1 = await create_test_question(test_db, test["_id"], question_number=1)
        q2 = await create_test_question(test_db, test["_id"], question_number=2)
        
        # Submit answer for first question
        answer_collection = test_db.get_collection("answers")
        await answer_collection.insert_one({
            "user_id": user["_id"],
            "test_id": test["_id"],
            "question_id": q1["_id"],
            "question_number": 1,
            "question_type": "mcq",
            "answer_data": {"selected_option": "a"},
            "validation": {"is_valid": True, "errors": [], "warnings": []},
            "time_spent_seconds": 45,
            "submitted_at": datetime.utcnow()
        })
        
        # Get questions with include_answers
        response = await async_client.get(
            f"/api/v1/questions/test/{test['_id']}?include_answers=true",
            headers=test_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # First question should have answer
        assert data["data"]["questions"][0]["has_answer"] is True
        assert "answer" in data["data"]["questions"][0]
        assert data["data"]["questions"][0]["answer"]["answer_data"]["selected_option"] == "a"
        
        # Second question should not have answer
        assert data["data"]["questions"][1]["has_answer"] is False
        assert "answer" not in data["data"]["questions"][1]
        
    @pytest.mark.asyncio
    async def test_validate_question_answer(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test validating answer before submission."""
        # Create test and question
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        question = await create_test_question(
            test_db,
            test["_id"],
            question_type=QuestionType.STATEMENT_SET,
            content={
                "question_text": "Rate these statements",
                "statements": [
                    {"id": "1", "text": "Statement 1", "dimensions": {"R": 1.0}},
                    {"id": "2", "text": "Statement 2", "dimensions": {"A": 1.0}},
                    {"id": "3", "text": "Statement 3", "dimensions": {"I": 1.0}}
                ]
            }
        )
        
        # Validate complete answer
        response = await async_client.post(
            f"/api/v1/questions/{question['_id']}/validate",
            headers=test_auth_headers,
            json={
                "answer_data": {
                    "ratings": {"1": 5, "2": 4, "3": 3}
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["is_valid"] is True
        assert len(data["data"]["errors"]) == 0
        
        # Validate incomplete answer
        response = await async_client.post(
            f"/api/v1/questions/{question['_id']}/validate",
            headers=test_auth_headers,
            json={
                "answer_data": {
                    "ratings": {"1": 5}  # Missing ratings for statements 2 and 3
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["is_valid"] is False
        assert len(data["data"]["errors"]) > 0
        

class TestQuestionRegeneration:
    """Test question regeneration functionality."""
    
    @pytest.mark.asyncio
    async def test_regenerate_question(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test regenerating a question."""
        # Create test and question
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        question = await create_test_question(
            test_db,
            test["_id"],
            is_ai_generated=False  # Static question
        )
        
        # Regenerate question
        response = await async_client.post(
            f"/api/v1/questions/{question['_id']}/regenerate",
            headers=test_auth_headers,
            json={
                "reason": "Too easy",
                "use_ai": False  # Use static for testing
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == str(question["_id"])
        
    @pytest.mark.asyncio
    async def test_regenerate_answered_question(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test regenerating a question that has been answered."""
        # Create test and question
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        question = await create_test_question(test_db, test["_id"])
        
        # Submit answer
        answer_collection = test_db.get_collection("answers")
        await answer_collection.insert_one({
            "user_id": user["_id"],
            "test_id": test["_id"],
            "question_id": question["_id"],
            "question_number": 1,
            "question_type": "mcq",
            "answer_data": {"selected_option": "a"},
            "submitted_at": datetime.utcnow()
        })
        
        # Try to regenerate
        response = await async_client.post(
            f"/api/v1/questions/{question['_id']}/regenerate",
            headers=test_auth_headers,
            json={
                "reason": "Want different question",
                "use_ai": False
            }
        )
        
        assert response.status_code == 400
        assert "already answered" in response.json()["error"]["message"].lower()
        

class TestQuestionBulkOperations:
    """Test bulk question operations."""
    
    @pytest.mark.asyncio
    async def test_bulk_question_status(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test getting status of multiple questions."""
        # Create test with questions
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        
        question_ids = []
        for i in range(3):
            q = await create_test_question(
                test_db,
                test["_id"],
                question_number=i+1
            )
            question_ids.append(str(q["_id"]))
            
        # Submit answer for first question
        answer_collection = test_db.get_collection("answers")
        await answer_collection.insert_one({
            "user_id": user["_id"],
            "test_id": test["_id"],
            "question_id": ObjectId(question_ids[0]),
            "question_number": 1,
            "question_type": "mcq",
            "answer_data": {"selected_option": "a"},
            "submitted_at": datetime.utcnow()
        })
        
        # Get bulk status
        response = await async_client.post(
            "/api/v1/questions/bulk-status",
            headers=test_auth_headers,
            json={
                "question_ids": question_ids
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["questions"]) == 3
        
        # Check status
        statuses = {q["id"]: q for q in data["data"]["questions"]}
        assert statuses[question_ids[0]]["has_answer"] is True
        assert statuses[question_ids[1]]["has_answer"] is False
        assert statuses[question_ids[2]]["has_answer"] is False
        

class TestQuestionFiltering:
    """Test question filtering and search."""
    
    @pytest.mark.asyncio
    async def test_filter_questions_by_type(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test filtering questions by type."""
        # Create test with mixed question types
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        
        # Create different question types
        await create_test_question(
            test_db,
            test["_id"],
            question_number=1,
            question_type=QuestionType.MCQ
        )
        await create_test_question(
            test_db,
            test["_id"],
            question_number=2,
            question_type=QuestionType.STATEMENT_SET
        )
        await create_test_question(
            test_db,
            test["_id"],
            question_number=3,
            question_type=QuestionType.MCQ
        )
        
        # Filter by MCQ
        response = await async_client.get(
            f"/api/v1/questions/test/{test['_id']}?question_type=mcq",
            headers=test_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["questions"]) == 2
        assert all(q["question_type"] == "mcq" for q in data["data"]["questions"])
        
    @pytest.mark.asyncio
    async def test_filter_unanswered_questions(
        self,
        async_client: AsyncClient,
        test_auth_headers,
        test_db
    ):
        """Test filtering for unanswered questions only."""
        # Create test with questions
        user = await create_test_user(test_db)
        test = await create_test_test(test_db, user["_id"])
        
        # Create questions
        questions = []
        for i in range(3):
            q = await create_test_question(
                test_db,
                test["_id"],
                question_number=i+1
            )
            questions.append(q)
            
        # Answer first two questions
        answer_collection = test_db.get_collection("answers")
        for i in range(2):
            await answer_collection.insert_one({
                "user_id": user["_id"],
                "test_id": test["_id"],
                "question_id": questions[i]["_id"],
                "question_number": i+1,
                "question_type": "mcq",
                "answer_data": {"selected_option": "a"},
                "submitted_at": datetime.utcnow()
            })
            
        # Get unanswered questions
        response = await async_client.get(
            f"/api/v1/questions/test/{test['_id']}?unanswered_only=true",
            headers=test_auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]["questions"]) == 1
        assert data["data"]["questions"][0]["question_number"] == 3