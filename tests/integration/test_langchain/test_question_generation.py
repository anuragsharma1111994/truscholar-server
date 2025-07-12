"""Integration tests for LangChain question generation.

Tests the complete question generation pipeline including LangChain chains,
parsers, and integration with the question service.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

from src.langchain_handlers.question_generator import (
    QuestionGenerator, 
    QuestionGenerationError,
    generate_mcq_question,
    generate_statement_set_question,
    generate_plot_day_question
)
from src.langchain_handlers.chains.question_chain import QuestionChain
from src.langchain_handlers.parsers.question_parser import QuestionParser
from src.services.question_service import QuestionService
from src.utils.constants import QuestionType, AgeGroup, RaisecDimension
from src.utils.formatters import QuestionFormatter


class TestQuestionGenerator:
    """Test suite for QuestionGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create QuestionGenerator instance."""
        return QuestionGenerator(enable_caching=False, max_retries=2)
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response data."""
        return {
            "question_text": "Which type of work environment appeals to you most?",
            "options": [
                {
                    "id": "A",
                    "text": "Hands-on workshop or laboratory",
                    "scoring_guide": {"A": {"dimension": "R", "score": 3}}
                },
                {
                    "id": "B", 
                    "text": "Creative studio or design space",
                    "scoring_guide": {"B": {"dimension": "A", "score": 3}}
                },
                {
                    "id": "C",
                    "text": "Research library or data center", 
                    "scoring_guide": {"C": {"dimension": "I", "score": 3}}
                },
                {
                    "id": "D",
                    "text": "Community center or counseling office",
                    "scoring_guide": {"D": {"dimension": "S", "score": 3}}
                }
            ],
            "dimensions_evaluated": ["R", "A", "I", "S"],
            "instructions": "Select the option that best matches your preferences."
        }
    
    @pytest.mark.asyncio
    async def test_generate_mcq_question_success(self, generator, mock_llm_response):
        """Test successful MCQ question generation."""
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_llm_response
            
            result = await generator.generate_question(
                question_type=QuestionType.MCQ,
                age_group=AgeGroup.YOUNG_ADULT,
                question_number=1
            )
            
            assert result is not None
            assert result["question_text"] == mock_llm_response["question_text"]
            assert len(result["options"]) == 4
            assert "generation_metadata" in result
            assert result["generation_metadata"]["question_type"] == "mcq"
            
            # Verify chain was called with correct parameters
            mock_invoke.assert_called_once()
            call_args = mock_invoke.call_args[0][0]
            assert call_args["question_number"] == 1
            assert "dimensions_focus" in call_args
    
    @pytest.mark.asyncio
    async def test_generate_statement_set_question(self, generator):
        """Test statement set question generation."""
        mock_response = {
            "question_text": "Rate how much you agree with these statements:",
            "statements": [
                {"id": 1, "text": "I enjoy working with tools and machinery", "dimension": "R"},
                {"id": 2, "text": "I like expressing myself through art", "dimension": "A"},
                {"id": 3, "text": "I enjoy solving complex problems", "dimension": "I"},
                {"id": 4, "text": "I feel fulfilled when helping others", "dimension": "S"},
                {"id": 5, "text": "I like leading teams and projects", "dimension": "E"},
                {"id": 6, "text": "I prefer organized, systematic work", "dimension": "C"}
            ],
            "dimensions_evaluated": ["R", "A", "I", "S", "E", "C"],
            "instructions": "Rate each statement from 1 (strongly disagree) to 5 (strongly agree)."
        }
        
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            result = await generator.generate_question(
                question_type=QuestionType.STATEMENT_SET,
                age_group=AgeGroup.ADULT,
                question_number=2
            )
            
            assert result is not None
            assert len(result["statements"]) == 6
            assert all("dimension" in stmt for stmt in result["statements"])
    
    @pytest.mark.asyncio
    async def test_generate_plot_day_question(self, generator):
        """Test plot day question generation."""
        mock_response = {
            "question_text": "Arrange these activities in your ideal daily schedule:",
            "tasks": [
                {
                    "id": "task1",
                    "title": "Team Meeting",
                    "description": "Collaborate with colleagues",
                    "primary_dimension": "S",
                    "category": "collaboration",
                    "duration": "1 hour"
                },
                {
                    "id": "task2", 
                    "title": "Data Analysis",
                    "description": "Analyze complex datasets",
                    "primary_dimension": "I",
                    "category": "analysis",
                    "duration": "2 hours"
                },
                {
                    "id": "task3",
                    "title": "Creative Design",
                    "description": "Design visual materials", 
                    "primary_dimension": "A",
                    "category": "creative",
                    "duration": "1.5 hours"
                },
                {
                    "id": "task4",
                    "title": "Equipment Maintenance",
                    "description": "Fix and maintain tools",
                    "primary_dimension": "R",
                    "category": "technical",
                    "duration": "1 hour"
                },
                {
                    "id": "task5",
                    "title": "Project Planning",
                    "description": "Plan and organize projects",
                    "primary_dimension": "E", 
                    "category": "management",
                    "duration": "1 hour"
                },
                {
                    "id": "task6",
                    "title": "Documentation",
                    "description": "Create detailed records",
                    "primary_dimension": "C",
                    "category": "administrative",
                    "duration": "30 minutes"
                }
            ],
            "dimensions_evaluated": ["R", "A", "I", "S", "E", "C"]
        }
        
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            
            result = await generator.generate_question(
                question_type=QuestionType.PLOT_DAY,
                age_group=AgeGroup.TEEN,
                question_number=3
            )
            
            assert result is not None
            assert len(result["tasks"]) == 6
            assert all("primary_dimension" in task for task in result["tasks"])
            assert all(task["primary_dimension"] in ["R", "A", "I", "S", "E", "C"] for task in result["tasks"])
    
    @pytest.mark.asyncio
    async def test_generation_with_retries(self, generator):
        """Test question generation with retries on failure."""
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            # First call fails, second succeeds
            mock_invoke.side_effect = [
                Exception("LLM timeout"),
                {
                    "question_text": "Test question",
                    "options": [
                        {"id": "A", "text": "Option A", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                        {"id": "B", "text": "Option B", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
                    ],
                    "dimensions_evaluated": ["R", "A"]
                }
            ]
            
            result = await generator.generate_question(
                question_type=QuestionType.MCQ,
                age_group=AgeGroup.YOUNG_ADULT,
                question_number=1
            )
            
            assert result is not None
            assert result["generation_metadata"]["generation_attempt"] == 2
            assert mock_invoke.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generation_failure_after_max_retries(self, generator):
        """Test question generation failure after max retries."""
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("Persistent LLM error")
            
            with pytest.raises(QuestionGenerationError) as exc_info:
                await generator.generate_question(
                    question_type=QuestionType.MCQ,
                    age_group=AgeGroup.YOUNG_ADULT,
                    question_number=1
                )
            
            assert "Failed to generate mcq question after 2 attempts" in str(exc_info.value)
            assert mock_invoke.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_question_generation(self, generator):
        """Test batch question generation."""
        mock_responses = [
            {
                "question_text": f"Question {i}",
                "options": [
                    {"id": "A", "text": f"Option A {i}", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                    {"id": "B", "text": f"Option B {i}", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
                ],
                "dimensions_evaluated": ["R", "A"]
            }
            for i in range(1, 4)
        ]
        
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = mock_responses
            
            question_configs = [
                {
                    "question_type": QuestionType.MCQ,
                    "age_group": AgeGroup.YOUNG_ADULT,
                    "question_number": i
                }
                for i in range(1, 4)
            ]
            
            results = await generator.generate_batch_questions(
                question_configs, concurrent_limit=2
            )
            
            assert len(results) == 3
            assert all(result is not None for result in results)
            assert all("question_text" in result for result in results if result)
    
    @pytest.mark.asyncio
    async def test_validation_failure(self, generator):
        """Test handling of validation failure."""
        invalid_response = {
            "question_text": "Too short",  # Validation will fail
            "options": [],  # No options
            "dimensions_evaluated": []
        }
        
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = invalid_response
            
            result = await generator.generate_question(
                question_type=QuestionType.MCQ,
                age_group=AgeGroup.YOUNG_ADULT,
                question_number=1
            )
            
            # Should return None due to validation failure
            assert result is None
    
    @pytest.mark.asyncio
    async def test_generation_statistics(self, generator):
        """Test generation statistics retrieval."""
        stats = await generator.get_generation_statistics()
        
        assert "total_generated" in stats
        assert "success_rate" in stats
        assert "by_question_type" in stats
        assert "by_age_group" in stats
        assert isinstance(stats["success_rate"], float)
        assert 0 <= stats["success_rate"] <= 1


class TestQuestionChainIntegration:
    """Test integration between components."""
    
    @pytest.mark.asyncio
    async def test_chain_parser_integration(self):
        """Test integration between QuestionChain and QuestionParser."""
        mock_llm_output = '''
        {
            "question_text": "Which work environment appeals to you?",
            "options": [
                {"id": "A", "text": "Workshop", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                {"id": "B", "text": "Studio", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
            ],
            "dimensions_evaluated": ["R", "A"]
        }
        '''
        
        parser = QuestionParser(question_type=QuestionType.MCQ)
        
        # Test parser can handle LLM output
        result = parser.parse(mock_llm_output)
        
        assert result["question_text"] == "Which work environment appeals to you?"
        assert len(result["options"]) == 2
        assert "parser_metadata" in result
    
    @pytest.mark.asyncio
    async def test_formatter_integration(self):
        """Test integration with QuestionFormatter."""
        raw_llm_data = {
            "question_text": "Test question with   extra   spaces",
            "options": [
                {"id": "A", "text": "Option A with <script>", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                {"id": "B", "text": "Option B", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
            ],
            "dimensions_evaluated": ["R", "A"],
            "generation_metadata": {"generator": "test"}
        }
        
        formatter = QuestionFormatter()
        formatted = formatter.format_llm_response(raw_llm_data, QuestionType.MCQ)
        
        # Check text cleaning
        assert "extra   spaces" not in formatted["question_text"]
        assert "<script>" not in formatted["options"][0]["text"]
        assert formatted["generation_metadata"]["generator"] == "test"


class TestQuestionServiceIntegration:
    """Test integration with QuestionService."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database."""
        db_mock = MagicMock()
        db_mock.insert_many = AsyncMock(return_value=MagicMock(inserted_ids=["id1", "id2"]))
        db_mock.count_documents = AsyncMock(return_value=0)
        return db_mock
    
    @pytest.fixture
    def mock_cache(self):
        """Mock cache."""
        return MagicMock()
    
    @pytest.fixture
    def question_service(self, mock_db, mock_cache):
        """Create QuestionService with mocks."""
        return QuestionService(db=mock_db, cache=mock_cache)
    
    @pytest.mark.asyncio
    async def test_llm_generation_integration(self, question_service, mock_db):
        """Test LLM generation through QuestionService."""
        mock_response = {
            "question_text": "Which activity interests you most?",
            "options": [
                {"id": "A", "text": "Building things", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                {"id": "B", "text": "Creating art", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
            ],
            "dimensions_evaluated": ["R", "A"],
            "generation_metadata": {"generator": "langchain", "prompt_version": "1.0"}
        }
        
        with patch('src.core.settings.feature_flags.ENABLE_DYNAMIC_QUESTIONS', True):
            with patch.object(QuestionGenerator, 'generate_question', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = mock_response
                
                from bson import ObjectId
                test_id = ObjectId()
                
                questions = await question_service.generate_test_questions(
                    test_id=test_id,
                    age_group=AgeGroup.YOUNG_ADULT,
                    distribution={QuestionType.MCQ: 2}
                )
                
                assert len(questions) == 2
                assert all(not q.is_static for q in questions)
                assert all(q.llm_metadata is not None for q in questions)
                
                # Verify database interaction
                mock_db.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_to_static(self, question_service, mock_db):
        """Test fallback to static questions when LLM fails."""
        with patch('src.core.settings.feature_flags.ENABLE_DYNAMIC_QUESTIONS', True):
            with patch.object(QuestionGenerator, 'generate_question', new_callable=AsyncMock) as mock_generate:
                mock_generate.side_effect = Exception("LLM unavailable")
                
                from bson import ObjectId
                test_id = ObjectId()
                
                questions = await question_service.generate_test_questions(
                    test_id=test_id,
                    age_group=AgeGroup.YOUNG_ADULT,
                    distribution={QuestionType.MCQ: 1}
                )
                
                assert len(questions) == 1
                assert questions[0].is_static  # Should fallback to static


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_generate_mcq_question_function(self):
        """Test generate_mcq_question convenience function."""
        mock_response = {
            "question_text": "Test MCQ",
            "options": [
                {"id": "A", "text": "Option A", "scoring_guide": {"A": {"dimension": "R", "score": 3}}},
                {"id": "B", "text": "Option B", "scoring_guide": {"B": {"dimension": "A", "score": 3}}}
            ],
            "dimensions_evaluated": ["R", "A"]
        }
        
        with patch.object(QuestionGenerator, 'generate_question', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = mock_response
            
            result = await generate_mcq_question(
                age_group=AgeGroup.YOUNG_ADULT,
                question_number=1,
                dimensions_focus=[RaisecDimension.REALISTIC, RaisecDimension.ARTISTIC]
            )
            
            assert result is not None
            assert result["question_text"] == "Test MCQ"
            
            # Verify correct parameters were passed
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["question_type"] == QuestionType.MCQ
            assert call_kwargs["age_group"] == AgeGroup.YOUNG_ADULT
    
    @pytest.mark.asyncio
    async def test_generate_statement_set_question_function(self):
        """Test generate_statement_set_question convenience function."""
        with patch.object(QuestionGenerator, 'generate_question', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {"statements": []}
            
            await generate_statement_set_question(AgeGroup.TEEN)
            
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["question_type"] == QuestionType.STATEMENT_SET
            assert call_kwargs["age_group"] == AgeGroup.TEEN
    
    @pytest.mark.asyncio
    async def test_generate_plot_day_question_function(self):
        """Test generate_plot_day_question convenience function."""
        with patch.object(QuestionGenerator, 'generate_question', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {"tasks": []}
            
            await generate_plot_day_question(AgeGroup.ADULT)
            
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["question_type"] == QuestionType.PLOT_DAY
            assert call_kwargs["age_group"] == AgeGroup.ADULT


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_age_group(self):
        """Test handling of invalid age group."""
        generator = QuestionGenerator()
        
        with pytest.raises(Exception):  # Should raise validation error
            await generator.generate_question(
                question_type=QuestionType.MCQ,
                age_group="invalid_age_group",  # Invalid
                question_number=1
            )
    
    @pytest.mark.asyncio
    async def test_invalid_question_type(self):
        """Test handling of invalid question type."""
        generator = QuestionGenerator()
        
        with pytest.raises(Exception):  # Should raise validation error
            await generator.generate_question(
                question_type="invalid_type",  # Invalid
                age_group=AgeGroup.YOUNG_ADULT,
                question_number=1
            )
    
    @pytest.mark.asyncio
    async def test_malformed_llm_response(self):
        """Test handling of malformed LLM response."""
        generator = QuestionGenerator()
        
        with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
            # Return invalid JSON
            mock_invoke.return_value = {"invalid": "structure"}
            
            result = await generator.generate_question(
                question_type=QuestionType.MCQ,
                age_group=AgeGroup.YOUNG_ADULT,
                question_number=1
            )
            
            # Should return None due to validation failure
            assert result is None


@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """Test the complete question generation pipeline."""
    # This test simulates a full end-to-end question generation
    mock_response = {
        "question_text": "In your ideal workplace, which environment would you prefer?",
        "options": [
            {
                "id": "A",
                "text": "A hands-on workshop with tools and equipment",
                "scoring_guide": {"A": {"dimension": "R", "score": 3}}
            },
            {
                "id": "B", 
                "text": "A creative studio with art supplies and design tools",
                "scoring_guide": {"B": {"dimension": "A", "score": 3}}
            },
            {
                "id": "C",
                "text": "A research laboratory with computers and data",
                "scoring_guide": {"C": {"dimension": "I", "score": 3}}
            },
            {
                "id": "D",
                "text": "A community space where you help and counsel others",
                "scoring_guide": {"D": {"dimension": "S", "score": 3}}
            }
        ],
        "dimensions_evaluated": ["R", "A", "I", "S"],
        "instructions": "Choose the option that most appeals to you.",
        "generation_metadata": {
            "generator": "langchain",
            "prompt_version": "1.0",
            "generated_at": "2024-01-15T10:00:00Z"
        }
    }
    
    # Mock the entire chain
    with patch.object(QuestionChain, 'ainvoke', new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = mock_response
        
        # Generate question
        generator = QuestionGenerator()
        result = await generator.generate_question(
            question_type=QuestionType.MCQ,
            age_group=AgeGroup.YOUNG_ADULT,
            question_number=1,
            dimensions_focus=[RaisecDimension.REALISTIC, RaisecDimension.ARTISTIC, RaisecDimension.INVESTIGATIVE, RaisecDimension.SOCIAL]
        )
        
        # Verify result structure
        assert result is not None
        assert result["question_text"] == mock_response["question_text"]
        assert len(result["options"]) == 4
        assert result["dimensions_evaluated"] == ["R", "A", "I", "S"]
        assert "generation_metadata" in result
        
        # Verify generation metadata was added
        metadata = result["generation_metadata"]
        assert metadata["question_type"] == "mcq"
        assert metadata["age_group"] == "18-25"
        assert metadata["generation_attempt"] == 1
        assert "generated_at" in metadata
        
        # Test formatting
        formatter = QuestionFormatter()
        formatted = formatter.format_question_for_display(result, AgeGroup.YOUNG_ADULT)
        
        assert formatted["question_text"] == result["question_text"]
        assert len(formatted["options"]) == 4
        assert formatted["time_estimate_seconds"] > 0