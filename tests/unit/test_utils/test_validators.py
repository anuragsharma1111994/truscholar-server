"""Unit tests for validators and validation utilities."""

import pytest
from typing import List, Dict, Any

from src.utils.validators import (
    validate_phone,
    validate_email,
    validate_age,
    validate_name,
    validate_raisec_scores,
    validate_test_answers,
    validate_user_input,
    validate_text_length,
    validate_numeric_range,
    validate_array_length,
    validate_file_upload,
    ValidationResult,
    BulkValidator
)
from src.utils.constants import QuestionType, RaisecDimension


class TestPhoneValidation:
    """Test phone number validation."""
    
    def test_valid_indian_phone_numbers(self):
        """Test valid Indian phone number formats."""
        valid_numbers = [
            "9876543210",
            "8765432109", 
            "7654321098",
            "6543210987"
        ]
        
        for number in valid_numbers:
            result = validate_phone(number)
            assert result.is_valid, f"Failed for {number}: {result.errors}"
            assert result.cleaned_value == number
            
    def test_invalid_phone_numbers(self):
        """Test invalid phone number formats."""
        invalid_numbers = [
            "123456789",      # Too short
            "98765432101",    # Too long
            "5876543210",     # Doesn't start with 6-9
            "abcdefghij",     # Letters
            "",               # Empty
            "987654321a",     # Contains letter
        ]
        
        for number in invalid_numbers:
            result = validate_phone(number)
            assert not result.is_valid, f"Should fail for {number}"


class TestEmailValidation:
    """Test email validation."""
    
    def test_valid_emails(self):
        """Test valid email formats."""
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.co.in",
            "user_123@test-domain.com",
            "a@b.co",
            "test.email.with+symbol@example4u.net"
        ]
        
        for email in valid_emails:
            result = validate_email(email)
            assert result.is_valid, f"Failed for {email}: {result.errors}"
            assert result.cleaned_value == email.lower()
            
    def test_invalid_emails(self):
        """Test invalid email formats."""
        invalid_emails = [
            "plainaddress",
            "@missinglocal.com", 
            "missing@domain",
            "missing.domain@.com",
            "two@@example.com",
            "spaces in@example.com",
            "user@",
            "@example.com",
            "",
            "user@.com"
        ]
        
        for email in invalid_emails:
            result = validate_email(email)
            assert not result.is_valid, f"Should fail for {email}"
            
    def test_email_normalization(self):
        """Test email normalization."""
        test_cases = [
            ("User@Example.COM", "user@example.com"),
            ("  user@example.com  ", "user@example.com"),
            ("USER.NAME@EXAMPLE.COM", "user.name@example.com")
        ]
        
        for input_email, expected in test_cases:
            result = validate_email(input_email)
            assert result.is_valid
            assert result.cleaned_value == expected


class TestAgeValidation:
    """Test age validation."""
    
    def test_valid_ages(self):
        """Test valid age ranges."""
        valid_ages = [13, 17, 18, 25, 26, 35]
        
        for age in valid_ages:
            result = validate_age(age)
            assert result.is_valid, f"Failed for age {age}: {result.errors}"
            
    def test_invalid_ages(self):
        """Test invalid age values."""
        invalid_ages = [12, 36, 0, -5, 150]
        
        for age in invalid_ages:
            result = validate_age(age)
            assert not result.is_valid, f"Should fail for age {age}"
            
    def test_age_with_age_group(self):
        """Test age validation with age group assignment."""
        test_cases = [
            (15, True),   # Valid for 13-17 group
            (20, True),   # Valid for 18-25 group  
            (30, True),   # Valid for 26-35 group
            (12, False),  # Too young
            (36, False),  # Too old
        ]
        
        for age, should_be_valid in test_cases:
            result = validate_age(age)
            assert result.is_valid == should_be_valid, f"Failed for age {age}"
            
            if should_be_valid:
                assert "age" in result.cleaned_value
                assert "age_group" in result.cleaned_value


class TestNameValidation:
    """Test name validation."""
    
    def test_valid_names(self):
        """Test valid name formats."""
        valid_names = [
            "John Doe",
            "Mary Jane Smith",
            "O'Connor",
            "Jean-Pierre",
            "Dr. Smith",
            "Anne-Marie"
        ]
        
        for name in valid_names:
            result = validate_name(name)
            assert result.is_valid, f"Failed for {name}: {result.errors}"
            
    def test_invalid_names(self):
        """Test invalid name formats."""
        invalid_names = [
            "",           # Empty
            "A",          # Too short
            "X" * 101,    # Too long
            "John123",    # Contains numbers
            "John@Doe",   # Invalid characters
        ]
        
        for name in invalid_names:
            result = validate_name(name)
            assert not result.is_valid, f"Should fail for {name}"
            
    def test_name_normalization(self):
        """Test name normalization."""
        test_cases = [
            ("  john  doe  ", "john doe"),
            ("JOHN DOE", "JOHN DOE"),  # Preserves case
            ("john\tdoe", "john doe"),   # Normalizes whitespace
        ]
        
        for input_name, expected in test_cases:
            result = validate_name(input_name)
            if result.is_valid:
                assert result.cleaned_value == expected


class TestRaisecScoresValidation:
    """Test RAISEC scores validation."""
    
    def test_valid_raisec_scores(self):
        """Test valid RAISEC score formats."""
        valid_scores = {
            "R": 85.5,
            "A": 70.2,
            "I": 92.0,
            "S": 65.8,
            "E": 78.3,
            "C": 80.1
        }
        
        result = validate_raisec_scores(valid_scores)
        assert result.is_valid
        assert result.cleaned_value == valid_scores
        
    def test_missing_dimensions(self):
        """Test missing RAISEC dimensions."""
        incomplete_scores = {
            "R": 85,
            "A": 70,
            "I": 92
            # Missing S, E, C
        }
        
        result = validate_raisec_scores(incomplete_scores)
        assert not result.is_valid
        assert "Missing RAISEC dimensions" in result.errors[0]
        
    def test_invalid_score_values(self):
        """Test invalid score values."""
        invalid_scores = {
            "R": 101,     # Too high
            "A": -5,      # Too low
            "I": "invalid",  # Not a number
            "S": 65,
            "E": 78,
            "C": 80
        }
        
        result = validate_raisec_scores(invalid_scores)
        assert not result.is_valid
        
    def test_score_warnings(self):
        """Test score pattern warnings."""
        # All scores very similar
        similar_scores = {dim: 50 for dim in ["R", "A", "I", "S", "E", "C"]}
        result = validate_raisec_scores(similar_scores)
        assert result.is_valid
        assert len(result.warnings) > 0
        
        # All scores very low
        low_scores = {dim: 20 for dim in ["R", "A", "I", "S", "E", "C"]}
        result = validate_raisec_scores(low_scores)
        assert result.is_valid
        assert len(result.warnings) > 0


class TestTextLengthValidation:
    """Test text length validation."""
    
    def test_valid_text_lengths(self):
        """Test valid text lengths."""
        test_cases = [
            ("Hello", 1, 10),
            ("Test message", 5, 50),
            ("", 0, 10),  # Empty allowed if min is 0
        ]
        
        for text, min_len, max_len in test_cases:
            result = validate_text_length(text, min_len, max_len)
            assert result.is_valid, f"Failed for '{text}' with min={min_len}, max={max_len}"
            
    def test_invalid_text_lengths(self):
        """Test invalid text lengths."""
        test_cases = [
            ("", 1, 10),        # Too short
            ("A" * 20, 1, 10),  # Too long
        ]
        
        for text, min_len, max_len in test_cases:
            result = validate_text_length(text, min_len, max_len)
            assert not result.is_valid, f"Should fail for '{text}' with min={min_len}, max={max_len}"


class TestNumericRangeValidation:
    """Test numeric range validation."""
    
    def test_valid_numeric_ranges(self):
        """Test valid numeric values."""
        test_cases = [
            (5, 1, 10),
            (0, 0, 100),
            (99.5, 0, 100),
            (-5, -10, 0),
        ]
        
        for value, min_val, max_val in test_cases:
            result = validate_numeric_range(value, min_val, max_val)
            assert result.is_valid, f"Failed for {value} with min={min_val}, max={max_val}"
            
    def test_invalid_numeric_ranges(self):
        """Test invalid numeric values."""
        test_cases = [
            (0, 1, 10),      # Too low
            (11, 1, 10),     # Too high
            ("text", 1, 10), # Not a number
        ]
        
        for value, min_val, max_val in test_cases:
            result = validate_numeric_range(value, min_val, max_val)
            assert not result.is_valid, f"Should fail for {value} with min={min_val}, max={max_val}"


class TestArrayLengthValidation:
    """Test array length validation."""
    
    def test_valid_array_lengths(self):
        """Test valid array lengths."""
        test_cases = [
            ([1, 2, 3], 1, 5),
            ([], 0, 10),
            (list(range(10)), 5, 15),
        ]
        
        for array, min_len, max_len in test_cases:
            result = validate_array_length(array, min_len, max_len)
            assert result.is_valid, f"Failed for array length {len(array)} with min={min_len}, max={max_len}"
            
    def test_invalid_array_lengths(self):
        """Test invalid array lengths."""
        test_cases = [
            ([], 1, 5),           # Too short
            ([1, 2, 3, 4, 5, 6], 1, 5),  # Too long
            ("not an array", 1, 5),      # Not an array
        ]
        
        for array, min_len, max_len in test_cases:
            result = validate_array_length(array, min_len, max_len)
            assert not result.is_valid


class TestFileUploadValidation:
    """Test file upload validation."""
    
    def test_valid_file_uploads(self):
        """Test valid file uploads."""
        valid_files = [
            {
                "filename": "document.pdf",
                "size": 1024 * 1024,  # 1MB
                "content_type": "application/pdf"
            },
            {
                "filename": "image.jpg", 
                "size": 500 * 1024,   # 500KB
                "content_type": "image/jpeg"
            }
        ]
        
        for file_data in valid_files:
            result = validate_file_upload(file_data)
            assert result.is_valid, f"Failed for {file_data['filename']}: {result.errors}"
            
    def test_invalid_file_uploads(self):
        """Test invalid file uploads."""
        invalid_files = [
            {
                "filename": "script.exe",
                "size": 1024,
                "content_type": "application/x-executable"
            },
            {
                "filename": "large.pdf",
                "size": 50 * 1024 * 1024,  # 50MB (too large)
                "content_type": "application/pdf"
            },
            {
                "filename": "",  # No filename
                "size": 1024,
                "content_type": "application/pdf"
            },
            {
                "filename": "empty.pdf",
                "size": 0,  # Empty file
                "content_type": "application/pdf"
            }
        ]
        
        for file_data in invalid_files:
            result = validate_file_upload(file_data)
            assert not result.is_valid, f"Should fail for {file_data.get('filename', 'no filename')}"


class TestUserInputValidation:
    """Test general user input validation."""
    
    def test_valid_user_input(self):
        """Test valid user input."""
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 25,
            "extra_field": "ignored"
        }
        
        required_fields = ["name", "email"]
        optional_fields = ["age"]
        
        result = validate_user_input(data, required_fields, optional_fields)
        assert result.is_valid
        assert "name" in result.cleaned_value
        assert "email" in result.cleaned_value
        assert "age" in result.cleaned_value
        assert "extra_field" not in result.cleaned_value
        
    def test_missing_required_fields(self):
        """Test missing required fields."""
        data = {
            "name": "John Doe"
            # Missing required email
        }
        
        required_fields = ["name", "email"]
        
        result = validate_user_input(data, required_fields)
        assert not result.is_valid
        assert "email" in result.errors[0]


class TestTestAnswersValidation:
    """Test test answers validation."""
    
    def test_valid_test_answers(self):
        """Test valid test answers."""
        answers = []
        for i in range(12):  # Minimum valid test length
            answers.append({
                "question_id": f"q_{i}",
                "question_type": "mcq",
                "answer_data": {"selected_option": "a"}
            })
            
        result = validate_test_answers(answers)
        assert result.is_valid
        
    def test_insufficient_answers(self):
        """Test insufficient number of answers."""
        answers = [
            {
                "question_id": "q_1",
                "question_type": "mcq", 
                "answer_data": {"selected_option": "a"}
            }
        ]
        
        result = validate_test_answers(answers)
        assert not result.is_valid
        assert "Insufficient answers" in result.errors[0]
        
    def test_invalid_answer_format(self):
        """Test invalid answer format."""
        answers = [
            "invalid_answer",  # Not a dict
            {
                "question_id": "q_2"
                # Missing required fields
            }
        ]
        # Add more valid answers to meet minimum
        for i in range(10):
            answers.append({
                "question_id": f"q_{i+3}",
                "question_type": "mcq",
                "answer_data": {"selected_option": "a"}
            })
            
        result = validate_test_answers(answers)
        assert not result.is_valid


class TestBulkValidator:
    """Test bulk validation utilities."""
    
    def test_bulk_user_validation(self):
        """Test bulk user validation."""
        users_data = [
            {
                "phone": "9876543210",
                "name": "John Doe",
                "age": 25
            },
            {
                "phone": "invalid_phone",
                "name": "Jane Doe",
                "age": 30
            },
            {
                "phone": "8765432109",
                "name": "Bob Smith"
                # No age (optional)
            }
        ]
        
        results = BulkValidator.validate_users(users_data)
        
        assert len(results["valid_users"]) == 2  # First and third users
        assert len(results["invalid_users"]) == 1  # Second user
        
        # Check invalid user has errors
        invalid_user = results["invalid_users"][0]
        assert invalid_user["index"] == 1
        assert any("Phone" in error for error in invalid_user["errors"])


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        # Success result
        success = ValidationResult.success("cleaned_value")
        assert success.is_valid
        assert success.cleaned_value == "cleaned_value"
        assert len(success.errors) == 0
        
        # Failure result
        failure = ValidationResult.failure("Error message")
        assert not failure.is_valid
        assert "Error message" in failure.errors
        
        # Failure with multiple errors
        failure_multi = ValidationResult.failure(["Error 1", "Error 2"])
        assert not failure_multi.is_valid
        assert len(failure_multi.errors) == 2
        
    def test_adding_errors_and_warnings(self):
        """Test adding errors and warnings."""
        result = ValidationResult.success()
        
        result.add_warning("Warning message")
        assert len(result.warnings) == 1
        assert result.is_valid  # Still valid
        
        result.add_error("Error message")
        assert len(result.errors) == 1
        assert not result.is_valid  # Now invalid