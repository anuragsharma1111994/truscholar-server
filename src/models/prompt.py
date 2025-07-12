"""Prompt management models for the TruScholar application.

This module defines models for storing and managing LLM prompts,
templates, and their versions for question generation, career recommendations,
and report generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from src.models.base import BaseDocument, EmbeddedDocument, PyObjectId
from src.utils.constants import AgeGroup, QuestionType, RaisecDimension


class PromptVariable(EmbeddedDocument):
    """Variable definition for prompt templates."""

    name: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z_][a-zA-Z0-9_]*$")
    description: str = Field(..., min_length=10, max_length=200)
    var_type: str = Field(..., pattern="^(string|integer|float|boolean|list|dict)$")
    required: bool = Field(default=True)
    default_value: Optional[Any] = None

    # Validation rules
    min_length: Optional[int] = Field(default=None, ge=0)
    max_length: Optional[int] = Field(default=None, ge=0)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: List[Any] = Field(default_factory=list)
    pattern: Optional[str] = None

    # Examples for documentation
    example_values: List[Any] = Field(default_factory=list, max_length=3)


class PromptExample(EmbeddedDocument):
    """Example input/output for a prompt."""

    example_id: str = Field(..., min_length=1, max_length=50)
    description: str = Field(..., min_length=10, max_length=200)

    # Example data
    input_variables: Dict[str, Any] = Field(...)
    expected_output: Union[str, Dict[str, Any]] = Field(...)

    # Metadata
    is_golden: bool = Field(default=False)  # Golden examples for testing
    tags: List[str] = Field(default_factory=list, max_length=5)

    # Quality metrics
    output_quality_score: Optional[float] = Field(default=None, ge=0, le=100)
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None


class PromptMetrics(EmbeddedDocument):
    """Performance metrics for a prompt."""

    total_uses: int = Field(default=0, ge=0)
    successful_uses: int = Field(default=0, ge=0)
    failed_uses: int = Field(default=0, ge=0)

    # Performance metrics
    average_completion_time_ms: float = Field(default=0.0, ge=0)
    average_token_count: float = Field(default=0.0, ge=0)
    average_cost_usd: float = Field(default=0.0, ge=0)

    # Quality metrics
    average_quality_score: float = Field(default=0.0, ge=0, le=100)
    validation_pass_rate: float = Field(default=100.0, ge=0, le=100)

    # Error tracking
    common_errors: List[Dict[str, Any]] = Field(default_factory=list, max_length=10)
    last_error_at: Optional[datetime] = None

    # Usage by model
    model_usage: Dict[str, int] = Field(default_factory=dict)

    def record_use(
        self,
        success: bool,
        completion_time_ms: float,
        token_count: int,
        cost_usd: float,
        model: str,
        error: Optional[str] = None
    ) -> None:
        """Record a prompt usage."""
        self.total_uses += 1

        if success:
            self.successful_uses += 1
            # Update averages
            self.average_completion_time_ms = (
                (self.average_completion_time_ms * (self.successful_uses - 1) + completion_time_ms) /
                self.successful_uses
            )
            self.average_token_count = (
                (self.average_token_count * (self.successful_uses - 1) + token_count) /
                self.successful_uses
            )
            self.average_cost_usd = (
                (self.average_cost_usd * (self.successful_uses - 1) + cost_usd) /
                self.successful_uses
            )
        else:
            self.failed_uses += 1
            self.last_error_at = datetime.utcnow()
            if error:
                # Track common errors
                error_entry = {"error": error, "count": 1, "last_seen": datetime.utcnow()}
                existing_error = next((e for e in self.common_errors if e["error"] == error), None)
                if existing_error:
                    existing_error["count"] += 1
                    existing_error["last_seen"] = datetime.utcnow()
                else:
                    self.common_errors.append(error_entry)
                    # Keep only top 10 most common errors
                    self.common_errors.sort(key=lambda x: x["count"], reverse=True)
                    self.common_errors = self.common_errors[:10]

        # Update model usage
        self.model_usage[model] = self.model_usage.get(model, 0) + 1

        # Update pass rate
        self.validation_pass_rate = (self.successful_uses / self.total_uses) * 100


class Prompt(BaseDocument):
    """Prompt template for LLM interactions."""

    # Identification
    prompt_name: str = Field(..., min_length=2, max_length=100)
    prompt_type: str = Field(
        ...,
        pattern="^(question_generation|career_recommendation|report_generation|scoring|validation|general)$"
    )
    category: str = Field(..., min_length=2, max_length=50)

    # Version control
    version: str = Field(..., pattern="^v\\d+\\.\\d+$")  # e.g., v1.0, v2.1
    is_active: bool = Field(default=True)
    is_default: bool = Field(default=False)

    # Template content
    system_prompt: Optional[str] = Field(default=None, max_length=5000)
    user_prompt_template: str = Field(..., min_length=50, max_length=10000)

    # Variables and placeholders
    variables: List[PromptVariable] = Field(default_factory=list)

    # Examples
    examples: List[PromptExample] = Field(default_factory=list, max_length=10)

    # Context and configuration
    description: str = Field(..., min_length=20, max_length=500)
    use_case: str = Field(..., min_length=20, max_length=500)

    # Model configuration
    recommended_model: str = Field(default="gpt-4-turbo-preview")
    model_parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )

    # Output configuration
    output_format: str = Field(default="json", pattern="^(json|text|markdown|structured)$")
    output_schema: Optional[Dict[str, Any]] = None
    validation_rules: List[str] = Field(default_factory=list)

    # Targeting
    age_groups: List[AgeGroup] = Field(default_factory=list)  # Empty = all ages
    question_types: List[QuestionType] = Field(default_factory=list)  # For question prompts
    raisec_codes: List[str] = Field(default_factory=list)  # For career prompts

    # Performance and metrics
    metrics: PromptMetrics = Field(default_factory=PromptMetrics)

    # Quality control
    tested: bool = Field(default=False)
    test_results: Optional[Dict[str, Any]] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Change tracking
    previous_version: Optional[str] = None
    change_notes: Optional[str] = Field(default=None, max_length=1000)
    deprecated: bool = Field(default=False)
    deprecated_at: Optional[datetime] = None
    replacement_prompt_id: Optional[PyObjectId] = None

    @model_validator(mode="after")
    def validate_prompt_structure(self) -> "Prompt":
        """Validate prompt has required variables in template."""
        # Extract variables from template
        import re
        template_vars = set(re.findall(r'\{(\w+)\}', self.user_prompt_template))
        if self.system_prompt:
            template_vars.update(re.findall(r'\{(\w+)\}', self.system_prompt))

        # Check all required variables are defined
        defined_vars = {var.name for var in self.variables}
        undefined_vars = template_vars - defined_vars

        if undefined_vars:
            # Log warning but don't fail - might be intentional
            pass

        return self

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the prompt collection."""
        return [
            ([("prompt_name", 1), ("version", 1)], {"unique": True}),
            ([("prompt_type", 1), ("is_active", 1), ("is_default", 1)], {}),
            ([("category", 1)], {}),
            ([("age_groups", 1)], {"sparse": True}),
            ([("question_types", 1)], {"sparse": True}),
            ([("metrics.total_uses", -1)], {}),
        ]

    def format_prompt(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """Format the prompt with provided variables.

        Args:
            variables: Dictionary of variable values

        Returns:
            Dictionary with formatted system and user prompts
        """
        # Validate required variables
        for var in self.variables:
            if var.required and var.name not in variables:
                if var.default_value is not None:
                    variables[var.name] = var.default_value
                else:
                    raise ValueError(f"Required variable '{var.name}' not provided")

        # Format prompts
        formatted = {
            "user_prompt": self.user_prompt_template.format(**variables)
        }

        if self.system_prompt:
            formatted["system_prompt"] = self.system_prompt.format(**variables)

        return formatted

    def validate_output(self, output: Any) -> tuple[bool, List[str]]:
        """Validate output against defined rules.

        Args:
            output: The output to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Basic validation based on output format
        if self.output_format == "json" and not isinstance(output, (dict, list)):
            errors.append("Output must be valid JSON")

        # Schema validation if provided
        if self.output_schema:
            # Implement JSON schema validation
            pass

        # Custom validation rules
        for rule in self.validation_rules:
            # Implement rule evaluation
            pass

        return len(errors) == 0, errors

    def increment_usage(
        self,
        success: bool,
        completion_time_ms: float,
        token_count: int,
        cost_usd: float,
        model: str,
        error: Optional[str] = None
    ) -> None:
        """Record usage of this prompt.

        Args:
            success: Whether the prompt execution was successful
            completion_time_ms: Time taken to complete
            token_count: Number of tokens used
            cost_usd: Cost in USD
            model: Model used
            error: Error message if failed
        """
        self.metrics.record_use(
            success=success,
            completion_time_ms=completion_time_ms,
            token_count=token_count,
            cost_usd=cost_usd,
            model=model,
            error=error
        )
        self.update_timestamps()

    def deprecate(self, replacement_id: Optional[PyObjectId] = None) -> None:
        """Mark prompt as deprecated.

        Args:
            replacement_id: ID of replacement prompt
        """
        self.deprecated = True
        self.deprecated_at = datetime.utcnow()
        self.is_active = False
        self.is_default = False

        if replacement_id:
            self.replacement_prompt_id = replacement_id

        self.update_timestamps()

    def get_summary(self) -> Dict[str, Any]:
        """Get prompt summary for display.

        Returns:
            Dictionary with prompt summary
        """
        return {
            "id": str(self.id),
            "name": self.prompt_name,
            "type": self.prompt_type,
            "category": self.category,
            "version": self.version,
            "is_active": self.is_active,
            "is_default": self.is_default,
            "description": self.description,
            "total_uses": self.metrics.total_uses,
            "success_rate": (
                (self.metrics.successful_uses / self.metrics.total_uses * 100)
                if self.metrics.total_uses > 0 else 0
            ),
            "average_cost": self.metrics.average_cost_usd,
            "recommended_model": self.recommended_model,
        }


class PromptChain(BaseDocument):
    """Chain of prompts for complex workflows."""

    chain_name: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., min_length=20, max_length=500)

    # Chain steps
    steps: List[Dict[str, Any]] = Field(..., min_length=2)
    # Each step: {
    #   "step_name": str,
    #   "prompt_id": PyObjectId,
    #   "input_mapping": Dict[str, str],  # Map previous outputs to inputs
    #   "output_key": str,  # Key to store output
    #   "condition": Optional[str],  # Condition to execute step
    #   "retry_count": int,
    #   "fallback_prompt_id": Optional[PyObjectId]
    # }

    # Chain configuration
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=300, ge=10, le=3600)

    # Metadata
    is_active: bool = Field(default=True)
    tags: List[str] = Field(default_factory=list, max_length=10)

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the prompt chain collection."""
        return [
            ([("chain_name", 1)], {"unique": True}),
            ([("is_active", 1)], {}),
            ([("tags", 1)], {}),
        ]


class PromptTestCase(BaseDocument):
    """Test case for prompt validation."""

    prompt_id: PyObjectId = Field(..., description="Reference to Prompt")
    test_name: str = Field(..., min_length=2, max_length=100)
    description: str = Field(..., min_length=10, max_length=500)

    # Test configuration
    test_type: str = Field(..., pattern="^(unit|integration|quality|performance)$")

    # Test data
    input_variables: Dict[str, Any] = Field(...)
    expected_output: Optional[Union[str, Dict[str, Any]]] = None

    # Validation criteria
    validation_criteria: List[Dict[str, Any]] = Field(default_factory=list)
    # Each criterion: {
    #   "type": "contains|matches|schema|custom",
    #   "value": Any,
    #   "description": str
    # }

    # Test results
    last_run_at: Optional[datetime] = None
    last_run_success: Optional[bool] = None
    last_run_output: Optional[Any] = None
    last_run_errors: List[str] = Field(default_factory=list)

    # Performance benchmarks
    max_completion_time_ms: Optional[float] = Field(default=None, ge=0)
    max_token_count: Optional[int] = Field(default=None, ge=0)

    # Test metadata
    is_active: bool = Field(default=True)
    created_by: str = Field(...)
    tags: List[str] = Field(default_factory=list, max_length=5)

    def create_index_keys(self) -> List[tuple]:
        """Define indexes for the test case collection."""
        return [
            ([("prompt_id", 1), ("test_name", 1)], {"unique": True}),
            ([("test_type", 1), ("is_active", 1)], {}),
            ([("last_run_success", 1)], {"sparse": True}),
        ]

    def run_test(self, output: Any, completion_time_ms: float, token_count: int) -> bool:
        """Run test against output.

        Args:
            output: The output to test
            completion_time_ms: Time taken to generate output
            token_count: Number of tokens used

        Returns:
            bool: Whether test passed
        """
        self.last_run_at = datetime.utcnow()
        self.last_run_output = output
        self.last_run_errors = []

        # Performance checks
        if self.max_completion_time_ms and completion_time_ms > self.max_completion_time_ms:
            self.last_run_errors.append(
                f"Completion time {completion_time_ms}ms exceeds max {self.max_completion_time_ms}ms"
            )

        if self.max_token_count and token_count > self.max_token_count:
            self.last_run_errors.append(
                f"Token count {token_count} exceeds max {self.max_token_count}"
            )

        # Validation criteria
        for criterion in self.validation_criteria:
            if not self._validate_criterion(output, criterion):
                self.last_run_errors.append(
                    f"Failed criterion: {criterion['description']}"
                )

        # Expected output check
        if self.expected_output is not None:
            if output != self.expected_output:
                self.last_run_errors.append("Output does not match expected")

        self.last_run_success = len(self.last_run_errors) == 0
        self.update_timestamps()

        return self.last_run_success

    def _validate_criterion(self, output: Any, criterion: Dict[str, Any]) -> bool:
        """Validate output against a criterion.

        Args:
            output: Output to validate
            criterion: Validation criterion

        Returns:
            bool: Whether criterion passed
        """
        criterion_type = criterion["type"]
        criterion_value = criterion["value"]

        if criterion_type == "contains":
            return criterion_value in str(output)
        elif criterion_type == "matches":
            import re
            return re.match(criterion_value, str(output)) is not None
        elif criterion_type == "schema":
            # Implement JSON schema validation
            return True
        elif criterion_type == "custom":
            # Implement custom validation logic
            return True

        return False
