"""Prompt management system for TruScholar LLM integration.

This module provides comprehensive prompt templating, versioning,
and management capabilities for AI-powered features.
"""

import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field

from jinja2 import Template, Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel, Field, field_validator

from .base_llm import LLMMessage, LLMRole
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptType(str, Enum):
    """Types of prompts in the system."""
    
    QUESTION_GENERATION = "question_generation"
    CAREER_ANALYSIS = "career_analysis"
    REPORT_GENERATION = "report_generation"
    VALIDATION = "validation"
    SCORING = "scoring"
    RECOMMENDATION = "recommendation"


class PromptVersion(str, Enum):
    """Prompt versions for A/B testing and rollback."""
    
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "latest"
    EXPERIMENTAL = "experimental"


@dataclass
class PromptMetadata:
    """Metadata for prompt templates."""
    
    name: str
    version: PromptVersion
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    model_compatibility: List[str] = field(default_factory=list)
    performance_notes: Optional[str] = None


class PromptTemplate(BaseModel):
    """A prompt template with metadata and content."""
    
    # Metadata
    name: str = Field(..., description="Template name")
    type: PromptType = Field(..., description="Prompt type")
    version: PromptVersion = Field(default=PromptVersion.V1, description="Template version")
    description: str = Field(..., description="Template description")
    
    # Template content
    system_template: Optional[str] = Field(None, description="System message template")
    user_template: str = Field(..., description="User message template")
    assistant_template: Optional[str] = Field(None, description="Assistant message template")
    
    # Configuration
    required_parameters: List[str] = Field(default_factory=list, description="Required template parameters")
    optional_parameters: List[str] = Field(default_factory=list, description="Optional template parameters")
    default_values: Dict[str, Any] = Field(default_factory=dict, description="Default parameter values")
    
    # Metadata
    author: str = Field(default="system", description="Template author")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Template tags")
    
    # Performance tracking
    usage_count: int = Field(default=0, description="Number of times used")
    success_rate: float = Field(default=0.0, description="Success rate (0.0-1.0)")
    avg_response_time: float = Field(default=0.0, description="Average response time in seconds")
    
    @field_validator("required_parameters", "optional_parameters")
    @classmethod
    def validate_parameters(cls, v: List[str]) -> List[str]:
        """Validate parameter names."""
        for param in v:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', param):
                raise ValueError(f"Invalid parameter name: {param}")
        return v
    
    def get_all_parameters(self) -> List[str]:
        """Get all parameters (required + optional)."""
        return self.required_parameters + self.optional_parameters
    
    def render(self, **kwargs) -> List[LLMMessage]:
        """Render the template with provided parameters.
        
        Args:
            **kwargs: Template parameters
            
        Returns:
            List[LLMMessage]: Rendered messages
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Check required parameters
        missing_params = set(self.required_parameters) - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Merge with default values
        params = {**self.default_values, **kwargs}
        
        # Create Jinja2 environment
        env = Environment()
        
        messages = []
        
        # Render system message if present
        if self.system_template:
            system_content = env.from_string(self.system_template).render(**params)
            messages.append(LLMMessage(role=LLMRole.SYSTEM, content=system_content))
        
        # Render user message
        user_content = env.from_string(self.user_template).render(**params)
        messages.append(LLMMessage(role=LLMRole.USER, content=user_content))
        
        # Render assistant message if present (for few-shot examples)
        if self.assistant_template:
            assistant_content = env.from_string(self.assistant_template).render(**params)
            messages.append(LLMMessage(role=LLMRole.ASSISTANT, content=assistant_content))
        
        return messages
    
    def validate_template_syntax(self) -> List[str]:
        """Validate Jinja2 template syntax.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        env = Environment()
        
        templates_to_check = [
            ("system_template", self.system_template),
            ("user_template", self.user_template),
            ("assistant_template", self.assistant_template)
        ]
        
        for name, template_str in templates_to_check:
            if template_str:
                try:
                    env.from_string(template_str)
                except Exception as e:
                    errors.append(f"Error in {name}: {str(e)}")
        
        return errors
    
    def extract_template_variables(self) -> List[str]:
        """Extract all variables used in templates.
        
        Returns:
            List[str]: List of variable names found in templates
        """
        variables = set()
        
        templates = [self.system_template, self.user_template, self.assistant_template]
        
        for template_str in templates:
            if template_str:
                # Simple regex to find Jinja2 variables
                var_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
                matches = re.findall(var_pattern, template_str)
                variables.update(matches)
        
        return list(variables)
    
    def update_performance_metrics(
        self,
        success: bool,
        response_time: float
    ) -> None:
        """Update performance metrics.
        
        Args:
            success: Whether the prompt execution was successful
            response_time: Response time in seconds
        """
        self.usage_count += 1
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            current_success = 1.0 if success else 0.0
            self.success_rate = (1 - alpha) * self.success_rate + alpha * current_success
        
        # Update average response time
        if self.usage_count == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (1 - alpha) * self.avg_response_time + alpha * response_time
        
        self.updated_at = datetime.utcnow()


class PromptManager:
    """Manages prompt templates, versions, and performance tracking."""
    
    def __init__(
        self,
        template_dir: Optional[str] = None,
        auto_load: bool = True,
        enable_caching: bool = True
    ):
        """Initialize prompt manager.
        
        Args:
            template_dir: Directory containing template files
            auto_load: Whether to automatically load templates from files
            enable_caching: Whether to cache rendered templates
        """
        self.template_dir = template_dir or os.path.join(os.path.dirname(__file__), "prompts")
        self.enable_caching = enable_caching
        
        # Template storage
        self._templates: Dict[str, Dict[PromptVersion, PromptTemplate]] = {}
        self._template_aliases: Dict[str, str] = {}  # alias -> template_name mapping
        
        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []
        
        # Caching
        self._render_cache: Dict[str, List[LLMMessage]] = {} if enable_caching else None
        
        # A/B testing
        self._ab_tests: Dict[str, Dict[str, Any]] = {}
        
        if auto_load:
            self.load_templates_from_directory()
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a prompt template.
        
        Args:
            template: Template to register
        """
        if template.name not in self._templates:
            self._templates[template.name] = {}
        
        self._templates[template.name][template.version] = template
        logger.info(f"Registered template: {template.name} ({template.version})")
    
    def get_template(
        self,
        name: str,
        version: PromptVersion = PromptVersion.LATEST
    ) -> Optional[PromptTemplate]:
        """Get a prompt template.
        
        Args:
            name: Template name (or alias)
            version: Template version
            
        Returns:
            Optional[PromptTemplate]: Template if found
        """
        # Resolve alias
        actual_name = self._template_aliases.get(name, name)
        
        if actual_name not in self._templates:
            logger.warning(f"Template not found: {actual_name}")
            return None
        
        templates = self._templates[actual_name]
        
        if version == PromptVersion.LATEST:
            # Get the latest version (highest version number)
            if not templates:
                return None
            
            # Prefer non-experimental versions
            non_experimental = {v: t for v, t in templates.items() if v != PromptVersion.EXPERIMENTAL}
            if non_experimental:
                latest_version = max(non_experimental.keys(), key=lambda x: x.value)
                return non_experimental[latest_version]
            else:
                # Fall back to experimental if it's the only one
                return templates.get(PromptVersion.EXPERIMENTAL)
        
        return templates.get(version)
    
    def render_template(
        self,
        name: str,
        version: PromptVersion = PromptVersion.LATEST,
        use_cache: bool = True,
        **kwargs
    ) -> List[LLMMessage]:
        """Render a prompt template.
        
        Args:
            name: Template name
            version: Template version
            use_cache: Whether to use cached results
            **kwargs: Template parameters
            
        Returns:
            List[LLMMessage]: Rendered messages
            
        Raises:
            ValueError: If template not found or rendering fails
        """
        template = self.get_template(name, version)
        if not template:
            raise ValueError(f"Template not found: {name} ({version})")
        
        # Generate cache key
        cache_key = None
        if self.enable_caching and use_cache:
            cache_key = f"{name}:{version.value}:{hash(json.dumps(kwargs, sort_keys=True))}"
            if cache_key in self._render_cache:
                logger.debug(f"Using cached render for {name}")
                return self._render_cache[cache_key]
        
        # Render template
        try:
            messages = template.render(**kwargs)
            
            # Cache result
            if cache_key and self._render_cache is not None:
                self._render_cache[cache_key] = messages
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to render template {name}: {e}")
            raise ValueError(f"Template rendering failed: {e}")
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates.
        
        Returns:
            List[Dict[str, Any]]: Template information
        """
        templates = []
        
        for name, versions in self._templates.items():
            for version, template in versions.items():
                templates.append({
                    "name": template.name,
                    "type": template.type.value,
                    "version": template.version.value,
                    "description": template.description,
                    "author": template.author,
                    "created_at": template.created_at.isoformat(),
                    "usage_count": template.usage_count,
                    "success_rate": template.success_rate,
                    "tags": template.tags
                })
        
        return templates
    
    def search_templates(
        self,
        query: Optional[str] = None,
        template_type: Optional[PromptType] = None,
        tags: Optional[List[str]] = None
    ) -> List[PromptTemplate]:
        """Search for templates.
        
        Args:
            query: Text query to search in name/description
            template_type: Filter by template type
            tags: Filter by tags
            
        Returns:
            List[PromptTemplate]: Matching templates
        """
        results = []
        
        for versions in self._templates.values():
            for template in versions.values():
                # Type filter
                if template_type and template.type != template_type:
                    continue
                
                # Tags filter
                if tags and not any(tag in template.tags for tag in tags):
                    continue
                
                # Text query filter
                if query:
                    query_lower = query.lower()
                    if (query_lower not in template.name.lower() and 
                        query_lower not in template.description.lower()):
                        continue
                
                results.append(template)
        
        # Sort by usage count and success rate
        results.sort(key=lambda t: (t.usage_count * t.success_rate), reverse=True)
        return results
    
    def create_template_alias(self, alias: str, template_name: str) -> None:
        """Create an alias for a template.
        
        Args:
            alias: Alias name
            template_name: Actual template name
        """
        if template_name not in self._templates:
            raise ValueError(f"Template not found: {template_name}")
        
        self._template_aliases[alias] = template_name
        logger.info(f"Created alias: {alias} -> {template_name}")
    
    def load_templates_from_directory(self) -> int:
        """Load templates from the template directory.
        
        Returns:
            int: Number of templates loaded
        """
        if not os.path.exists(self.template_dir):
            logger.warning(f"Template directory not found: {self.template_dir}")
            return 0
        
        loaded_count = 0
        
        for file_path in Path(self.template_dir).glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = PromptTemplate(**template_data)
                self.register_template(template)
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load template from {file_path}: {e}")
        
        logger.info(f"Loaded {loaded_count} templates from {self.template_dir}")
        return loaded_count
    
    def save_template_to_file(self, template: PromptTemplate, file_path: str) -> None:
        """Save a template to a JSON file.
        
        Args:
            template: Template to save
            file_path: Path to save the template
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(template.model_dump(), f, indent=2, default=str)
            
            logger.info(f"Saved template {template.name} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
            raise
    
    def validate_all_templates(self) -> Dict[str, List[str]]:
        """Validate all registered templates.
        
        Returns:
            Dict[str, List[str]]: Validation errors by template name
        """
        validation_results = {}
        
        for name, versions in self._templates.items():
            for version, template in versions.items():
                errors = template.validate_template_syntax()
                if errors:
                    validation_results[f"{name}:{version.value}"] = errors
        
        return validation_results
    
    def start_ab_test(
        self,
        test_name: str,
        template_a: str,
        template_b: str,
        traffic_split: float = 0.5
    ) -> None:
        """Start an A/B test between two templates.
        
        Args:
            test_name: Name of the A/B test
            template_a: First template name
            template_b: Second template name
            traffic_split: Percentage of traffic to send to template A (0.0-1.0)
        """
        if template_a not in self._templates or template_b not in self._templates:
            raise ValueError("Both templates must exist for A/B testing")
        
        self._ab_tests[test_name] = {
            "template_a": template_a,
            "template_b": template_b,
            "traffic_split": traffic_split,
            "started_at": datetime.utcnow(),
            "results_a": {"count": 0, "successes": 0},
            "results_b": {"count": 0, "successes": 0}
        }
        
        logger.info(f"Started A/B test: {test_name}")
    
    def get_ab_test_template(self, test_name: str) -> Optional[str]:
        """Get template for A/B test based on traffic split.
        
        Args:
            test_name: A/B test name
            
        Returns:
            Optional[str]: Template name to use
        """
        if test_name not in self._ab_tests:
            return None
        
        test = self._ab_tests[test_name]
        
        import random
        if random.random() < test["traffic_split"]:
            return test["template_a"]
        else:
            return test["template_b"]
    
    def record_ab_test_result(
        self,
        test_name: str,
        template_name: str,
        success: bool
    ) -> None:
        """Record A/B test result.
        
        Args:
            test_name: A/B test name
            template_name: Template that was used
            success: Whether the request was successful
        """
        if test_name not in self._ab_tests:
            return
        
        test = self._ab_tests[test_name]
        
        if template_name == test["template_a"]:
            results = test["results_a"]
        elif template_name == test["template_b"]:
            results = test["results_b"]
        else:
            return
        
        results["count"] += 1
        if success:
            results["successes"] += 1
    
    def get_ab_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results.
        
        Args:
            test_name: A/B test name
            
        Returns:
            Optional[Dict[str, Any]]: Test results
        """
        if test_name not in self._ab_tests:
            return None
        
        test = self._ab_tests[test_name]
        
        # Calculate success rates
        results_a = test["results_a"]
        results_b = test["results_b"]
        
        success_rate_a = results_a["successes"] / results_a["count"] if results_a["count"] > 0 else 0
        success_rate_b = results_b["successes"] / results_b["count"] if results_b["count"] > 0 else 0
        
        return {
            "test_name": test_name,
            "template_a": test["template_a"],
            "template_b": test["template_b"],
            "traffic_split": test["traffic_split"],
            "started_at": test["started_at"],
            "results_a": {
                **results_a,
                "success_rate": success_rate_a
            },
            "results_b": {
                **results_b,
                "success_rate": success_rate_b
            },
            "winner": test["template_a"] if success_rate_a > success_rate_b else test["template_b"]
        }
    
    def clear_cache(self) -> None:
        """Clear the render cache."""
        if self._render_cache is not None:
            self._render_cache.clear()
            logger.info("Prompt render cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prompt manager statistics.
        
        Returns:
            Dict[str, Any]: Statistics
        """
        total_templates = sum(len(versions) for versions in self._templates.values())
        total_usage = sum(
            template.usage_count
            for versions in self._templates.values()
            for template in versions.values()
        )
        
        avg_success_rate = 0.0
        if total_templates > 0:
            avg_success_rate = sum(
                template.success_rate
                for versions in self._templates.values()
                for template in versions.values()
            ) / total_templates
        
        return {
            "total_templates": total_templates,
            "unique_template_names": len(self._templates),
            "total_usage": total_usage,
            "average_success_rate": avg_success_rate,
            "cache_size": len(self._render_cache) if self._render_cache else 0,
            "active_ab_tests": len(self._ab_tests),
            "template_aliases": len(self._template_aliases)
        }


# Export main classes
__all__ = [
    "PromptManager",
    "PromptTemplate",
    "PromptType",
    "PromptVersion",
    "PromptMetadata"
]