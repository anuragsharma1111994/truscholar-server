"""Report parser for RAISEC assessment report generation responses.

This module provides parsing and validation for LLM-generated
comprehensive assessment reports.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

from src.utils.constants import ReportType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportParser(BaseOutputParser):
    """Parser for assessment report outputs.
    
    Handles parsing and validation of LLM responses for different
    types of assessment reports.
    """
    
    def __init__(self, report_type: ReportType):
        """Initialize the parser for a specific report type.
        
        Args:
            report_type: Type of report being parsed
        """
        self.report_type = report_type
        self.version = "1.0.0"
    
    def get_format_instructions(self) -> str:
        """Get format instructions for the LLM.
        
        Returns:
            str: Format instructions
        """
        return f"""Return a valid JSON object for a {self.report_type.value} assessment report.
        The JSON must follow the exact structure specified in the prompt.
        Do not include any text before or after the JSON."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output text synchronously.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            Dict[str, Any]: Parsed and validated report data
            
        Raises:
            OutputParserException: If parsing fails
        """
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Parse JSON
            parsed_json = json.loads(cleaned_text)
            
            # Validate structure
            validated_data = self._validate_report_structure(parsed_json)
            
            # Add metadata
            validated_data["parser_metadata"] = {
                "parser_version": self.version,
                "report_type": self.report_type.value,
                "parsed_at": datetime.utcnow().isoformat(),
                "validation_passed": True,
                "word_count": self._estimate_word_count(validated_data),
                "section_count": self._count_sections(validated_data)
            }
            
            return validated_data
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in LLM response: {str(e)}"
            logger.error(error_msg)
            raise OutputParserException(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to parse assessment report: {str(e)}"
            logger.error(error_msg)
            raise OutputParserException(error_msg)
    
    async def aparse(self, text: str) -> Dict[str, Any]:
        """Parse the output text asynchronously.
        
        Args:
            text: Raw text output from LLM
            
        Returns:
            Dict[str, Any]: Parsed and validated report data
        """
        # For now, just call the sync version
        return self.parse(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean the raw text to extract JSON.
        
        Args:
            text: Raw text from LLM
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove common prefixes/suffixes
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        
        # Find JSON object boundaries
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in response")
        
        return text[json_start:json_end]
    
    def _validate_report_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the report structure based on type.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Dict[str, Any]: Validated and normalized data
            
        Raises:
            ValueError: If validation fails
        """
        # Check for report wrapper
        if "report" not in data:
            raise ValueError("Response must contain a 'report' field")
        
        report = data["report"]
        if not isinstance(report, dict):
            raise ValueError("Report must be a dictionary")
        
        # Type-specific validation
        if self.report_type == ReportType.COMPREHENSIVE:
            return self._validate_comprehensive_report(data)
        elif self.report_type == ReportType.SUMMARY:
            return self._validate_summary_report(data)
        elif self.report_type == ReportType.DETAILED:
            return self._validate_detailed_report(data)
        else:
            raise ValueError(f"Unsupported report type: {self.report_type}")
    
    def _validate_comprehensive_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate comprehensive report structure.
        
        Args:
            data: Report data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        report = data["report"]
        
        # Required sections for comprehensive report
        required_sections = [
            "title", "executive_summary", "personality_profile",
            "career_recommendations", "development_plan", "next_steps",
            "resources", "conclusion"
        ]
        self._check_required_fields(report, required_sections)
        
        # Validate personality profile
        personality = report["personality_profile"]
        personality_fields = ["overview", "dominant_traits", "raisec_analysis", 
                            "strengths", "development_areas", "work_style_preferences"]
        self._check_required_fields(personality, personality_fields)
        
        # Validate RAISEC analysis
        raisec_analysis = personality["raisec_analysis"]
        raisec_fields = ["primary_dimension", "secondary_dimension", 
                        "tertiary_dimension", "dimension_interactions"]
        self._check_required_fields(raisec_analysis, raisec_fields)
        
        # Validate career recommendations
        career = report["career_recommendations"]
        career_fields = ["overview", "top_careers", "career_clusters", "entrepreneurship_potential"]
        self._check_required_fields(career, career_fields)
        
        if not isinstance(career["top_careers"], list) or len(career["top_careers"]) < 3:
            raise ValueError("Must have at least 3 top career recommendations")
        
        # Validate development plan
        dev_plan = report["development_plan"]
        dev_fields = ["immediate_actions", "short_term_goals", "long_term_vision", 
                     "skill_development", "education_recommendations"]
        self._check_required_fields(dev_plan, dev_fields)
        
        # Validate next steps
        next_steps = report["next_steps"]
        step_fields = ["week_1", "month_1", "month_3", "year_1"]
        self._check_required_fields(next_steps, step_fields)
        
        # Validate resources
        resources = report["resources"]
        resource_fields = ["books", "websites", "organizations", "networking"]
        self._check_required_fields(resources, resource_fields)
        
        return data
    
    def _validate_summary_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate summary report structure.
        
        Args:
            data: Report data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        report = data["report"]
        
        # Required sections for summary report
        required_sections = [
            "title", "key_insights", "raisec_profile", 
            "career_matches", "action_plan", "development_focus", "encouragement"
        ]
        self._check_required_fields(report, required_sections)
        
        # Validate key insights
        insights = report["key_insights"]
        insight_fields = ["personality_snapshot", "dominant_themes", "core_strengths"]
        self._check_required_fields(insights, insight_fields)
        
        # Validate RAISEC profile
        raisec = report["raisec_profile"]
        raisec_fields = ["code", "primary_trait", "secondary_trait", "profile_summary"]
        self._check_required_fields(raisec, raisec_fields)
        
        # Validate career matches
        careers = report["career_matches"]
        if not isinstance(careers, list) or len(careers) < 3:
            raise ValueError("Must have at least 3 career matches")
        
        for career in careers:
            career_fields = ["career", "why_its_perfect", "get_started"]
            self._check_required_fields(career, career_fields)
        
        # Validate action plan
        action_plan = report["action_plan"]
        action_fields = ["priority_1", "priority_2", "priority_3", "quick_wins"]
        self._check_required_fields(action_plan, action_fields)
        
        # Validate development focus
        dev_focus = report["development_focus"]
        dev_fields = ["skills_to_build", "experiences_to_seek", "knowledge_to_gain"]
        self._check_required_fields(dev_focus, dev_fields)
        
        return data
    
    def _validate_detailed_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate detailed report structure.
        
        Args:
            data: Report data
            
        Returns:
            Dict[str, Any]: Validated data
        """
        report = data["report"]
        
        # Required sections for detailed report
        required_sections = [
            "title", "introduction", "assessment_overview", "personality_analysis",
            "career_exploration", "development_strategy", "implementation_guide",
            "success_factors", "closing_thoughts"
        ]
        self._check_required_fields(report, required_sections)
        
        # Validate assessment overview
        overview = report["assessment_overview"]
        overview_fields = ["completion_summary", "reliability_indicators", "interpretation_notes"]
        self._check_required_fields(overview, overview_fields)
        
        # Validate personality analysis
        personality = report["personality_analysis"]
        personality_fields = ["raisec_breakdown", "personality_type", "core_motivations", 
                            "work_values", "preferred_environments"]
        self._check_required_fields(personality, personality_fields)
        
        # Validate RAISEC breakdown
        raisec_breakdown = personality["raisec_breakdown"]
        expected_dimensions = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
        for dim in expected_dimensions:
            if dim not in raisec_breakdown:
                raise ValueError(f"Missing RAISEC dimension: {dim}")
            
            dim_data = raisec_breakdown[dim]
            if not isinstance(dim_data, dict) or "score" not in dim_data or "interpretation" not in dim_data:
                raise ValueError(f"Invalid structure for {dim} dimension")
        
        # Validate career exploration
        career = report["career_exploration"]
        career_fields = ["highly_recommended", "worth_exploring", "industry_sectors"]
        self._check_required_fields(career, career_fields)
        
        # Validate development strategy
        dev_strategy = report["development_strategy"]
        dev_fields = ["strengths_to_leverage", "areas_to_develop", "learning_style", "development_timeline"]
        self._check_required_fields(dev_strategy, dev_fields)
        
        # Validate implementation guide
        impl_guide = report["implementation_guide"]
        phase_fields = ["phase_1", "phase_2", "phase_3"]
        self._check_required_fields(impl_guide, phase_fields)
        
        for phase in phase_fields:
            phase_data = impl_guide[phase]
            phase_required = ["title", "objectives", "actions", "milestones"]
            self._check_required_fields(phase_data, phase_required)
        
        # Validate success factors
        success = report["success_factors"]
        success_fields = ["critical_success_factors", "potential_obstacles", "support_systems"]
        self._check_required_fields(success, success_fields)
        
        return data
    
    def _check_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """Check if required fields are present.
        
        Args:
            data: Data to check
            required_fields: List of required field names
            
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
    
    def _estimate_word_count(self, data: Dict[str, Any]) -> int:
        """Estimate word count in the report.
        
        Args:
            data: Report data
            
        Returns:
            int: Estimated word count
        """
        def count_words_in_value(value: Any) -> int:
            if isinstance(value, str):
                return len(value.split())
            elif isinstance(value, dict):
                return sum(count_words_in_value(v) for v in value.values())
            elif isinstance(value, list):
                return sum(count_words_in_value(item) for item in value)
            else:
                return 0
        
        return count_words_in_value(data.get("report", {}))
    
    def _count_sections(self, data: Dict[str, Any]) -> int:
        """Count the number of main sections in the report.
        
        Args:
            data: Report data
            
        Returns:
            int: Number of sections
        """
        report = data.get("report", {})
        return len([k for k, v in report.items() if isinstance(v, dict) and k != "parser_metadata"])
    
    def get_version(self) -> str:
        """Get parser version.
        
        Returns:
            str: Parser version
        """
        return self.version
    
    @property
    def _type(self) -> str:
        """Return the parser type."""
        return "report_parser"


# Export the parser
__all__ = ["ReportParser"]