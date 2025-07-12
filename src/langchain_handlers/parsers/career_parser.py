"""Career output parsers for LangChain chains.

This module provides parsers to extract structured data from LLM responses
for career recommendations, insights, and analysis. Handles multiple output
formats and includes validation and error recovery.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Type
from datetime import datetime
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

from src.schemas.career_schemas import (
    CareerMatch,
    CareerDetailResponse,
    CareerPathResponse,
    CareerTrendsResponse,
    CareerFieldCategory,
    EducationLevel,
    ExperienceLevel,
    WorkEnvironment
)
from src.utils.constants import RaisecDimension
from src.utils.logger import get_logger
from src.utils.exceptions import ValidationError as TruScholarValidationError

logger = get_logger(__name__)


class CareerRecommendationParser(BaseOutputParser[List[CareerMatch]]):
    """Parser for career recommendation responses."""
    
    def __init__(self, max_recommendations: int = 25):
        self.max_recommendations = max_recommendations
    
    def parse(self, text: str) -> List[CareerMatch]:
        """Parse LLM response into list of CareerMatch objects."""
        try:
            # Try to extract JSON from response
            json_data = self._extract_json(text)
            
            if not json_data:
                # Fallback to structured text parsing
                return self._parse_structured_text(text)
            
            # Parse JSON data
            recommendations = []
            careers_data = json_data.get("recommendations", json_data.get("careers", []))
            
            for career_data in careers_data[:self.max_recommendations]:
                try:
                    career_match = self._parse_career_match(career_data)
                    if career_match:
                        recommendations.append(career_match)
                except Exception as e:
                    logger.warning(f"Failed to parse career: {str(e)}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Career recommendation parsing failed: {str(e)}")
            raise OutputParserException(f"Failed to parse career recommendations: {str(e)}")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Look for JSON blocks
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*?\})',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_structured_text(self, text: str) -> List[CareerMatch]:
        """Parse structured text format."""
        recommendations = []
        
        # Split by career entries
        career_sections = re.split(r'\n(?=\d+\.|\*|\-)', text)
        
        for section in career_sections:
            career_match = self._parse_text_career(section)
            if career_match:
                recommendations.append(career_match)
        
        return recommendations[:self.max_recommendations]
    
    def _parse_text_career(self, text: str) -> Optional[CareerMatch]:
        """Parse individual career from text."""
        try:
            # Extract career title
            title_match = re.search(r'(?:Career|Title):\s*([^\n]+)', text, re.IGNORECASE)
            if not title_match:
                return None
            
            career_title = title_match.group(1).strip()
            career_id = career_title.lower().replace(' ', '_').replace('-', '_')
            
            # Extract match score
            score_match = re.search(r'(?:Score|Match|Rating):\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            match_score = float(score_match.group(1)) if score_match else 75.0
            
            # Extract category
            category_match = re.search(r'(?:Category|Field):\s*([^\n]+)', text, re.IGNORECASE)
            category_str = category_match.group(1).strip() if category_match else "other"
            category = self._parse_category(category_str)
            
            # Extract RAISEC dimensions
            raisec_match = re.search(r'(?:RAISEC|Dimensions):\s*([^\n]+)', text, re.IGNORECASE)
            primary_dims, secondary_dims = self._parse_raisec_dimensions(
                raisec_match.group(1) if raisec_match else ""
            )
            
            return CareerMatch(
                career_id=career_id,
                career_title=career_title,
                category=category,
                raisec_match_score=match_score,
                primary_raisec_dimensions=primary_dims,
                secondary_raisec_dimensions=secondary_dims,
                match_explanation=f"Recommended based on profile analysis",
                confidence_level=0.75
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse text career: {str(e)}")
            return None
    
    def _parse_career_match(self, data: Dict[str, Any]) -> Optional[CareerMatch]:
        """Parse CareerMatch from dictionary data."""
        try:
            # Extract required fields
            career_id = data.get("career_id", data.get("id", ""))
            career_title = data.get("career_title", data.get("title", data.get("name", "")))
            
            if not career_id and career_title:
                career_id = career_title.lower().replace(' ', '_').replace('-', '_')
            
            if not career_title:
                return None
            
            # Parse category
            category_str = data.get("category", data.get("field", "other"))
            category = self._parse_category(category_str)
            
            # Parse RAISEC match score
            match_score = float(data.get("raisec_match_score", data.get("match_score", data.get("score", 75.0))))
            
            # Parse RAISEC dimensions
            primary_dims = self._parse_raisec_list(data.get("primary_raisec_dimensions", []))
            secondary_dims = self._parse_raisec_list(data.get("secondary_raisec_dimensions", []))
            
            # Parse optional fields
            match_explanation = data.get("match_explanation", data.get("explanation", "Career recommendation based on profile"))
            confidence_level = float(data.get("confidence_level", data.get("confidence", 0.75)))
            
            # Parse education requirements
            education_reqs = self._parse_education_levels(data.get("education_requirements", []))
            
            # Parse experience level
            experience_str = data.get("experience_level", "entry_level")
            experience_level = self._parse_experience_level(experience_str)
            
            # Parse work environments
            work_envs = self._parse_work_environments(data.get("work_environments", []))
            
            # Parse salary range
            salary_range = data.get("salary_range", {})
            if isinstance(salary_range, str):
                salary_range = self._parse_salary_string(salary_range)
            
            return CareerMatch(
                career_id=career_id,
                career_title=career_title,
                category=category,
                raisec_match_score=match_score,
                primary_raisec_dimensions=primary_dims,
                secondary_raisec_dimensions=secondary_dims,
                match_explanation=match_explanation,
                confidence_level=confidence_level,
                education_requirements=education_reqs,
                experience_level=experience_level,
                work_environments=work_envs,
                salary_range=salary_range,
                growth_potential=data.get("growth_potential", "Medium"),
                key_skills=data.get("key_skills", []),
                development_recommendations=data.get("development_recommendations", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to parse career match: {str(e)}")
            return None
    
    def _parse_category(self, category_str: str) -> CareerFieldCategory:
        """Parse career category from string."""
        category_mapping = {
            "technology": CareerFieldCategory.INFORMATION_TECHNOLOGY,
            "it": CareerFieldCategory.INFORMATION_TECHNOLOGY,
            "software": CareerFieldCategory.INFORMATION_TECHNOLOGY,
            "healthcare": CareerFieldCategory.HEALTHCARE_LIFE_SCIENCES,
            "health": CareerFieldCategory.HEALTHCARE_LIFE_SCIENCES,
            "medical": CareerFieldCategory.HEALTHCARE_LIFE_SCIENCES,
            "finance": CareerFieldCategory.FINANCE_ACCOUNTING,
            "banking": CareerFieldCategory.FINANCE_ACCOUNTING,
            "accounting": CareerFieldCategory.FINANCE_ACCOUNTING,
            "education": CareerFieldCategory.EDUCATION_TRAINING,
            "teaching": CareerFieldCategory.EDUCATION_TRAINING,
            "engineering": CareerFieldCategory.ENGINEERING_MANUFACTURING,
            "manufacturing": CareerFieldCategory.ENGINEERING_MANUFACTURING,
            "business": CareerFieldCategory.BUSINESS_MANAGEMENT,
            "management": CareerFieldCategory.BUSINESS_MANAGEMENT,
            "arts": CareerFieldCategory.ARTS_ENTERTAINMENT,
            "creative": CareerFieldCategory.ARTS_ENTERTAINMENT,
            "media": CareerFieldCategory.ARTS_ENTERTAINMENT,
            "sales": CareerFieldCategory.SALES_MARKETING,
            "marketing": CareerFieldCategory.SALES_MARKETING,
            "science": CareerFieldCategory.SCIENCE_RESEARCH,
            "research": CareerFieldCategory.SCIENCE_RESEARCH,
            "government": CareerFieldCategory.GOVERNMENT_PUBLIC_SERVICE,
            "public": CareerFieldCategory.GOVERNMENT_PUBLIC_SERVICE,
            "legal": CareerFieldCategory.LEGAL_COMPLIANCE,
            "law": CareerFieldCategory.LEGAL_COMPLIANCE,
            "consulting": CareerFieldCategory.CONSULTING_ADVISORY,
            "advisory": CareerFieldCategory.CONSULTING_ADVISORY
        }
        
        category_lower = category_str.lower().strip()
        for key, value in category_mapping.items():
            if key in category_lower:
                return value
        
        return CareerFieldCategory.OTHER
    
    def _parse_raisec_dimensions(self, raisec_str: str) -> tuple[List[RaisecDimension], List[RaisecDimension]]:
        """Parse RAISEC dimensions from string."""
        dimensions = []
        
        # Extract RAISEC codes
        raisec_codes = re.findall(r'[RAISEC]', raisec_str.upper())
        
        for code in raisec_codes:
            try:
                dimensions.append(RaisecDimension(code))
            except ValueError:
                continue
        
        # Split primary and secondary (first 2 are primary, rest secondary)
        primary = dimensions[:2]
        secondary = dimensions[2:]
        
        return primary, secondary
    
    def _parse_raisec_list(self, raisec_list: List[str]) -> List[RaisecDimension]:
        """Parse list of RAISEC dimension strings."""
        dimensions = []
        
        for item in raisec_list:
            if isinstance(item, str):
                # Handle full names or codes
                item_upper = item.upper().strip()
                if len(item_upper) == 1 and item_upper in 'RAISEC':
                    try:
                        dimensions.append(RaisecDimension(item_upper))
                    except ValueError:
                        continue
                else:
                    # Map full names to codes
                    name_mapping = {
                        "REALISTIC": RaisecDimension.REALISTIC,
                        "ARTISTIC": RaisecDimension.ARTISTIC,
                        "INVESTIGATIVE": RaisecDimension.INVESTIGATIVE,
                        "SOCIAL": RaisecDimension.SOCIAL,
                        "ENTERPRISING": RaisecDimension.ENTERPRISING,
                        "CONVENTIONAL": RaisecDimension.CONVENTIONAL
                    }
                    
                    for name, dimension in name_mapping.items():
                        if name in item_upper:
                            dimensions.append(dimension)
                            break
        
        return dimensions
    
    def _parse_education_levels(self, education_list: List[str]) -> List[EducationLevel]:
        """Parse education levels from list."""
        levels = []
        
        education_mapping = {
            "high school": EducationLevel.HIGH_SCHOOL,
            "bachelor": EducationLevel.BACHELOR,
            "master": EducationLevel.MASTER,
            "phd": EducationLevel.PHD,
            "doctorate": EducationLevel.PHD,
            "diploma": EducationLevel.DIPLOMA,
            "certificate": EducationLevel.CERTIFICATE
        }
        
        for item in education_list:
            if isinstance(item, str):
                item_lower = item.lower().strip()
                for key, value in education_mapping.items():
                    if key in item_lower:
                        levels.append(value)
                        break
        
        return levels or [EducationLevel.BACHELOR]
    
    def _parse_experience_level(self, experience_str: str) -> ExperienceLevel:
        """Parse experience level from string."""
        experience_mapping = {
            "entry": ExperienceLevel.ENTRY_LEVEL,
            "junior": ExperienceLevel.ENTRY_LEVEL,
            "mid": ExperienceLevel.MID_LEVEL,
            "intermediate": ExperienceLevel.MID_LEVEL,
            "senior": ExperienceLevel.SENIOR_LEVEL,
            "lead": ExperienceLevel.SENIOR_LEVEL,
            "executive": ExperienceLevel.EXECUTIVE
        }
        
        experience_lower = experience_str.lower().strip()
        for key, value in experience_mapping.items():
            if key in experience_lower:
                return value
        
        return ExperienceLevel.ENTRY_LEVEL
    
    def _parse_work_environments(self, environments_list: List[str]) -> List[WorkEnvironment]:
        """Parse work environments from list."""
        environments = []
        
        env_mapping = {
            "office": WorkEnvironment.OFFICE,
            "remote": WorkEnvironment.REMOTE,
            "hybrid": WorkEnvironment.HYBRID,
            "field": WorkEnvironment.FIELD,
            "laboratory": WorkEnvironment.LABORATORY,
            "lab": WorkEnvironment.LABORATORY,
            "hospital": WorkEnvironment.HOSPITAL,
            "outdoor": WorkEnvironment.OUTDOOR,
            "factory": WorkEnvironment.FACTORY,
            "home": WorkEnvironment.HOME_BASED
        }
        
        for item in environments_list:
            if isinstance(item, str):
                item_lower = item.lower().strip()
                for key, value in env_mapping.items():
                    if key in item_lower:
                        environments.append(value)
                        break
        
        return environments or [WorkEnvironment.OFFICE]
    
    def _parse_salary_string(self, salary_str: str) -> Dict[str, Any]:
        """Parse salary range from string."""
        # Extract numbers from salary string
        numbers = re.findall(r'[\d,]+', salary_str.replace(',', ''))
        
        if len(numbers) >= 2:
            return {
                "min": int(numbers[0]),
                "max": int(numbers[1]),
                "currency": "INR"
            }
        elif len(numbers) == 1:
            base_salary = int(numbers[0])
            return {
                "min": base_salary,
                "max": int(base_salary * 1.5),
                "currency": "INR"
            }
        
        return {"min": 500000, "max": 1000000, "currency": "INR"}

    def get_format_instructions(self) -> str:
        """Return format instructions for the LLM."""
        return """
        Please provide career recommendations in the following JSON format:
        
        {
            "recommendations": [
                {
                    "career_id": "unique_identifier",
                    "career_title": "Career Title",
                    "category": "category_name",
                    "raisec_match_score": 85.5,
                    "primary_raisec_dimensions": ["I", "E"],
                    "secondary_raisec_dimensions": ["C"],
                    "match_explanation": "Explanation of why this career matches",
                    "confidence_level": 0.85,
                    "education_requirements": ["bachelor"],
                    "experience_level": "entry_level",
                    "work_environments": ["office", "remote"],
                    "salary_range": {"min": 500000, "max": 1200000, "currency": "INR"},
                    "growth_potential": "High",
                    "key_skills": ["skill1", "skill2"],
                    "development_recommendations": ["recommendation1", "recommendation2"]
                }
            ]
        }
        """


class CareerInsightParser(BaseOutputParser[Dict[str, Any]]):
    """Parser for career insights and analysis."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into career insights."""
        try:
            # Try JSON first
            json_data = self._extract_json(text)
            if json_data:
                return json_data
            
            # Fallback to structured text parsing
            return self._parse_structured_insights(text)
            
        except Exception as e:
            logger.error(f"Career insight parsing failed: {str(e)}")
            raise OutputParserException(f"Failed to parse career insights: {str(e)}")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response."""
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*?\})',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_structured_insights(self, text: str) -> Dict[str, Any]:
        """Parse structured text format."""
        insights = {
            "summary": "",
            "key_points": [],
            "recommendations": [],
            "market_trends": [],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Extract summary
        summary_match = re.search(r'(?:Summary|Overview):\s*([^\n]+(?:\n[^\n:]+)*)', text, re.IGNORECASE)
        if summary_match:
            insights["summary"] = summary_match.group(1).strip()
        
        # Extract key points
        key_points = re.findall(r'[•\-\*]\s*([^\n]+)', text)
        insights["key_points"] = [point.strip() for point in key_points]
        
        # Extract recommendations
        rec_section = re.search(r'(?:Recommendations?|Suggestions?):(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if rec_section:
            recommendations = re.findall(r'[•\-\*]\s*([^\n]+)', rec_section.group(1))
            insights["recommendations"] = [rec.strip() for rec in recommendations]
        
        return insights

    def get_format_instructions(self) -> str:
        """Return format instructions for the LLM."""
        return """
        Please provide career insights in the following JSON format:
        
        {
            "summary": "Brief summary of the insights",
            "key_points": ["Key insight 1", "Key insight 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"],
            "market_trends": ["Trend 1", "Trend 2"],
            "confidence_level": 0.85,
            "data_sources": ["source1", "source2"]
        }
        """


class CareerComparisonParser(BaseOutputParser[Dict[str, Any]]):
    """Parser for career comparison responses."""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into career comparison data."""
        try:
            # Try JSON first
            json_data = self._extract_json(text)
            if json_data:
                return json_data
            
            # Fallback to structured parsing
            return self._parse_structured_comparison(text)
            
        except Exception as e:
            logger.error(f"Career comparison parsing failed: {str(e)}")
            raise OutputParserException(f"Failed to parse career comparison: {str(e)}")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response."""
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*?\})',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_structured_comparison(self, text: str) -> Dict[str, Any]:
        """Parse structured comparison text."""
        comparison = {
            "careers_compared": [],
            "comparison_matrix": {},
            "summary": "",
            "recommendation": "",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Extract summary
        summary_match = re.search(r'(?:Summary|Overview):\s*([^\n]+(?:\n[^\n:]+)*)', text, re.IGNORECASE)
        if summary_match:
            comparison["summary"] = summary_match.group(1).strip()
        
        # Extract recommendation
        rec_match = re.search(r'(?:Recommendation|Conclusion):\s*([^\n]+(?:\n[^\n:]+)*)', text, re.IGNORECASE)
        if rec_match:
            comparison["recommendation"] = rec_match.group(1).strip()
        
        return comparison

    def get_format_instructions(self) -> str:
        """Return format instructions for the LLM."""
        return """
        Please provide career comparison in the following JSON format:
        
        {
            "careers_compared": ["career1", "career2"],
            "comparison_matrix": {
                "salary": {"career1": "High", "career2": "Medium"},
                "growth": {"career1": "High", "career2": "Medium"}
            },
            "summary": "Brief comparison summary",
            "recommendation": "Which career is recommended and why",
            "pros_cons": {
                "career1": {"pros": ["pro1"], "cons": ["con1"]},
                "career2": {"pros": ["pro1"], "cons": ["con1"]}
            }
        }
        """


# Legacy parser for backward compatibility
class CareerParser(CareerRecommendationParser):
    """Legacy parser for backward compatibility."""
    
    def __init__(self, recommendation_type=None):
        super().__init__(max_recommendations=25)
        self.recommendation_type = recommendation_type
        self.version = "2.0.0"
    
    def get_version(self) -> str:
        return self.version
    
    @property
    def _type(self) -> str:
        return "career_parser"


# Utility functions for parser validation

def validate_parsed_career(career: CareerMatch) -> bool:
    """Validate a parsed career match."""
    try:
        # Check required fields
        if not career.career_id or not career.career_title:
            return False
        
        # Check score range
        if not (0 <= career.raisec_match_score <= 100):
            return False
        
        # Check confidence level
        if not (0 <= career.confidence_level <= 1):
            return False
        
        return True
        
    except Exception:
        return False


def sanitize_career_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and clean career data before parsing."""
    sanitized = {}
    
    # Clean string fields
    string_fields = ["career_id", "career_title", "match_explanation"]
    for field in string_fields:
        if field in data and isinstance(data[field], str):
            sanitized[field] = data[field].strip()[:500]  # Limit length
    
    # Clean numeric fields
    if "raisec_match_score" in data:
        try:
            score = float(data["raisec_match_score"])
            sanitized["raisec_match_score"] = max(0, min(100, score))
        except (ValueError, TypeError):
            sanitized["raisec_match_score"] = 75.0
    
    if "confidence_level" in data:
        try:
            confidence = float(data["confidence_level"])
            sanitized["confidence_level"] = max(0, min(1, confidence))
        except (ValueError, TypeError):
            sanitized["confidence_level"] = 0.75
    
    # Copy other fields
    for key, value in data.items():
        if key not in sanitized:
            sanitized[key] = value
    
    return sanitized


# Export main parsers
__all__ = [
    "CareerRecommendationParser",
    "CareerInsightParser", 
    "CareerComparisonParser",
    "CareerParser",  # Legacy compatibility
    "validate_parsed_career",
    "sanitize_career_data"
]