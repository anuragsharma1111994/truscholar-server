#!/usr/bin/env python3
"""Prompt seeding script for TruScholar Career Assessment platform.

This script initializes and validates the prompt management system,
ensuring all prompt templates are properly loaded and accessible.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.langchain_handlers.validation import (
    PromptFileValidator, 
    validate_prompts_directory,
    PromptValidationError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptSeeder:
    """Handles prompt seeding and initialization."""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize the prompt seeder.
        
        Args:
            base_path: Base path for prompt files (defaults to data/prompts)
        """
        self.base_path = Path(base_path) if base_path else project_root / "data" / "prompts"
        self.validator = PromptFileValidator(str(self.base_path))
        
    def seed_prompts(self, version: str = "v1.0", force: bool = False) -> bool:
        """Seed prompts for a specific version.
        
        Args:
            version: Version to seed (default: v1.0)
            force: Force overwrite existing prompts
            
        Returns:
            bool: True if seeding successful
        """
        logger.info(f"Starting prompt seeding for version {version}")
        
        try:
            # Check if version already exists
            version_path = self.base_path / version
            if version_path.exists() and not force:
                logger.warning(f"Version {version} already exists. Use --force to overwrite.")
                return False
            
            # Create version directory
            version_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created version directory: {version_path}")
            
            # Seed prompt files
            self._create_question_prompts(version_path)
            self._create_career_prompts(version_path)
            self._create_report_prompts(version_path)
            
            # Validate seeded prompts
            validation_results = self.validator.validate_all_prompts(version)
            
            if validation_results["valid"]:
                logger.info(f"✅ All prompts for version {version} are valid")
                
                # Update current symlink if this is a newer version
                self._update_current_symlink(version)
                
                return True
            else:
                logger.error(f"❌ Validation failed for version {version}")
                for error in validation_results["errors"]:
                    logger.error(f"  - {error}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to seed prompts: {e}")
            return False
    
    def _create_question_prompts(self, version_path: Path) -> None:
        """Create question prompts file."""
        question_prompts = {
            "version": "1.0",
            "created_at": "2024-01-15T00:00:00Z",
            "description": "Comprehensive question generation prompts for RAISEC assessment across all question types and age groups",
            "author": "TruScholar AI Team",
            "prompts": {
                "mcq": {
                    "system_prompt": "You are an expert psychologist specializing in RAISEC (Holland Code) career assessments. You create multiple-choice questions that accurately evaluate personality dimensions for career guidance.\n\nRAISEC Framework:\n- R (Realistic): Practical, hands-on, mechanical, outdoor-oriented personality\n- A (Artistic): Creative, expressive, aesthetic, innovative personality\n- I (Investigative): Analytical, intellectual, research-oriented personality\n- S (Social): People-focused, helping, teaching, interpersonal personality\n- E (Enterprising): Leadership, persuasive, business-oriented personality\n- C (Conventional): Organized, systematic, detail-oriented personality\n\nYour MCQ questions must:\n1. Be psychologically valid and culturally appropriate for Indian context\n2. Have 4 answer options representing different RAISEC dimensions\n3. Be clear, unambiguous, and age-appropriate\n4. Avoid cultural bias and gender stereotypes\n5. Focus on genuine preferences rather than abilities\n6. Include realistic scenarios relevant to career choices\n7. Be engaging and thought-provoking\n8. Have options that are mutually exclusive and comprehensive\n\nFormat your response as valid JSON with question_text, options array, correct_answer, and dimensions_evaluated.",
                    "age_groups": {
                        "13-17": {
                            "user_template": "Generate MCQ question #{question_number} for teenagers (13-17 years).\n\nFocus Dimensions: {dimensions_focus}\nContext: {context}\nConstraints: {constraints}\n\nCreate an engaging multiple-choice question about career preferences that a {age_group} year old can relate to. Use scenarios involving:\n- School subjects and activities\n- Part-time jobs and internships\n- Hobby and interest choices\n- Future education decisions\n- Social activities and leadership roles\n\nEnsure the question and options are:\n- Age-appropriate and relatable\n- Free from cultural bias\n- Clear and easy to understand\n- Focused on genuine interests rather than gender stereotypes\n\nGenerate the question following this JSON structure:\n\n{\n  \"question_text\": \"Engaging question about career preferences\",\n  \"options\": [\n    \"Option A representing dimension 1\",\n    \"Option B representing dimension 2\",\n    \"Option C representing dimension 3\",\n    \"Option D representing dimension 4\"\n  ],\n  \"dimensions_evaluated\": [\"R\", \"A\", \"I\", \"S\"],\n  \"scoring_guide\": {\n    \"A\": {\"dimension\": \"R\", \"score\": 3},\n    \"B\": {\"dimension\": \"A\", \"score\": 3},\n    \"C\": {\"dimension\": \"I\", \"score\": 3},\n    \"D\": {\"dimension\": \"S\", \"score\": 3}\n  }\n}"
                        },
                        "18-25": {
                            "user_template": "Generate MCQ question #{question_number} for young adults (18-25 years).\n\nFocus Dimensions: {dimensions_focus}\nContext: {context}\nConstraints: {constraints}\n\nCreate a sophisticated multiple-choice question about career preferences for {age_group} year olds. Use scenarios involving:\n- College coursework and majors\n- Career exploration and first jobs\n- Professional development choices\n- Work environment preferences\n- Life goals and aspirations\n\nEnsure the question addresses:\n- Early career decision-making\n- Professional identity formation\n- Work-life balance considerations\n- Skill development priorities\n\nGenerate the question following this JSON structure:\n\n{\n  \"question_text\": \"Professional scenario question about career choices\",\n  \"options\": [\n    \"Option A representing realistic work\",\n    \"Option B representing artistic work\", \n    \"Option C representing investigative work\",\n    \"Option D representing social work\"\n  ],\n  \"dimensions_evaluated\": [\"R\", \"A\", \"I\", \"S\"],\n  \"scoring_guide\": {\n    \"A\": {\"dimension\": \"R\", \"score\": 3},\n    \"B\": {\"dimension\": \"A\", \"score\": 3},\n    \"C\": {\"dimension\": \"I\", \"score\": 3},\n    \"D\": {\"dimension\": \"S\", \"score\": 3}\n  }\n}"
                        },
                        "26-35": {
                            "user_template": "Generate MCQ question #{question_number} for professionals (26-35 years).\n\nFocus Dimensions: {dimensions_focus}\nContext: {context}\nConstraints: {constraints}\n\nCreate a sophisticated multiple-choice question about career preferences for {age_group} year old professionals. Use scenarios involving:\n- Mid-career transitions and decisions\n- Leadership and management roles\n- Work-life integration\n- Professional growth opportunities\n- Industry expertise development\n\nEnsure the question addresses:\n- Career advancement choices\n- Professional fulfillment factors\n- Management and leadership preferences\n- Industry and sector preferences\n\nGenerate the question following this JSON structure:\n\n{\n  \"question_text\": \"Professional development scenario question\",\n  \"options\": [\n    \"Option A focusing on practical implementation\",\n    \"Option B focusing on creative innovation\",\n    \"Option C focusing on analytical problem-solving\", \n    \"Option D focusing on people leadership\"\n  ],\n  \"dimensions_evaluated\": [\"R\", \"A\", \"I\", \"S\"],\n  \"scoring_guide\": {\n    \"A\": {\"dimension\": \"R\", \"score\": 3},\n    \"B\": {\"dimension\": \"A\", \"score\": 3},\n    \"C\": {\"dimension\": \"I\", \"score\": 3},\n    \"D\": {\"dimension\": \"S\", \"score\": 3}\n  }\n}"
                        }
                    }
                },
                "statement_set": {
                    "system_prompt": "You are an expert psychologist creating statement-based assessments for RAISEC career evaluations. Your statement sets help individuals understand their career preferences through agreement-based responses.\n\nRAISEC Framework:\n- R (Realistic): Practical, hands-on, mechanical work\n- A (Artistic): Creative, expressive, aesthetic work\n- I (Investigative): Analytical, research-oriented work\n- S (Social): People-focused, helping work\n- E (Enterprising): Leadership, business-oriented work\n- C (Conventional): Organized, systematic work\n\nYour statement sets must:\n1. Include 5-7 diverse statements covering different RAISEC dimensions\n2. Use \"I prefer...\" or \"I enjoy...\" format\n3. Be specific and actionable rather than abstract\n4. Cover different aspects of work preferences\n5. Be culturally appropriate for Indian context\n6. Allow for nuanced preference measurement\n7. Include both positive preferences and work environment factors\n\nUsers respond with: Strongly Agree (4), Agree (3), Neutral (2), Disagree (1), Strongly Disagree (0)",
                    "age_groups": {
                        "13-17": {
                            "user_template": "Generate statement set #{question_number} for teenagers (13-17 years).\n\nFocus Dimensions: {dimensions_focus}\nContext: {context}\nAge Group: {age_group}\n\nCreate 6 statements about work and activity preferences that teenagers can relate to. Include statements about:\n- School subject preferences\n- Extracurricular activities\n- Summer job interests\n- Learning styles\n- Social interaction preferences\n- Future aspirations\n\nGenerate the statement set following this JSON structure:\n\n{\n  \"instructions\": \"Rate how much you agree with each statement\",\n  \"statements\": [\n    {\n      \"text\": \"I prefer working with my hands to build or fix things\",\n      \"dimension\": \"R\"\n    },\n    {\n      \"text\": \"I enjoy creating art, music, or writing stories\", \n      \"dimension\": \"A\"\n    },\n    {\n      \"text\": \"I like solving complex math or science problems\",\n      \"dimension\": \"I\"\n    },\n    {\n      \"text\": \"I prefer helping friends with their problems\",\n      \"dimension\": \"S\"\n    },\n    {\n      \"text\": \"I enjoy leading group projects or teams\",\n      \"dimension\": \"E\"\n    },\n    {\n      \"text\": \"I like organizing my study materials and schedules\",\n      \"dimension\": \"C\"\n    }\n  ],\n  \"dimensions_evaluated\": [\"R\", \"A\", \"I\", \"S\", \"E\", \"C\"],\n  \"response_scale\": {\n    \"4\": \"Strongly Agree\",\n    \"3\": \"Agree\", \n    \"2\": \"Neutral\",\n    \"1\": \"Disagree\",\n    \"0\": \"Strongly Disagree\"\n  }\n}"
                        },
                        "18-25": {
                            "user_template": "Generate statement set #{question_number} for young adults (18-25 years).\n\nFocus Dimensions: {dimensions_focus}\nContext: {context}\nAge Group: {age_group}\n\nCreate 6 statements about career and work preferences for college students and early professionals. Include statements about:\n- Work environment preferences\n- Career goal orientations\n- Skill development interests\n- Professional interaction styles\n- Industry preferences\n- Work-life balance factors\n\nGenerate the statement set following this JSON structure:\n\n{\n  \"instructions\": \"Rate how much you agree with each statement about your career preferences\",\n  \"statements\": [\n    {\n      \"text\": \"I prefer jobs that involve hands-on technical work\",\n      \"dimension\": \"R\"\n    },\n    {\n      \"text\": \"I want a career that allows creative expression\",\n      \"dimension\": \"A\"\n    },\n    {\n      \"text\": \"I enjoy analyzing data and conducting research\",\n      \"dimension\": \"I\"\n    },\n    {\n      \"text\": \"I want to work directly with people to help them\",\n      \"dimension\": \"S\"\n    },\n    {\n      \"text\": \"I aspire to lead teams and influence business decisions\",\n      \"dimension\": \"E\"\n    },\n    {\n      \"text\": \"I prefer structured work with clear procedures\",\n      \"dimension\": \"C\"\n    }\n  ],\n  \"dimensions_evaluated\": [\"R\", \"A\", \"I\", \"S\", \"E\", \"C\"],\n  \"response_scale\": {\n    \"4\": \"Strongly Agree\",\n    \"3\": \"Agree\",\n    \"2\": \"Neutral\", \n    \"1\": \"Disagree\",\n    \"0\": \"Strongly Disagree\"\n  }\n}"
                        },
                        "26-35": {
                            "user_template": "Generate statement set #{question_number} for professionals (26-35 years).\n\nFocus Dimensions: {dimensions_focus}\nContext: {context}\nAge Group: {age_group}\n\nCreate 6 statements about career satisfaction and professional preferences for mid-career professionals. Include statements about:\n- Leadership and management preferences\n- Work impact and meaning\n- Professional development priorities\n- Industry and sector preferences\n- Work environment and culture\n- Career advancement factors\n\nGenerate the statement set following this JSON structure:\n\n{\n  \"instructions\": \"Rate how much you agree with each statement about your professional preferences\",\n  \"statements\": [\n    {\n      \"text\": \"I find satisfaction in hands-on problem-solving and implementation\",\n      \"dimension\": \"R\"\n    },\n    {\n      \"text\": \"I want my work to involve creative innovation and original thinking\",\n      \"dimension\": \"A\"\n    },\n    {\n      \"text\": \"I prefer roles that require deep analysis and strategic thinking\",\n      \"dimension\": \"I\"\n    },\n    {\n      \"text\": \"I want to make a direct impact on people's lives through my work\",\n      \"dimension\": \"S\"\n    },\n    {\n      \"text\": \"I aspire to influence organizational direction and business strategy\",\n      \"dimension\": \"E\"\n    },\n    {\n      \"text\": \"I prefer well-structured roles with clear metrics and processes\",\n      \"dimension\": \"C\"\n    }\n  ],\n  \"dimensions_evaluated\": [\"R\", \"A\", \"I\", \"S\", \"E\", \"C\"],\n  \"response_scale\": {\n    \"4\": \"Strongly Agree\",\n    \"3\": \"Agree\",\n    \"2\": \"Neutral\",\n    \"1\": \"Disagree\", \n    \"0\": \"Strongly Disagree\"\n  }\n}"
                        }
                    }
                }
            }
        }
        
        file_path = version_path / "question_prompts.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(question_prompts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created question prompts: {file_path}")
    
    def _create_career_prompts(self, version_path: Path) -> None:
        """Create career prompts file."""
        career_prompts = {
            "version": "1.0",
            "created_at": "2024-01-15T00:00:00Z",
            "description": "Comprehensive career recommendation prompts for RAISEC analysis across all recommendation types",
            "author": "TruScholar AI Team",
            "prompts": {
                "traditional": {
                    "system_prompt": "You are an expert career counselor specializing in traditional, stable career paths based on RAISEC personality assessments. You provide well-established career recommendations with proven pathways and stability.",
                    "user_template": "Generate traditional career recommendations for:\nRAISEC Code: {raisec_code}\nTop Dimensions: {top_three_dimensions}\nUser Profile: {user_age} years, {education_level}, {experience_level}\nLocation: {user_location}\n\nProvide 3-5 traditional, stable career options with clear progression paths."
                },
                "innovative": {
                    "system_prompt": "You are an expert career counselor specializing in innovative, emerging career paths based on RAISEC personality assessments. You provide forward-thinking career recommendations in new and evolving fields.",
                    "user_template": "Generate innovative career recommendations for:\nRAISEC Code: {raisec_code}\nTop Dimensions: {top_three_dimensions}\nUser Profile: {user_age} years, {education_level}, {experience_level}\nLocation: {user_location}\n\nProvide 3-5 innovative, emerging career options with growth potential."
                },
                "hybrid": {
                    "system_prompt": "You are an expert career counselor specializing in hybrid career paths that combine traditional stability with innovative opportunities. You provide balanced career recommendations.",
                    "user_template": "Generate hybrid career recommendations for:\nRAISEC Code: {raisec_code}\nTop Dimensions: {top_three_dimensions}\nUser Profile: {user_age} years, {education_level}, {experience_level}\nLocation: {user_location}\n\nProvide 3-5 hybrid career options balancing stability and innovation."
                }
            }
        }
        
        file_path = version_path / "career_prompts.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(career_prompts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created career prompts: {file_path}")
    
    def _create_report_prompts(self, version_path: Path) -> None:
        """Create report prompts file."""
        report_prompts = {
            "version": "1.0",
            "created_at": "2024-01-15T00:00:00Z",
            "description": "Comprehensive report generation prompts for RAISEC assessment results across all report types",
            "author": "TruScholar AI Team",
            "prompts": {
                "comprehensive": {
                    "system_prompt": "You are an expert career psychologist creating comprehensive RAISEC assessment reports. Provide detailed, actionable career guidance with in-depth analysis.",
                    "user_template": "Generate a comprehensive assessment report for:\nUser: {user_name}, {user_age} years\nRAISEC Code: {raisec_code}\nTop Dimensions: {primary_dimension}, {secondary_dimension}, {tertiary_dimension}\nCareer Stage: {career_stage}\n\nCreate a detailed report with personality analysis, career recommendations, and development plan."
                },
                "summary": {
                    "system_prompt": "You are an expert career psychologist creating concise RAISEC assessment summaries. Provide focused, actionable career guidance with key insights.",
                    "user_template": "Generate a summary assessment report for:\nUser: {user_name}, {user_age} years\nRAISEC Code: {raisec_code}\nTop Dimensions: {primary_dimension}, {secondary_dimension}, {tertiary_dimension}\nCareer Stage: {career_stage}\n\nCreate a concise summary with key insights and immediate action steps."
                },
                "detailed": {
                    "system_prompt": "You are an expert career psychologist creating detailed RAISEC assessment reports. Provide thorough career guidance with moderate depth and practical implementation steps.",
                    "user_template": "Generate a detailed assessment report for:\nUser: {user_name}, {user_age} years\nRAISEC Code: {raisec_code}\nTop Dimensions: {primary_dimension}, {secondary_dimension}, {tertiary_dimension}\nCareer Stage: {career_stage}\n\nCreate a thorough report with personality analysis, career exploration, and implementation guide."
                }
            }
        }
        
        file_path = version_path / "report_prompts.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_prompts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created report prompts: {file_path}")
    
    def _update_current_symlink(self, version: str) -> None:
        """Update the current symlink to point to the specified version.
        
        Args:
            version: Version to point current symlink to
        """
        current_link = self.base_path / "current"
        version_path = self.base_path / version
        
        # Remove existing symlink if it exists
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        
        # Create new symlink
        current_link.symlink_to(version, target_is_directory=True)
        logger.info(f"Updated current symlink to point to {version}")
    
    def validate_prompts(self, version: str = "current") -> bool:
        """Validate prompts for a specific version.
        
        Args:
            version: Version to validate
            
        Returns:
            bool: True if validation passes
        """
        logger.info(f"Validating prompts for version {version}")
        
        try:
            results = self.validator.validate_all_prompts(version)
            
            if results["valid"]:
                logger.info(f"✅ All prompts for version {version} are valid")
                logger.info(f"Files validated: {results['files_validated']}")
                logger.info(f"Files passed: {results['files_passed']}")
                
                if results["warnings"]:
                    logger.warning("Warnings found:")
                    for warning in results["warnings"]:
                        logger.warning(f"  - {warning}")
                
                return True
            else:
                logger.error(f"❌ Validation failed for version {version}")
                logger.error("Errors found:")
                for error in results["errors"]:
                    logger.error(f"  - {error}")
                
                if results["warnings"]:
                    logger.warning("Warnings found:")
                    for warning in results["warnings"]:
                        logger.warning(f"  - {warning}")
                
                return False
                
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False
    
    def list_versions(self) -> List[str]:
        """List all available prompt versions.
        
        Returns:
            List[str]: List of version directories
        """
        if not self.base_path.exists():
            return []
        
        versions = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name != "current":
                versions.append(item.name)
        
        return sorted(versions)
    
    def get_current_version(self) -> Optional[str]:
        """Get the currently active version.
        
        Returns:
            str: Current version or None if not set
        """
        current_link = self.base_path / "current"
        
        if current_link.is_symlink():
            return current_link.readlink().name
        
        return None


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="TruScholar Prompt Seeding Script")
    parser.add_argument("--version", default="v1.0", help="Version to seed (default: v1.0)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing prompts")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing prompts")
    parser.add_argument("--list-versions", action="store_true", help="List all available versions")
    parser.add_argument("--base-path", help="Base path for prompt files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        seeder = PromptSeeder(args.base_path)
        
        if args.list_versions:
            versions = seeder.list_versions()
            current = seeder.get_current_version()
            
            print("Available versions:")
            for version in versions:
                marker = " (current)" if version == current else ""
                print(f"  - {version}{marker}")
            
            return 0
        
        if args.validate_only:
            success = seeder.validate_prompts(args.version)
            return 0 if success else 1
        
        # Seed prompts
        success = seeder.seed_prompts(args.version, args.force)
        
        if success:
            logger.info(f"✅ Successfully seeded prompts for version {args.version}")
            return 0
        else:
            logger.error(f"❌ Failed to seed prompts for version {args.version}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())