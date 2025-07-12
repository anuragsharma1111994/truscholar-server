"""LangChain prompts for TruScholar application.

This module contains prompt templates for different AI-powered features:
- Question generation prompts for RAISEC assessments
- Career recommendation prompts based on user profiles
- Report generation prompts from assessment results
"""

from .question_prompts import QuestionPrompts
from .career_prompts import CareerPrompts
from .report_prompts import ReportPrompts

__all__ = [
    "QuestionPrompts",
    "CareerPrompts", 
    "ReportPrompts",
]