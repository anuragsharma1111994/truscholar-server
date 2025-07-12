"""Question generation prompts for RAISEC assessments.

This module provides comprehensive prompt templates for generating
different types of RAISEC assessment questions across age groups.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.utils.constants import QuestionType, AgeGroup


class QuestionPrompts:
    """Prompt templates for question generation."""
    
    # Base system prompt for question generation
    BASE_SYSTEM_PROMPT = """You are an expert psychologist and career counselor specializing in RAISEC (Holland Code) assessments. You create engaging, age-appropriate questions that accurately measure career interests and personality traits.

RAISEC Framework:
- R (Realistic): Practical, hands-on, mechanical, physical activities
- A (Artistic): Creative, expressive, aesthetic, innovative activities  
- I (Investigative): Analytical, intellectual, research, problem-solving activities
- S (Social): Helping, teaching, interpersonal, service-oriented activities
- E (Enterprising): Leadership, persuasive, business, influential activities
- C (Conventional): Organized, systematic, detail-oriented, structured activities

Your questions must:
1. Be culturally appropriate for Indian context
2. Use age-appropriate language and scenarios
3. Clearly differentiate between RAISEC dimensions
4. Be engaging and relatable
5. Avoid bias and stereotypes
6. Include diverse scenarios and examples

Format your response as valid JSON following the exact structure specified."""
    
    @classmethod
    def get_template(cls, question_type: QuestionType, age_group: AgeGroup) -> ChatPromptTemplate:
        """Get appropriate prompt template for question type and age group.
        
        Args:
            question_type: Type of question to generate
            age_group: Target age group
            
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        system_prompt = cls._get_system_prompt(question_type, age_group)
        human_prompt = cls._get_human_prompt(question_type, age_group)
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    @classmethod
    def _get_system_prompt(cls, question_type: QuestionType, age_group: AgeGroup) -> str:
        """Get system prompt for specific question type and age group.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            
        Returns:
            str: System prompt template
        """
        age_context = cls._get_age_context(age_group)
        type_context = cls._get_type_context(question_type)
        
        return f"""{cls.BASE_SYSTEM_PROMPT}

AGE GROUP CONTEXT:
{age_context}

QUESTION TYPE CONTEXT:
{type_context}

IMPORTANT: Your response must be valid JSON. Do not include any text before or after the JSON structure."""
    
    @classmethod
    def _get_age_context(cls, age_group: AgeGroup) -> str:
        """Get age-specific context for prompts.
        
        Args:
            age_group: Target age group
            
        Returns:
            str: Age-specific context
        """
        if age_group == AgeGroup.TEEN:
            return """Target Age: 13-17 years (High school students)
- Use simple, clear language
- Focus on school subjects, hobbies, and activities they know
- Include scenarios about future aspirations and interests
- Avoid complex professional terminology
- Use relatable examples from their daily life and school experience"""
        
        elif age_group == AgeGroup.YOUNG_ADULT:
            return """Target Age: 18-25 years (College/Early career)
- Use moderate complexity language
- Include college, internship, and early career scenarios
- Focus on skill development and career exploration
- Use examples from academic and professional contexts
- Include technology and modern workplace scenarios"""
        
        elif age_group == AgeGroup.ADULT:
            return """Target Age: 26-35 years (Established professionals)
- Use professional language and terminology
- Focus on career advancement and leadership scenarios
- Include workplace challenges and management situations
- Use examples from diverse professional fields
- Address career transition and growth opportunities"""
        
        return ""
    
    @classmethod
    def _get_type_context(cls, question_type: QuestionType) -> str:
        """Get question type-specific context.
        
        Args:
            question_type: Type of question
            
        Returns:
            str: Type-specific context
        """
        if question_type == QuestionType.MCQ:
            return """MULTIPLE CHOICE QUESTION (MCQ):
Generate a question with 4 options (a, b, c, d). Each option should:
- Clearly represent different RAISEC dimensions
- Be mutually exclusive
- Be equally plausible and attractive
- Avoid obvious "correct" answers

JSON Structure:
{
  "question_text": "Question content here",
  "instructions": "Optional instructions",
  "options": [
    {"id": "a", "text": "Option A text", "primary_dimension": "R", "dimension_weights": {"R": 1.0}},
    {"id": "b", "text": "Option B text", "primary_dimension": "A", "dimension_weights": {"A": 1.0}},
    {"id": "c", "text": "Option C text", "primary_dimension": "I", "dimension_weights": {"I": 1.0}},
    {"id": "d", "text": "Option D text", "primary_dimension": "S", "dimension_weights": {"S": 1.0}}
  ],
  "dimensions_evaluated": ["R", "A", "I", "S"]
}"""
        
        elif question_type == QuestionType.STATEMENT_SET:
            return """STATEMENT SET (Likert Scale):
Generate 5-6 statements for rating on a 1-5 scale (Strongly Disagree to Strongly Agree).
- Each statement should clearly relate to specific RAISEC dimensions
- Include both positive and reverse-scored statements
- Ensure statements are balanced across dimensions
- Use "I enjoy..." or "I prefer..." format

JSON Structure:
{
  "question_text": "Rate how much you agree with each statement",
  "instructions": "Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree)",
  "statements": [
    {"id": 1, "text": "Statement 1", "dimension": "R", "reverse_scored": false},
    {"id": 2, "text": "Statement 2", "dimension": "A", "reverse_scored": false},
    {"id": 3, "text": "Statement 3", "dimension": "I", "reverse_scored": true},
    {"id": 4, "text": "Statement 4", "dimension": "S", "reverse_scored": false},
    {"id": 5, "text": "Statement 5", "dimension": "E", "reverse_scored": false}
  ],
  "dimensions_evaluated": ["R", "A", "I", "S", "E"]
}"""
        
        elif question_type == QuestionType.SCENARIO_MCQ:
            return """SCENARIO MULTIPLE CHOICE:
Create a realistic scenario followed by 4 response options.
- Scenario should be relatable and engaging
- Options should represent different personality types/approaches
- Each option should clearly align with RAISEC dimensions

JSON Structure:
{
  "question_text": "Scenario description here. What would you most likely do?",
  "instructions": "Choose the option that best describes what you would do",
  "options": [
    {"id": "a", "text": "Response A", "primary_dimension": "R", "dimension_weights": {"R": 1.0}},
    {"id": "b", "text": "Response B", "primary_dimension": "A", "dimension_weights": {"A": 1.0}},
    {"id": "c", "text": "Response C", "primary_dimension": "I", "dimension_weights": {"I": 1.0}},
    {"id": "d", "text": "Response D", "primary_dimension": "S", "dimension_weights": {"S": 1.0}}
  ],
  "dimensions_evaluated": ["R", "A", "I", "S"]
}"""
        
        elif question_type == QuestionType.SCENARIO_MULTI_SELECT:
            return """SCENARIO MULTI-SELECT:
Create a scenario where multiple responses are appropriate.
- Allow selection of 2-3 options from 5-6 choices
- Options can have overlapping dimensions with different weights
- Scenario should be complex enough to warrant multiple approaches

JSON Structure:
{
  "question_text": "Scenario description. Which approaches would you use? (Select 2-3)",
  "instructions": "Select 2-3 options that best describe your approach",
  "options": [
    {"id": "a", "text": "Approach A", "primary_dimension": "R", "dimension_weights": {"R": 0.8, "C": 0.2}},
    {"id": "b", "text": "Approach B", "primary_dimension": "A", "dimension_weights": {"A": 0.9, "I": 0.1}},
    {"id": "c", "text": "Approach C", "primary_dimension": "I", "dimension_weights": {"I": 0.7, "A": 0.3}},
    {"id": "d", "text": "Approach D", "primary_dimension": "S", "dimension_weights": {"S": 0.8, "E": 0.2}},
    {"id": "e", "text": "Approach E", "primary_dimension": "E", "dimension_weights": {"E": 0.9, "S": 0.1}}
  ],
  "dimensions_evaluated": ["R", "A", "I", "S", "E"],
  "min_selections": 2,
  "max_selections": 3
}"""
        
        elif question_type == QuestionType.THIS_OR_THAT:
            return """THIS OR THAT (Binary Choice):
Create two clearly contrasting options representing different RAISEC dimensions.
- Options should be equally appealing but fundamentally different
- Should highlight clear personality/interest differences
- Use "Would you rather..." format

JSON Structure:
{
  "question_text": "Would you rather...",
  "instructions": "Choose the option that appeals to you more",
  "option_a": {
    "text": "Option A description",
    "primary_dimension": "R",
    "dimension_weights": {"R": 1.0}
  },
  "option_b": {
    "text": "Option B description", 
    "primary_dimension": "A",
    "dimension_weights": {"A": 1.0}
  },
  "dimensions_evaluated": ["R", "A"]
}"""
        
        elif question_type == QuestionType.SCALE_RATING:
            return """SCALE RATING (1-10):
Create a question asking to rate interest/enjoyment on a 1-10 scale.
- Focus on specific activities or scenarios
- Should clearly relate to the target RAISEC dimensions
- Use engaging, specific examples

JSON Structure:
{
  "question_text": "How much would you enjoy [specific activity/scenario]?",
  "instructions": "Rate from 1 (Not at all) to 10 (Extremely)",
  "scale_min": 1,
  "scale_max": 10,
  "scale_labels": {
    "1": "Not at all",
    "5": "Moderately", 
    "10": "Extremely"
  },
  "dimensions_evaluated": ["Primary dimension being measured"]
}"""
        
        elif question_type == QuestionType.PLOT_DAY:
            return """PLOT DAY (Time Allocation):
Create 8-10 activities for a hypothetical day schedule.
- Activities should span all RAISEC dimensions
- Include realistic time blocks (9-12, 12-3, 3-6, 6-9)
- Activities should be engaging and diverse
- Each activity should have clear RAISEC alignment

JSON Structure:
{
  "question_text": "Plan your ideal day by placing these activities in different time slots",
  "instructions": "Drag activities to time slots based on when you'd prefer to do them",
  "tasks": [
    {"id": "task1", "title": "Activity 1", "description": "Brief description", "primary_dimension": "R", "secondary_dimensions": ["C"]},
    {"id": "task2", "title": "Activity 2", "description": "Brief description", "primary_dimension": "A", "secondary_dimensions": ["I"]},
    {"id": "task3", "title": "Activity 3", "description": "Brief description", "primary_dimension": "I", "secondary_dimensions": []},
    {"id": "task4", "title": "Activity 4", "description": "Brief description", "primary_dimension": "S", "secondary_dimensions": ["E"]},
    {"id": "task5", "title": "Activity 5", "description": "Brief description", "primary_dimension": "E", "secondary_dimensions": ["S"]},
    {"id": "task6", "title": "Activity 6", "description": "Brief description", "primary_dimension": "C", "secondary_dimensions": ["R"]},
    {"id": "task7", "title": "Activity 7", "description": "Brief description", "primary_dimension": "R", "secondary_dimensions": []},
    {"id": "task8", "title": "Activity 8", "description": "Brief description", "primary_dimension": "A", "secondary_dimensions": []}
  ],
  "time_slots": ["9:00-12:00", "12:00-15:00", "15:00-18:00", "18:00-21:00"],
  "dimensions_evaluated": ["R", "A", "I", "S", "E", "C"]
}"""
        
        return ""
    
    @classmethod
    def _get_human_prompt(cls, question_type: QuestionType, age_group: AgeGroup) -> str:
        """Get human prompt template.
        
        Args:
            question_type: Type of question
            age_group: Target age group
            
        Returns:
            str: Human prompt template
        """
        return """Generate a {question_type} question for {age_group} age group ({age_range} years old).

REQUIREMENTS:
- Question number: {question_number}
- Primary focus dimensions: {dimensions_focus}
- Additional context: {context}
- Constraints: {constraints}

Focus primarily on the "{primary_dimension[name]}" dimension: {primary_dimension[description]}

Ensure the question:
1. Is culturally appropriate for Indian users
2. Uses age-appropriate language for {age_group}
3. Clearly measures the specified RAISEC dimensions
4. Is engaging and realistic
5. Follows the exact JSON structure specified

Generate the question now:"""


# Export the prompts class
__all__ = ["QuestionPrompts"]