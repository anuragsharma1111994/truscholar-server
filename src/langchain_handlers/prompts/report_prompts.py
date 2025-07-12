"""Report generation prompts for RAISEC assessment results.

This module provides comprehensive prompt templates for generating
detailed, personalized assessment reports.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.utils.constants import ReportType


class ReportPrompts:
    """Prompt templates for report generation."""
    
    # Base system prompt for report generation
    BASE_SYSTEM_PROMPT = """You are an expert career psychologist and assessment specialist who creates comprehensive, personalized career assessment reports. Your reports combine scientific rigor with practical guidance to help individuals understand their career preferences and make informed decisions.

RAISEC Framework:
- R (Realistic): Practical, hands-on, mechanical, outdoor-oriented personality
- A (Artistic): Creative, expressive, aesthetic, innovative personality  
- I (Investigative): Analytical, intellectual, research-oriented personality
- S (Social): People-focused, helping, teaching, interpersonal personality
- E (Enterprising): Leadership, persuasive, business-oriented personality
- C (Conventional): Organized, systematic, detail-oriented personality

Your reports must:
1. Be professional yet accessible and engaging
2. Provide scientific backing for insights and recommendations
3. Be culturally sensitive to Indian context and values
4. Include both strengths and development areas
5. Offer practical, actionable guidance
6. Maintain an encouraging and positive tone
7. Be structured logically with clear sections
8. Include specific examples and applications

Format your response as valid JSON with structured content sections."""
    
    @classmethod
    def get_template(cls, report_type: ReportType) -> ChatPromptTemplate:
        """Get appropriate prompt template for report type.
        
        Args:
            report_type: Type of report to generate
            
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        system_prompt = cls._get_system_prompt(report_type)
        human_prompt = cls._get_human_prompt(report_type)
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    @classmethod
    def _get_system_prompt(cls, report_type: ReportType) -> str:
        """Get system prompt for specific report type.
        
        Args:
            report_type: Type of report
            
        Returns:
            str: System prompt template
        """
        type_context = cls._get_type_context(report_type)
        
        return f"""{cls.BASE_SYSTEM_PROMPT}

REPORT TYPE CONTEXT:
{type_context}

WRITING GUIDELINES:
- Use warm, professional tone throughout
- Write in second person ("You have...", "Your results show...")
- Include specific examples and scenarios
- Balance technical accuracy with readability
- Provide hope and encouragement while being realistic
- Use active voice and engaging language
- Include cultural sensitivity for Indian context

STRUCTURE REQUIREMENTS:
- Each section should be self-contained yet connected
- Use clear headings and subheadings
- Provide smooth transitions between sections
- Include relevant quotes or insights where appropriate
- End each section with actionable takeaways

IMPORTANT: Your response must be valid JSON. Do not include any text before or after the JSON structure."""
    
    @classmethod
    def _get_type_context(cls, report_type: ReportType) -> str:
        """Get report type-specific context.
        
        Args:
            report_type: Type of report
            
        Returns:
            str: Type-specific context
        """
        if report_type == ReportType.COMPREHENSIVE:
            return """COMPREHENSIVE REPORT:
Create a detailed, thorough report covering all aspects of the assessment:
- In-depth personality analysis
- Detailed career recommendations
- Comprehensive development plan
- Extensive next steps guidance
- Educational pathway suggestions
- Skill development recommendations

Target Length: 2000-3000 words
Depth: Deep analysis with extensive explanations
Audience: Users who want complete understanding

JSON Structure:
{
  "report": {
    "title": "Your Comprehensive Career Assessment Report",
    "executive_summary": "High-level overview of key findings",
    "personality_profile": {
      "overview": "Detailed personality analysis",
      "dominant_traits": ["Trait 1", "Trait 2"],
      "raisec_analysis": {
        "primary_dimension": "Detailed analysis of top dimension",
        "secondary_dimension": "Analysis of second dimension", 
        "tertiary_dimension": "Analysis of third dimension",
        "dimension_interactions": "How dimensions work together"
      },
      "strengths": ["Strength 1 with explanation", "Strength 2"],
      "development_areas": ["Area 1 with guidance", "Area 2"],
      "work_style_preferences": "Preferred work environment and style"
    },
    "career_recommendations": {
      "overview": "Career recommendation strategy",
      "top_careers": [
        {
          "career": "Career name",
          "match_explanation": "Why this fits your profile",
          "pathway": "How to get there",
          "timeline": "Realistic timeline"
        }
      ],
      "career_clusters": ["Industry 1", "Industry 2"],
      "entrepreneurship_potential": "Assessment of entrepreneurial fit"
    },
    "development_plan": {
      "immediate_actions": ["Action 1", "Action 2"],
      "short_term_goals": ["6-month goals"],
      "long_term_vision": ["2-3 year goals"],
      "skill_development": {
        "technical_skills": ["Skill 1", "Skill 2"],
        "soft_skills": ["Skill 1", "Skill 2"],
        "leadership_skills": ["Skill 1", "Skill 2"]
      },
      "education_recommendations": {
        "formal_education": ["Degree/certification options"],
        "online_learning": ["Course recommendations"],
        "experiential_learning": ["Internship/project ideas"]
      }
    },
    "next_steps": {
      "week_1": ["Immediate action"],
      "month_1": ["Short-term action"],
      "month_3": ["Medium-term action"],
      "year_1": ["Long-term goal"]
    },
    "resources": {
      "books": ["Recommended reading"],
      "websites": ["Useful websites"],
      "organizations": ["Professional associations"],
      "networking": ["Networking suggestions"]
    },
    "conclusion": "Encouraging conclusion with key takeaways"
  }
}"""
        
        elif report_type == ReportType.SUMMARY:
            return """SUMMARY REPORT:
Create a concise, focused report highlighting key insights:
- Core personality findings
- Top career matches
- Essential next steps
- Key recommendations

Target Length: 800-1200 words
Depth: Focused on most important insights
Audience: Users who want quick, actionable insights

JSON Structure:
{
  "report": {
    "title": "Your Career Assessment Summary",
    "key_insights": {
      "personality_snapshot": "Concise personality overview",
      "dominant_themes": ["Theme 1", "Theme 2", "Theme 3"],
      "core_strengths": ["Strength 1", "Strength 2", "Strength 3"]
    },
    "raisec_profile": {
      "code": "RIA",
      "primary_trait": "Detailed explanation of dominant trait",
      "secondary_trait": "Explanation of second trait",
      "profile_summary": "How traits combine to create unique profile"
    },
    "career_matches": [
      {
        "career": "Top career match",
        "why_its_perfect": "Concise explanation",
        "get_started": "First step to explore this career"
      },
      {
        "career": "Second career match", 
        "why_its_perfect": "Concise explanation",
        "get_started": "First step to explore this career"
      },
      {
        "career": "Third career match",
        "why_its_perfect": "Concise explanation", 
        "get_started": "First step to explore this career"
      }
    ],
    "action_plan": {
      "priority_1": "Most important next step",
      "priority_2": "Second most important step",
      "priority_3": "Third priority",
      "quick_wins": ["Easy action 1", "Easy action 2"]
    },
    "development_focus": {
      "skills_to_build": ["Skill 1", "Skill 2"],
      "experiences_to_seek": ["Experience 1", "Experience 2"],
      "knowledge_to_gain": ["Knowledge area 1", "Knowledge area 2"]
    },
    "encouragement": "Motivational closing message"
  }
}"""
        
        elif report_type == ReportType.DETAILED:
            return """DETAILED REPORT:
Create a thorough analysis with moderate depth:
- Complete personality assessment
- Multiple career options
- Detailed development guidance
- Comprehensive action steps

Target Length: 1500-2000 words  
Depth: Thorough analysis with good explanations
Audience: Users who want comprehensive guidance without overwhelming detail

JSON Structure:
{
  "report": {
    "title": "Your Detailed Career Assessment Report",
    "introduction": "Personalized introduction to the report",
    "assessment_overview": {
      "completion_summary": "Assessment completion details",
      "reliability_indicators": "Confidence in results",
      "interpretation_notes": "How to interpret the results"
    },
    "personality_analysis": {
      "raisec_breakdown": {
        "realistic": {"score": 85, "interpretation": "What this means"},
        "investigative": {"score": 78, "interpretation": "What this means"},
        "artistic": {"score": 65, "interpretation": "What this means"},
        "social": {"score": 45, "interpretation": "What this means"},
        "enterprising": {"score": 35, "interpretation": "What this means"},
        "conventional": {"score": 25, "interpretation": "What this means"}
      },
      "personality_type": "Overall personality description",
      "core_motivations": ["Motivation 1", "Motivation 2"],
      "work_values": ["Value 1", "Value 2"],
      "preferred_environments": ["Environment 1", "Environment 2"]
    },
    "career_exploration": {
      "highly_recommended": [
        {
          "career": "Career name",
          "match_score": 92,
          "description": "Career description",
          "why_good_fit": "Fit explanation",
          "pathway": "How to pursue",
          "outlook": "Job market outlook"
        }
      ],
      "worth_exploring": [
        {
          "career": "Career name",
          "match_score": 78,
          "description": "Career description", 
          "considerations": "Things to consider"
        }
      ],
      "industry_sectors": ["Sector 1", "Sector 2", "Sector 3"]
    },
    "development_strategy": {
      "strengths_to_leverage": [
        {
          "strength": "Strength name",
          "how_to_use": "Application guidance",
          "development_tips": "How to enhance further"
        }
      ],
      "areas_to_develop": [
        {
          "area": "Development area",
          "importance": "Why this matters",
          "development_approach": "How to improve"
        }
      ],
      "learning_style": "Preferred learning approach",
      "development_timeline": "Suggested development sequence"
    },
    "implementation_guide": {
      "phase_1": {
        "title": "Exploration Phase (Months 1-3)",
        "objectives": ["Objective 1", "Objective 2"],
        "actions": ["Action 1", "Action 2"],
        "milestones": ["Milestone 1", "Milestone 2"]
      },
      "phase_2": {
        "title": "Preparation Phase (Months 4-9)", 
        "objectives": ["Objective 1", "Objective 2"],
        "actions": ["Action 1", "Action 2"],
        "milestones": ["Milestone 1", "Milestone 2"]
      },
      "phase_3": {
        "title": "Action Phase (Months 10-12)",
        "objectives": ["Objective 1", "Objective 2"],
        "actions": ["Action 1", "Action 2"],
        "milestones": ["Milestone 1", "Milestone 2"]
      }
    },
    "success_factors": {
      "critical_success_factors": ["Factor 1", "Factor 2"],
      "potential_obstacles": ["Obstacle 1", "Obstacle 2"],
      "support_systems": ["Support 1", "Support 2"]
    },
    "closing_thoughts": "Personalized encouragement and final guidance"
  }
}"""
        
        return ""
    
    @classmethod
    def _get_human_prompt(cls, report_type: ReportType) -> str:
        """Get human prompt template.
        
        Args:
            report_type: Type of report
            
        Returns:
            str: Human prompt template
        """
        return """Generate a {report_type} career assessment report for this user:

USER PROFILE:
- Name: {user_name}
- Age: {user_age} years ({age_group})
- Career Stage: {career_stage}
- Location: {user_location}
- Education: {education_level}
- Experience: {experience_level}

ASSESSMENT RESULTS:
- RAISEC Code: {raisec_code}
- Dominant Code: {dominant_code}
- Total Score: {total_assessment_score}

DIMENSION ANALYSIS:
- Primary: {primary_dimension[name]} (Score: {primary_dimension[score]}) - {primary_dimension[description]}
- Secondary: {secondary_dimension[name]} (Score: {secondary_dimension[score]}) - {secondary_dimension[description]}
- Tertiary: {tertiary_dimension[name]} (Score: {tertiary_dimension[score]}) - {tertiary_dimension[description]}

ALL DIMENSIONS:
{all_dimensions}

SCORE PATTERN ANALYSIS:
- Profile Type: {score_analysis[profile_type]}
- Score Range: {score_analysis[score_range]}
- Mean Score: {score_analysis[mean_score]}
- {score_analysis[profile_description]}

CAREER RECOMMENDATIONS:
{career_recommendations}

COMPLETION METRICS:
- Questions Answered: {completion_metrics[questions_answered]}/{completion_metrics[total_questions]}
- Completion Rate: {completion_metrics[completion_percentage]}%
- Time Spent: {completion_metrics[time_spent_minutes]} minutes
- Confidence Score: {completion_metrics[confidence_score]}%

ADDITIONAL INSIGHTS:
{additional_insights}

REQUIREMENTS:
1. Write in a warm, professional, and encouraging tone
2. Personalize the content using the user's name and specific profile
3. Provide scientific backing for insights while keeping it accessible
4. Include specific, actionable recommendations
5. Consider Indian cultural context and job market
6. Structure the report logically with smooth transitions
7. End on an encouraging and motivational note

Generate the {report_type} report following the JSON structure:"""


# Export the prompts class
__all__ = ["ReportPrompts"]