"""Enhanced career recommendation prompts for comprehensive RAISEC analysis.

This module provides sophisticated prompt templates for generating
personalized career recommendations, comparisons, narratives, and insights
based on RAISEC profiles with advanced analysis capabilities.
"""

from typing import Dict, Any, List, Optional, Union
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    PromptTemplate
)

from src.utils.constants import RecommendationType


class CareerPrompts:
    """Prompt templates for career recommendation generation."""
    
    # Base system prompt for career recommendations
    BASE_SYSTEM_PROMPT = """You are an expert career counselor and psychologist specializing in RAISEC (Holland Code) career assessments. You provide personalized, actionable career recommendations based on scientific personality-career fit research.

RAISEC Framework:
- R (Realistic): Practical, hands-on work with tools, machines, animals, or outdoors
- A (Artistic): Creative, expressive work involving art, music, writing, or design
- I (Investigative): Analytical, research-oriented work involving problem-solving and data
- S (Social): People-focused work involving helping, teaching, or counseling others
- E (Enterprising): Leadership, business, and influential work with people and projects
- C (Conventional): Organized, systematic work with data, details, and procedures

Your recommendations must:
1. Be based on solid RAISEC theory and research
2. Consider the Indian job market and cultural context
3. Account for current and emerging career trends
4. Be realistic and achievable for the user's profile
5. Include both traditional and innovative career paths
6. Provide actionable next steps and development suggestions
7. Consider education, skills, and experience requirements

Format your response as valid JSON following the exact structure specified."""
    
    @classmethod
    def get_template(cls, recommendation_type: RecommendationType) -> ChatPromptTemplate:
        """Get appropriate prompt template for recommendation type.
        
        Args:
            recommendation_type: Type of recommendations to generate
            
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        system_prompt = cls._get_system_prompt(recommendation_type)
        human_prompt = cls._get_human_prompt(recommendation_type)
        
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ])
    
    @classmethod
    def _get_system_prompt(cls, recommendation_type: RecommendationType) -> str:
        """Get system prompt for specific recommendation type.
        
        Args:
            recommendation_type: Type of recommendations
            
        Returns:
            str: System prompt template
        """
        type_context = cls._get_type_context(recommendation_type)
        
        return f"""{cls.BASE_SYSTEM_PROMPT}

RECOMMENDATION TYPE CONTEXT:
{type_context}

INDIAN CONTEXT CONSIDERATIONS:
- Include both public and private sector opportunities
- Consider family expectations and social factors
- Account for regional job market variations
- Include startup and entrepreneurship opportunities
- Consider work-life balance preferences
- Account for diverse educational backgrounds (engineering, commerce, arts, science)

CAREER PROGRESSION STAGES:
- Entry Level: 0-2 years experience
- Early Career: 2-5 years experience  
- Mid Career: 5-10 years experience
- Senior Career: 10+ years experience

IMPORTANT: Your response must be valid JSON. Do not include any text before or after the JSON structure."""
    
    @classmethod
    def _get_type_context(cls, recommendation_type: RecommendationType) -> str:
        """Get recommendation type-specific context.
        
        Args:
            recommendation_type: Type of recommendations
            
        Returns:
            str: Type-specific context
        """
        if recommendation_type == RecommendationType.TRADITIONAL:
            return """TRADITIONAL CAREER RECOMMENDATIONS:
Focus on well-established, mainstream career paths with:
- Clear career progression ladders
- Stable employment opportunities
- Recognized professional qualifications
- Traditional industry sectors
- Conventional work environments
- Established educational pathways

Include careers in:
- Government services and public sector
- Large corporations and MNCs
- Professional services (law, medicine, accounting)
- Traditional engineering and IT roles
- Banking and financial services
- Education and academia

JSON Structure:
{
  "recommendations": [
    {
      "title": "Career Title",
      "category": "Industry/Field",
      "description": "Detailed career description",
      "raisec_match": ["R", "I", "C"],
      "match_score": 85,
      "education_requirements": ["Degree/qualification needed"],
      "skills_required": ["Key skills"],
      "salary_range": {"min": 300000, "max": 1500000, "currency": "INR"},
      "growth_prospects": "Growth description",
      "work_environment": "Work environment description",
      "daily_activities": ["Activity 1", "Activity 2"],
      "career_path": ["Entry role", "Mid-level", "Senior level"],
      "companies_hiring": ["Company examples"],
      "next_steps": ["Action 1", "Action 2"]
    }
  ],
  "summary": "Overall recommendation summary",
  "key_strengths": ["Strength 1", "Strength 2"],
  "development_areas": ["Area 1", "Area 2"]
}"""
        
        elif recommendation_type == RecommendationType.INNOVATIVE:
            return """INNOVATIVE CAREER RECOMMENDATIONS:
Focus on emerging, non-traditional career paths with:
- New-age industries and technologies
- Creative and entrepreneurial opportunities
- Digital and tech-enabled roles
- Flexible work arrangements
- Cross-disciplinary career paths
- Future-focused skill requirements

Include careers in:
- Technology and digital innovation
- Creative industries and content creation
- Sustainability and green economy
- Social entrepreneurship
- Remote and freelance opportunities
- Emerging fields (AI, blockchain, IoT, etc.)

JSON Structure:
{
  "recommendations": [
    {
      "title": "Career Title", 
      "category": "Emerging Field",
      "description": "Detailed career description",
      "raisec_match": ["A", "E", "I"],
      "match_score": 82,
      "education_requirements": ["Skills/certifications needed"],
      "skills_required": ["Key skills"],
      "salary_range": {"min": 400000, "max": 2000000, "currency": "INR"},
      "growth_prospects": "High growth potential description",
      "work_environment": "Modern/flexible work environment",
      "daily_activities": ["Innovation-focused activity 1", "Activity 2"],
      "career_path": ["Entry role", "Specialist", "Expert/Leader"],
      "companies_hiring": ["Startups", "Tech companies"],
      "innovation_factor": "What makes this career innovative",
      "future_trends": "Relevant future trends",
      "next_steps": ["Skill development", "Portfolio building"]
    }
  ],
  "summary": "Innovation-focused recommendation summary",
  "key_strengths": ["Innovation strength 1", "Strength 2"],
  "development_areas": ["Future skill 1", "Area 2"],
  "market_trends": ["Trend 1", "Trend 2"]
}"""
        
        elif recommendation_type == RecommendationType.HYBRID:
            return """HYBRID CAREER RECOMMENDATIONS:
Combine traditional stability with innovative opportunities:
- Traditional roles with modern applications
- Cross-industry career transitions
- Portfolio careers combining multiple interests
- Traditional companies adopting new technologies
- Established fields with innovative approaches
- Blend of employment and entrepreneurship

Include careers that:
- Bridge traditional and emerging sectors
- Combine multiple RAISEC dimensions
- Offer both stability and growth
- Allow for career pivoting and evolution
- Balance conventional and creative elements

JSON Structure:
{
  "recommendations": [
    {
      "title": "Hybrid Career Title",
      "category": "Hybrid Field",
      "description": "Career combining traditional and innovative elements",
      "raisec_match": ["Primary dimensions"],
      "match_score": 88,
      "traditional_aspects": ["Stable element 1", "Element 2"],
      "innovative_aspects": ["Innovation 1", "Innovation 2"],
      "education_requirements": ["Blend of qualifications"],
      "skills_required": ["Traditional skills", "Modern skills"],
      "salary_range": {"min": 350000, "max": 1800000, "currency": "INR"},
      "growth_prospects": "Balanced growth description",
      "work_environment": "Hybrid work environment",
      "daily_activities": ["Traditional activity", "Innovative activity"],
      "career_path": ["Traditional entry", "Hybrid mid-level", "Innovation leader"],
      "companies_hiring": ["Traditional firms", "Modern companies"],
      "transition_paths": ["How to transition from traditional role"],
      "next_steps": ["Bridge building action 1", "Action 2"]
    }
  ],
  "summary": "Hybrid approach recommendation summary",
  "key_strengths": ["Versatility", "Adaptability"],
  "development_areas": ["Bridge skill 1", "Area 2"],
  "transition_strategies": ["Strategy 1", "Strategy 2"]
}"""
        
        return ""
    
    @classmethod
    def _get_human_prompt(cls, recommendation_type: RecommendationType) -> str:
        """Get human prompt template.
        
        Args:
            recommendation_type: Type of recommendations
            
        Returns:
            str: Human prompt template
        """
        return """Generate {recommendation_type} career recommendations for this user profile:

USER RAISEC PROFILE:
- RAISEC Code: {raisec_code}
- Top Dimensions: {top_three_dimensions}
- All Dimension Scores: {all_dimensions}
- Score Spread: {score_spread} (indicates preference clarity)

USER DEMOGRAPHICS:
- Age: {user_age} years ({age_group})
- Career Stage: {career_stage}
- Location: {user_location}
- Education Level: {education_level}
- Experience Level: {experience_level}

USER INTERESTS:
{interests}

CONSTRAINTS & PREFERENCES:
{constraints}

ANALYSIS REQUIRED:
1. Analyze the RAISEC profile for career fit patterns
2. Consider the user's demographic context and career stage
3. Evaluate current market trends and opportunities in India
4. Provide 3-5 highly relevant career recommendations
5. Include actionable next steps for each recommendation

Focus on the user's strongest dimensions: {top_three_dimensions}

Generate comprehensive {recommendation_type} career recommendations following the JSON structure:"""
    
    # Advanced Prompt Templates
    
    @classmethod
    def get_comprehensive_recommendation_prompt(
        cls,
        user_raisec_code: str,
        user_raisec_scores: Dict[str, float],
        user_preferences: Dict[str, Any],
        recommendation_count: int = 20
    ) -> PromptTemplate:
        """Get comprehensive career recommendation prompt."""
        return PromptTemplate(
            input_variables=["user_profile", "career_context", "market_data"],
            template=f"""You are an expert career counselor specializing in comprehensive RAISEC-based career guidance.

TASK: Generate {recommendation_count} highly personalized career recommendations.

USER RAISEC PROFILE:
- RAISEC Code: {user_raisec_code}
- Dimension Scores: {user_raisec_scores}
- Profile Strength: {cls._calculate_profile_strength(user_raisec_scores)}

USER CONTEXT:
{{user_profile}}

MARKET CONTEXT:
{{career_context}}

REQUIREMENTS:
1. Provide exactly {recommendation_count} career recommendations
2. Rank by compatibility score (0-100)
3. Include both traditional and emerging careers
4. Consider Indian job market dynamics
5. Account for future career trends
6. Provide actionable development pathways

OUTPUT FORMAT:
{{
    "recommendations": [
        {{
            "rank": 1,
            "career_title": "Software Engineering Manager",
            "category": "Technology Leadership",
            "compatibility_score": 94,
            "raisec_alignment": {{
                "primary_match": ["I", "E"],
                "secondary_match": ["C"],
                "alignment_explanation": "Strong analytical and leadership alignment"
            }},
            "market_demand": "High",
            "salary_potential": {{
                "entry": 800000,
                "mid": 1500000,
                "senior": 3000000,
                "currency": "INR"
            }},
            "growth_trajectory": "Excellent",
            "skill_requirements": ["Technical leadership", "Team management"],
            "education_pathway": ["B.Tech/B.E.", "MBA (preferred)"],
            "development_plan": [
                "Gain 3-5 years software development experience",
                "Develop leadership and communication skills",
                "Pursue technical management certifications"
            ],
            "pros": ["High growth", "Leadership opportunities"],
            "cons": ["High pressure", "Technical complexity"],
            "similar_careers": ["Product Manager", "Technical Architect"],
            "companies": ["TCS", "Infosys", "Amazon", "Google"],
            "confidence_level": 0.92
        }}
    ],
    "analysis_summary": "Comprehensive analysis based on dominant I-E profile...",
    "career_clusters": ["Technology", "Business", "Research"],
    "development_priorities": ["Leadership skills", "Technical depth"],
    "market_insights": "Technology sector showing 15% growth...",
    "next_steps": ["Assess current skills", "Create development plan"]
}}"""
        )
    
    @classmethod
    def get_focused_recommendation_prompt(
        cls,
        user_raisec_code: str,
        focus_dimensions: List[str],
        user_preferences: Dict[str, Any],
        recommendation_count: int = 15
    ) -> PromptTemplate:
        """Get focused career recommendation prompt based on top dimensions."""
        return PromptTemplate(
            input_variables=["user_context", "dimension_analysis"],
            template=f"""You are a specialized career counselor focusing on targeted career guidance.

TASK: Generate {recommendation_count} highly focused career recommendations based on dominant RAISEC dimensions.

FOCUS DIMENSIONS: {focus_dimensions}
USER RAISEC CODE: {user_raisec_code}

ANALYSIS APPROACH:
1. Prioritize careers strongly aligned with {focus_dimensions}
2. Look for careers that leverage these dimensional strengths
3. Consider complementary secondary dimensions
4. Focus on career paths that maximize user's natural strengths

USER CONTEXT:
{{user_context}}

DIMENSIONAL ANALYSIS:
{{dimension_analysis}}

OUTPUT: Provide {recommendation_count} laser-focused career recommendations in JSON format with deep analysis of dimensional alignment."""
        )
    
    @classmethod
    def get_exploratory_recommendation_prompt(
        cls,
        user_raisec_code: str,
        user_raisec_scores: Dict[str, float],
        include_emerging: bool = True,
        include_unconventional: bool = True,
        recommendation_count: int = 25
    ) -> PromptTemplate:
        """Get exploratory career recommendation prompt including emerging fields."""
        exploration_scope = []
        if include_emerging:
            exploration_scope.append("emerging and future-focused careers")
        if include_unconventional:
            exploration_scope.append("unconventional and creative career paths")
        
        return PromptTemplate(
            input_variables=["user_profile", "trend_analysis"],
            template=f"""You are an innovative career counselor specializing in exploratory career guidance.

TASK: Generate {recommendation_count} exploratory career recommendations including {', '.join(exploration_scope)}.

USER PROFILE:
- RAISEC Code: {user_raisec_code}
- Scores: {user_raisec_scores}

EXPLORATION FOCUS:
- Include cutting-edge careers in emerging industries
- Consider unconventional career combinations
- Explore interdisciplinary opportunities
- Look at future-oriented roles (2025-2035)
- Include entrepreneurial and freelance opportunities
- Consider global remote opportunities

USER CONTEXT:
{{user_profile}}

TREND ANALYSIS:
{{trend_analysis}}

SPECIAL FOCUS AREAS:
1. AI and Machine Learning careers
2. Sustainability and Green Economy
3. Digital Content and Creator Economy
4. Health-tech and Bio-technology
5. Space and Aerospace industries
6. Virtual and Augmented Reality
7. Blockchain and Web3
8. Social Impact and NGO sector

OUTPUT: {recommendation_count} innovative career recommendations with future-readiness analysis."""
        )
    
    @classmethod
    def get_career_comparison_prompt(
        cls,
        careers_data: List[Dict[str, Any]],
        user_raisec_code: str,
        user_preferences: Dict[str, Any],
        comparison_aspects: List[str]
    ) -> PromptTemplate:
        """Get career comparison analysis prompt."""
        return PromptTemplate(
            input_variables=["comparison_context"],
            template=f"""You are an expert career analyst conducting detailed career comparisons.

TASK: Compare the following careers for the user's profile and provide comprehensive analysis.

USER PROFILE:
- RAISEC Code: {user_raisec_code}
- Preferences: {user_preferences}

CAREERS TO COMPARE:
{[career.get('title', 'Unknown') for career in careers_data]}

COMPARISON ASPECTS:
{comparison_aspects}

ANALYSIS FRAMEWORK:
1. RAISEC Compatibility Analysis
2. Career Progression Potential
3. Salary and Benefits Comparison
4. Work-Life Balance Assessment
5. Market Demand and Stability
6. Skill Development Opportunities
7. Geographic Flexibility
8. Industry Growth Prospects

COMPARISON CONTEXT:
{{comparison_context}}

OUTPUT FORMAT:
{{
    "comparison_matrix": {{
        "aspect_scores": {{
            "career_1": {{"raisec_fit": 85, "growth": 90, "salary": 80}},
            "career_2": {{"raisec_fit": 78, "growth": 85, "salary": 85}}
        }},
        "detailed_analysis": {{
            "raisec_alignment": "Detailed comparison of RAISEC fit...",
            "growth_potential": "Growth comparison analysis...",
            "compensation": "Salary and benefits analysis..."
        }}
    }},
    "recommendations": {{
        "best_overall_fit": "Career name and reasoning",
        "best_for_growth": "Career name and reasoning",
        "best_for_salary": "Career name and reasoning"
    }},
    "decision_factors": ["Factor 1", "Factor 2"],
    "next_steps": "Recommended actions based on comparison"
}}"""
        )
    
    @classmethod
    def get_career_narrative_prompt(
        cls,
        user_raisec_code: str,
        user_raisec_scores: Dict[str, float],
        recommended_careers: List[Dict[str, Any]],
        user_preferences: Dict[str, Any]
    ) -> PromptTemplate:
        """Get personalized career narrative prompt."""
        return PromptTemplate(
            input_variables=["user_story", "career_context"],
            template=f"""You are a master storyteller and career counselor creating personalized career narratives.

TASK: Write an engaging, personalized career narrative that connects the user's RAISEC profile to their recommended careers.

USER RAISEC PROFILE:
- Code: {user_raisec_code}
- Scores: {user_raisec_scores}
- Top Careers: {[career.get('title', 'Unknown') for career in recommended_careers[:5]]}

NARRATIVE ELEMENTS:
1. Start with the user's unique RAISEC strengths
2. Paint a picture of their ideal work environment
3. Connect their personality to career opportunities
4. Address potential concerns or challenges
5. Inspire confidence and motivation
6. Provide a vision of career success

USER STORY:
{{user_story}}

CAREER CONTEXT:
{{career_context}}

TONE: Professional yet personal, encouraging, and inspiring
LENGTH: 800-1200 words
STRUCTURE:
- Opening: Acknowledge their unique strengths
- Body: Career journey possibilities
- Conclusion: Confident next steps

OUTPUT: A compelling narrative that makes the user excited about their career future."""
        )
    
    @classmethod
    def get_skill_development_prompt(
        cls,
        target_career: Dict[str, Any],
        user_raisec_code: str,
        current_skills: List[str],
        timeline: str,
        user_preferences: Dict[str, Any]
    ) -> PromptTemplate:
        """Get skill development plan prompt."""
        return PromptTemplate(
            input_variables=["skill_context", "career_requirements"],
            template=f"""You are a learning and development specialist creating personalized skill development plans.

TASK: Create a comprehensive skill development plan for transitioning to the target career.

TARGET CAREER: {target_career.get('title', 'Unknown')}
USER RAISEC CODE: {user_raisec_code}
CURRENT SKILLS: {current_skills}
DEVELOPMENT TIMELINE: {timeline}
USER PREFERENCES: {user_preferences}

SKILL ANALYSIS FRAMEWORK:
1. Gap Analysis: Current vs Required Skills
2. Transferable Skills Identification
3. Priority Skill Ranking
4. Learning Path Design
5. Resource Recommendations
6. Progress Milestones

SKILL CONTEXT:
{{skill_context}}

CAREER REQUIREMENTS:
{{career_requirements}}

OUTPUT FORMAT:
{{
    "skill_gap_analysis": {{
        "current_strengths": ["Skill 1", "Skill 2"],
        "skill_gaps": ["Gap 1", "Gap 2"],
        "transferable_skills": ["Transferable 1", "Transferable 2"]
    }},
    "development_roadmap": {{
        "phase_1": {{
            "duration": "0-3 months",
            "focus_skills": ["Foundation skill 1", "Foundation skill 2"],
            "learning_methods": ["Online courses", "Practice projects"],
            "resources": ["Coursera link", "Book recommendation"],
            "milestones": ["Complete certification", "Build portfolio"]
        }},
        "phase_2": {{
            "duration": "3-6 months",
            "focus_skills": ["Intermediate skill 1", "Intermediate skill 2"],
            "learning_methods": ["Advanced courses", "Real projects"],
            "resources": ["Advanced course link", "Community"],
            "milestones": ["Industry project", "Network building"]
        }}
    }},
    "success_metrics": ["Metric 1", "Metric 2"],
    "alternative_paths": ["Path 1", "Path 2"],
    "estimated_timeline": "{timeline}",
    "confidence_assessment": "High/Medium/Low with reasoning"
}}"""
        )
    
    @classmethod
    def get_market_analysis_prompt(
        cls,
        career_field: str,
        location: str,
        timeframe: str
    ) -> PromptTemplate:
        """Get job market analysis prompt."""
        return PromptTemplate(
            input_variables=["market_data", "trend_analysis"],
            template=f"""You are a labor market analyst providing comprehensive job market insights.

TASK: Analyze the job market for {career_field} in {location} with {timeframe} outlook.

ANALYSIS FRAMEWORK:
1. Current Demand and Supply Dynamics
2. Salary Trends and Compensation Analysis
3. Growth Projections and Opportunities
4. Geographic Hotspots and Remote Work
5. Industry Disruptions and Innovations
6. Entry Requirements and Competition
7. Future Skills and Qualifications

MARKET DATA:
{{market_data}}

TREND ANALYSIS:
{{trend_analysis}}

OUTPUT FORMAT:
{{
    "market_overview": {{
        "current_state": "Description of current market conditions",
        "demand_level": "High/Medium/Low",
        "competition": "Assessment of job competition",
        "growth_rate": "X% annual growth"
    }},
    "salary_analysis": {{
        "entry_level": {{"min": 400000, "max": 600000, "currency": "INR"}},
        "mid_level": {{"min": 800000, "max": 1200000, "currency": "INR"}},
        "senior_level": {{"min": 1500000, "max": 2500000, "currency": "INR"}},
        "trends": "Salary trend analysis"
    }},
    "geographic_insights": {{
        "top_cities": ["Bangalore", "Mumbai", "Delhi"],
        "emerging_hubs": ["Pune", "Hyderabad"],
        "remote_opportunities": "High/Medium/Low"
    }},
    "future_outlook": {{
        "growth_forecast": "5-year growth projection",
        "emerging_trends": ["Trend 1", "Trend 2"],
        "risks": ["Risk 1", "Risk 2"],
        "opportunities": ["Opportunity 1", "Opportunity 2"]
    }},
    "recommendations": "Strategic recommendations for career entry"
}}"""
        )
    
    @classmethod
    def get_career_insights_prompt(
        cls,
        user_raisec_code: str,
        recommendations: List[Dict[str, Any]],
        user_preferences: Dict[str, Any]
    ) -> PromptTemplate:
        """Get AI-powered career insights prompt."""
        return PromptTemplate(
            input_variables=["insight_context"],
            template=f"""You are an AI career insight specialist providing deep analytical insights.

TASK: Generate comprehensive career insights based on RAISEC analysis and recommendations.

USER PROFILE:
- RAISEC Code: {user_raisec_code}
- Recommended Careers: {len(recommendations)} options analyzed
- User Preferences: {user_preferences}

INSIGHT CATEGORIES:
1. Personality-Career Alignment Patterns
2. Hidden Strengths and Opportunities
3. Potential Career Challenges
4. Cross-Industry Opportunities
5. Entrepreneurial Potential
6. Long-term Career Evolution

INSIGHT CONTEXT:
{{insight_context}}

OUTPUT FORMAT:
{{
    "key_insights": [
        {{
            "category": "Personality Alignment",
            "insight": "Your strong Investigative-Enterprising profile...",
            "implication": "This suggests excellent potential for...",
            "action_items": ["Action 1", "Action 2"]
        }}
    ],
    "hidden_opportunities": [
        "Opportunity 1 with explanation",
        "Opportunity 2 with reasoning"
    ],
    "success_predictors": [
        "Factor 1: Strong analytical skills",
        "Factor 2: Leadership potential"
    ],
    "challenge_areas": [
        {{
            "challenge": "Detail orientation vs big picture thinking",
            "mitigation": "Strategies to balance both aspects"
        }}
    ],
    "unique_advantages": "What makes this profile special",
    "career_evolution_path": "How career might evolve over 10-15 years"
}}"""
        )
    
    # Utility Methods
    
    @classmethod
    def _calculate_profile_strength(cls, scores: Dict[str, float]) -> str:
        """Calculate RAISEC profile strength."""
        if not scores:
            return "Unclear"
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        spread = max_score - min_score
        
        if spread > 30:
            return "Very Clear"
        elif spread > 20:
            return "Clear"
        elif spread > 10:
            return "Moderate"
        else:
            return "Unclear"
    
    @classmethod
    def get_success_factors_prompt(
        cls,
        recommendations: List[Dict[str, Any]],
        user_raisec_code: str,
        user_preferences: Dict[str, Any]
    ) -> PromptTemplate:
        """Get success factors analysis prompt."""
        return PromptTemplate(
            input_variables=["success_context"],
            template=f"""Analyze success factors for the recommended careers based on the user's RAISEC profile.

USER RAISEC CODE: {user_raisec_code}
RECOMMENDED CAREERS: {len(recommendations)} options

Identify key success factors, potential challenges, and development strategies.

SUCCESS CONTEXT:
{{success_context}}

OUTPUT: List of success factors with specific strategies for each."""
        )
    
    @classmethod
    def get_development_priorities_prompt(
        cls,
        recommendations: List[Dict[str, Any]],
        user_raisec_code: str,
        user_preferences: Dict[str, Any]
    ) -> PromptTemplate:
        """Get development priorities prompt."""
        return PromptTemplate(
            input_variables=["development_context"],
            template=f"""Identify development priorities for career success based on recommended careers.

USER PROFILE: {user_raisec_code}
CAREER OPTIONS: {len(recommendations)} recommendations

Provide prioritized development areas with specific action plans.

DEVELOPMENT CONTEXT:
{{development_context}}

OUTPUT: Prioritized development plan with timelines and resources."""
        )


# Export the enhanced prompts class
__all__ = ["CareerPrompts"]