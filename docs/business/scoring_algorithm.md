# RAISEC Assessment Scoring Algorithm

## Overview

The TruScholar RAISEC (Holland Code) assessment uses a sophisticated multi-factor scoring algorithm to evaluate career interests and personality traits across six dimensions. This document describes the comprehensive scoring methodology implemented in the `ScoringService`.

## Table of Contents

1. [RAISEC Framework](#raisec-framework)
2. [Scoring Architecture](#scoring-architecture)
3. [Individual Answer Scoring](#individual-answer-scoring)
4. [Test-Level Aggregation](#test-level-aggregation)
5. [Consistency Analysis](#consistency-analysis)
6. [Confidence Metrics](#confidence-metrics)
7. [Time-Based Adjustments](#time-based-adjustments)
8. [Profile Generation](#profile-generation)
9. [Analytics and Insights](#analytics-and-insights)
10. [Validation and Quality Assurance](#validation-and-quality-assurance)

## RAISEC Framework

### The Six Dimensions

- **R - Realistic**: Practical, hands-on, mechanical interests
- **I - Investigative**: Analytical, scientific, research-oriented interests  
- **A - Artistic**: Creative, expressive, aesthetic interests
- **S - Social**: Helping, teaching, interpersonal interests
- **E - Enterprising**: Leadership, persuasive, business interests
- **C - Conventional**: Organized, detail-oriented, administrative interests

### Holland Code Generation

The primary result is a 3-letter Holland Code (e.g., "RIA", "SAE") representing the top three dimensional scores, ordered by strength.

## Scoring Architecture

### Multi-Layer Scoring System

```
Individual Answers → Question-Level Scores → Dimensional Scores → Test Profile
```

### Key Components

1. **Answer Processor**: Converts raw responses to dimensional contributions
2. **Consistency Analyzer**: Evaluates response patterns and reliability
3. **Confidence Calculator**: Assesses certainty in results
4. **Profile Generator**: Creates final RAISEC profile and recommendations

## Individual Answer Scoring

### Question Type Handling

#### Multiple Choice Questions (MCQ)
```python
# Each option maps to specific dimensions with weighted scores
option_A: {"R": 3.0, "I": 1.0}  # Strong realistic, weak investigative
option_B: {"A": 2.5, "S": 1.5}  # Moderate artistic, weak social
```

#### Statement Rating Sets
```python
# Likert scale conversion (1-5 rating to 0-10 points)
likert_map = {1: 0, 2: 2.5, 3: 5, 4: 7.5, 5: 10}
# Each statement contributes to specific dimensions
statement_1: {"R": 1.0}  # Full weight to Realistic
```

#### Scenario Multi-Select
```python
# Multiple selections allowed, each contributes proportionally
selected_options = ["A", "C", "D"]
# Scoring distributes across selected options
```

#### This-or-That Questions
```python
# Binary choice with strong dimensional weighting
option_A: {"E": 3.0}  # Strong enterprising
option_B: {"C": 3.0}  # Strong conventional
```

#### Scale Rating Questions
```python
# 1-10 scale normalized to dimensional strength
normalized_score = (rating / 10.0) * max_points
```

#### Plot-Your-Day Activities
```python
# Time slot placement with differential weighting
time_slots = {
    "9:00-12:00": 1.2,   # Morning preference (higher weight)
    "12:00-15:00": 1.0,  # Standard weight
    "15:00-18:00": 1.1,  # Afternoon preference
    "18:00-21:00": 0.9,  # Evening preference (lower weight)
    "not_interested": 0.0 # No contribution
}
```

### Question Type Weights

Different question types have varying reliability and are weighted accordingly:

```python
question_weights = {
    "MCQ": 1.0,                    # Standard weight
    "STATEMENT_SET": 0.8,          # Slightly lower (social desirability bias)
    "SCENARIO_MCQ": 1.0,           # Standard weight
    "SCENARIO_MULTI_SELECT": 0.6,  # Lower (multiple selections dilute signal)
    "THIS_OR_THAT": 1.2,          # Higher (forced choice reduces bias)
    "SCALE_RATING": 0.9,           # Slightly lower (scale interpretation varies)
    "PLOT_DAY": 1.5               # Highest (behavioral simulation)
}
```

## Test-Level Aggregation

### Dimensional Score Calculation

For each RAISEC dimension:

1. **Raw Score Aggregation**
   ```python
   raw_score = sum(answer_contribution * question_weight for each answer)
   ```

2. **Normalization**
   ```python
   normalized_score = (raw_score / max_possible_score) * 100
   ```

3. **Confidence Weighting**
   ```python
   final_score = normalized_score * (confidence_level / 100)
   ```

### Overall Test Score

```python
total_score = weighted_average(all_dimensional_scores)
```

## Consistency Analysis

### Temporal Consistency
Analyzes response timing patterns:
- **Speed Consistency**: Variation in response times
- **Pattern Detection**: Speeding up, slowing down, or consistent pace
- **Outlier Detection**: Unusually fast or slow responses

### Response Consistency
Evaluates answer patterns:
- **Dimensional Coherence**: Consistency within dimensional preferences
- **Question Type Consistency**: Similar responses across question formats
- **Revision Behavior**: Frequency and pattern of answer changes

### Consistency Scoring
```python
consistency_factors = {
    "time_consistency": 0.3,      # 30% weight
    "response_consistency": 0.4,   # 40% weight
    "dimensional_consistency": 0.3  # 30% weight
}

overall_consistency = weighted_average(consistency_factors)
```

## Confidence Metrics

### Answer-Level Confidence

Factors affecting confidence in individual answers:
- **Response Time**: Too fast (rushing) or too slow (overthinking)
- **Revision Count**: Multiple changes indicate uncertainty
- **Question Difficulty**: Complex questions reduce confidence
- **Hesitation Score**: Based on interaction patterns

```python
base_confidence = 100.0
if revision_count > 2:
    base_confidence -= 10 * (revision_count - 2)
if hesitation_score > 50:
    base_confidence -= (hesitation_score - 50) * 0.5
confidence = max(base_confidence, 50.0)  # Minimum 50%
```

### Test-Level Confidence

Aggregated confidence considering:
- Individual answer confidence levels
- Overall consistency scores
- Completion rate
- Response quality indicators

## Time-Based Adjustments

### Expected Time Calculation

Age-specific time expectations:
```python
expected_times = {
    "13-17": {"mcq": 45, "statement_set": 60, "plot_day": 120},
    "18-25": {"mcq": 35, "statement_set": 45, "plot_day": 100}, 
    "26-35": {"mcq": 30, "statement_set": 40, "plot_day": 90}
}
```

### Time Adjustment Factors

```python
time_factors = {
    "very_fast": 0.85,    # < 50% expected time (potential rushing)
    "fast": 0.92,         # 50-75% expected time
    "normal": 1.0,        # 75-125% expected time (optimal)
    "slow": 1.05,         # 125-150% expected time (thoughtful)
    "very_slow": 0.95     # > 150% expected time (potential overthinking)
}
```

## Profile Generation

### RAISEC Code Determination

1. **Rank Dimensions**: Sort by normalized scores
2. **Select Top 3**: Create 3-letter code from highest scores
3. **Apply Minimum Threshold**: Ensure meaningful differences between dimensions
4. **Generate Profile Description**: Create personalized narrative

### Profile Components

```python
raisec_profile = {
    "primary_code": "RIA",
    "dimension_scores": {
        "R": 85, "I": 75, "A": 70,
        "S": 45, "E": 35, "C": 25
    },
    "profile_description": "You show strong preference for...",
    "career_themes": ["Engineering", "Research", "Design"],
    "confidence_level": 82.5
}
```

## Analytics and Insights

### Behavioral Analysis

- **Response Patterns**: Preference for certain question types
- **Decision Making Style**: Quick vs. deliberate responses
- **Consistency Patterns**: Stable vs. variable preferences
- **Engagement Level**: Time investment and effort indicators

### Predictive Insights

- **Career Fit Probability**: Likelihood of satisfaction in career areas
- **Interest Stability**: Confidence in long-term preferences
- **Development Areas**: Dimensions with potential for growth
- **Risk Factors**: Indicators of uncertain or conflicted results

### Recommendation Generation

Based on profile analysis:
1. **Primary Career Areas**: Top matching career fields
2. **Secondary Options**: Alternative paths to consider
3. **Development Suggestions**: Skills/interests to explore
4. **Cautions**: Areas of potential mismatch

## Validation and Quality Assurance

### Validity Checks

1. **Response Quality**
   - Completion rate > 80%
   - Minimum time thresholds met
   - Reasonable response patterns

2. **Statistical Validity**
   - Dimensional score distribution
   - Consistency within acceptable ranges
   - No extreme outlier patterns

3. **Reliability Indicators**
   - Test-retest consistency (when available)
   - Internal consistency measures
   - Cross-validation with other assessments

### Quality Flags

```python
validity_flags = [
    "rushed_completion",     # Completed too quickly
    "incomplete_responses",  # Missing answers
    "inconsistent_patterns", # Contradictory responses
    "extreme_outliers",      # Unusual response patterns
    "low_engagement"         # Minimal time investment
]
```

### Reliability Scoring

```python
reliability_score = weighted_average([
    completion_quality * 0.3,
    consistency_score * 0.4,
    engagement_level * 0.2,
    response_validity * 0.1
])
```

## Implementation Details

### Database Storage

Scores and analytics are stored with versioning:
```python
test_scores = {
    "test_id": ObjectId,
    "scoring_version": "v2.0",
    "scored_at": datetime,
    "raisec_code": "RIA",
    "dimension_scores": {...},
    "analytics": {...},
    "validity_flags": [...]
}
```

### Caching Strategy

- **Test Scores**: Cached for 24 hours
- **Analytics**: Cached for 1 hour (more frequently updated)
- **Explanations**: Generated on-demand, cached for 30 minutes

### Performance Optimization

- **Batch Processing**: Multiple tests scored together
- **Async Operations**: Non-blocking database operations
- **Selective Analytics**: Detailed analysis only when requested
- **Incremental Updates**: Only recalculate changed components

## Configuration and Tuning

### Adjustable Parameters

```python
scoring_config = {
    "question_weights": {...},
    "time_adjustment_factors": {...},
    "consistency_thresholds": {...},
    "confidence_parameters": {...},
    "validation_criteria": {...}
}
```

### A/B Testing Support

The system supports configuration variants for:
- Different scoring algorithms
- Alternative weighting schemes
- Varied time adjustments
- Different consistency models

## Future Enhancements

### Planned Improvements

1. **Machine Learning Integration**
   - Predictive career outcome models
   - Adaptive questioning based on responses
   - Personalized scoring adjustments

2. **Enhanced Analytics**
   - Comparative analysis with peer groups
   - Longitudinal tracking of interest changes
   - Integration with actual career outcomes

3. **Advanced Validation**
   - Real-time response quality detection
   - Automated coaching for uncertain results
   - Cross-assessment validation

### Research Initiatives

- Validation studies with career outcome data
- Cross-cultural assessment reliability
- Integration with labor market analytics
- Predictive model development

## Conclusion

The TruScholar RAISEC scoring algorithm provides a comprehensive, reliable, and insightful assessment of career interests. By combining multiple scoring factors, consistency analysis, and confidence metrics, it delivers actionable insights while maintaining high standards of validity and reliability.

The system's flexible architecture allows for continuous improvement and adaptation based on research findings and user feedback, ensuring that the assessment remains current and effective for career guidance.

---

*Last Updated: Day 28-29 Implementation*  
*Version: 2.0*  
*Next Review: After initial deployment validation*