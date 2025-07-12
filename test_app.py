"""Simple test application to verify basic functionality."""

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create a minimal FastAPI app for testing
app = FastAPI(
    title="TruCareer Test",
    description="Test application to verify setup",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "TruCareer is running!", "status": "success"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "TruCareer",
        "version": "1.0.0"
    }

@app.get("/test/careers")
async def test_careers():
    """Test career endpoint with mock data."""
    mock_careers = [
        {
            "career_id": "software_engineer",
            "career_title": "Software Engineer",
            "category": "information_technology",
            "raisec_match_score": 92.5,
            "description": "Develop and maintain software applications"
        },
        {
            "career_id": "data_scientist",
            "career_title": "Data Scientist", 
            "category": "information_technology",
            "raisec_match_score": 88.3,
            "description": "Analyze data to extract business insights"
        }
    ]
    
    return {
        "careers": mock_careers,
        "total_count": len(mock_careers),
        "message": "Career recommendation system is working!"
    }

if __name__ == "__main__":
    print("Starting TruCareer Test Application...")
    uvicorn.run(
        "test_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )