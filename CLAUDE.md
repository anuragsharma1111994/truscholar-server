# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TruScholar (TruCareer) is an AI-powered RAISEC-based career counselling platform built with FastAPI, MongoDB, and LangChain. It provides personalized career recommendations through intelligent assessment and analysis for different age groups (13-17, 18-25, 26-35).

## Essential Development Commands

### Environment Setup
```bash
# Install all dependencies (including dev and test)
poetry install --with dev,test

# Start required services
docker run -d -p 27017:27017 --name mongodb mongo:6.0
docker run -d -p 6379:6379 --name redis redis:7.0

# Copy and configure environment
cp .env.example .env
# Update .env with API keys and configuration
```

### Running the Application
```bash
# Start FastAPI development server
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (in separate terminal)
poetry run celery -A src.workers.celery_app worker --loglevel=info

# Start Celery scheduler (if needed)
poetry run celery -A src.workers.celery_app beat --loglevel=info
```

### Testing
```bash
# Run all tests with coverage
poetry run pytest

# Run specific test categories
poetry run pytest -m unit           # Unit tests only
poetry run pytest -m integration    # Integration tests only
poetry run pytest -m "not slow"     # Skip slow tests

# Run specific test file
poetry run pytest tests/unit/test_services/test_user_service.py
poetry run pytest tests/unit/test_services/test_test_service.py
poetry run pytest tests/unit/test_services/test_question_service.py
poetry run pytest tests/integration/test_api/test_test_endpoints.py
poetry run pytest tests/integration/test_api/test_question_endpoints.py

# Run with coverage report
poetry run pytest --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code (MUST run before committing)
poetry run black src tests

# Sort imports
poetry run isort src tests

# Type checking
poetry run mypy src

# Linting
poetry run flake8 src tests
poetry run pylint src

# Security scanning
poetry run bandit -r src

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## Architecture Overview

### Clean Architecture Layers
1. **API Layer** (`/src/routers`): FastAPI route handlers for HTTP endpoints
2. **Service Layer** (`/src/services`): Business logic and orchestration
3. **Model Layer** (`/src/models`): MongoDB document models with Pydantic
4. **Schema Layer** (`/src/schemas`): Request/response validation schemas

### Key Components
- **LangChain Integration** (`/src/langchain_handlers`): AI chains, prompts, and parsers for RAISEC assessment
- **Multi-LLM Support** (`/src/llm`): Abstraction layer supporting OpenAI, Anthropic, and Google AI
- **Async Task Processing**: Celery with Redis for background jobs
- **Caching Layer** (`/src/cache`): Redis-based caching for performance

### Database Design
- All models extend `BaseDocument` with common fields (id, created_at, updated_at)
- MongoDB collections: users, tests, questions, responses, recommendations
- Consistent CRUD patterns using Motor async driver

### API Design Patterns
- RESTful endpoints with OpenAPI documentation
- JWT-based authentication
- Comprehensive middleware stack (CORS, rate limiting, logging)
- Request/response validation using Pydantic schemas

### Age-Based Customization
- Static questions organized by age groups: 13-17, 18-25, 26-35
- Dynamic question generation using LLMs
- Age-appropriate career recommendations

## Development Workflow

### Before Making Changes
1. Understand existing patterns by checking similar code
2. Check imports to understand which libraries are already in use
3. Follow existing naming conventions and code style

### Making Changes
1. Create/modify code following existing patterns
2. Add appropriate type hints
3. Update or add tests for new functionality
4. Run formatter and linter before committing

### Testing Changes
```bash
# Run affected tests
poetry run pytest path/to/test/file.py

# Check code quality
poetry run black src tests
poetry run mypy src
poetry run flake8 src tests
```

## Important Conventions

### Code Style
- Use type hints for all function parameters and returns
- Follow existing import ordering (standard library, third-party, local)
- Use async/await consistently throughout the codebase
- No inline comments unless absolutely necessary

### Error Handling
- Use custom exceptions from `/src/utils/exceptions.py`
- Log errors appropriately using the structured logging setup
- Return consistent error responses using schema models

### Security
- Never commit secrets or API keys
- Use environment variables for all configuration
- Validate all user inputs using Pydantic schemas
- Apply rate limiting to public endpoints

### Testing
- Maintain minimum 80% code coverage
- Use pytest fixtures for common test data
- Mock external services (LLMs, databases) in unit tests
- Write integration tests for API endpoints

## Common Tasks

### Adding New API Endpoint
1. Create router in `/src/routers`
2. Add service logic in `/src/services`
3. Define request/response schemas in `/src/schemas`
4. Add appropriate tests
5. Update API documentation if needed

### Adding New Celery Task
1. Create task in `/src/workers/tasks`
2. Register in celery app configuration
3. Add task tests
4. Consider adding monitoring/logging

### Modifying Database Models
1. Update model in `/src/models`
2. Consider migration strategy (no automatic migrations)
3. Update related schemas
4. Update tests

### Working with LangChain
1. Chains are in `/src/langchain_handlers/chains`
2. Prompts are in `/src/langchain_handlers/prompts`
3. Parsers handle output formatting
4. Always test with mock LLM responses first