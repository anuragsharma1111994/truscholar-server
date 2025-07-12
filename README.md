# TruScholar (TruCareer) 🎓

An AI-powered RAISEC-based career counselling platform that provides personalized career recommendations through intelligent assessment and analysis.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-6.0-green.svg)
![Redis](https://img.shields.io/badge/Redis-7.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

TruScholar's production-grade career counselling platform that uses the Holland RAISEC (Realistic, Artistic, Investigative, Social, Enterprising, Conventional) assessment methodology combined with advanced AI to provide personalized career guidance.

### Key Features

- **AI-Powered Assessment**: Dynamic question generation based on age groups using GPT-4
- **RAISEC Analysis**: Comprehensive personality assessment using Holland's theory
- **Personalized Recommendations**: Three unique career paths based on individual profiles
- **Multi-Age Support**: Tailored experiences for ages 13-17, 18-25, and 26-35
- **Scalable Architecture**: Built for high performance and reliability
- **Real-time Processing**: Asynchronous task handling for optimal user experience

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: MongoDB 6.0
- **Cache**: Redis 7.0
- **Task Queue**: Celery with Redis broker
- **AI/ML**: LangChain + OpenAI GPT-4

### Infrastructure
- **Container**: Docker
- **Orchestration**: Kubernetes (GKE)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Cloud**: Google Cloud Platform

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- MongoDB 6.0
- Redis 7.0
- OpenAI API key
- Google Cloud SDK (for deployment)
