"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.utils.logger import logger

app = FastAPI(
    title="Yamaha Feedback Analysis API",
    description="AI-powered multilingual motorcycle complaint analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

logger.info("FastAPI application initialized")


@app.on_event("startup")
async def startup():
    logger.info("Starting Yamaha Feedback Analysis API")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down Yamaha Feedback Analysis API")
