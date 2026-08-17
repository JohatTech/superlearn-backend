"""
Vercel Serverless Function entrypoint.
Imports the FastAPI application from app.main.
"""
from app.main import app

# Vercel serverless requires the ASGI application instance at module level
__all__ = ["app"]
