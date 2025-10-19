"""REST API for text-to-SQL inference.

This module provides a FastAPI-based REST API for generating SQL queries
from natural language questions, with support for single and batch requests.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging
from .inference_engine import SQLInferenceEngine


logger = logging.getLogger(__name__)


# Request models
class SQLGenerationRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    schema: Optional[str] = Field(None, description="Database schema")
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    num_beams: int = Field(1, ge=1, le=10)
    do_sample: bool = Field(False)


class BatchSQLGenerationRequest(BaseModel):
    questions: List[str] = Field(..., description="List of questions")
    schemas: Optional[List[str]] = Field(None, description="List of schemas")
    batch_size: int = Field(4, ge=1, le=32)
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)


# Response models
class SQLGenerationResponse(BaseModel):
    sql: str
    raw_output: str
    valid: bool
    metadata: dict


class BatchSQLGenerationResponse(BaseModel):
    results: List[SQLGenerationResponse]
    total_count: int


# Create API
def create_app(engine: SQLInferenceEngine) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Text-to-SQL API",
        description="REST API for generating SQL queries from natural language",
        version="1.0.0"
    )

    @app.get("/")
    def root():
        """Root endpoint."""
        return {
            "message": "Text-to-SQL API",
            "endpoints": {
                "generate": "/generate",
                "batch_generate": "/batch_generate",
                "health": "/health"
            }
        }

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/generate", response_model=SQLGenerationResponse)
    def generate_sql(request: SQLGenerationRequest):
        """
        Generate SQL query from natural language question.
        """
        try:
            result = engine.generate_sql(
                question=request.question,
                schema=request.schema,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                num_beams=request.num_beams,
                do_sample=request.do_sample
            )

            return SQLGenerationResponse(
                sql=result['sql'],
                raw_output=result['raw_output'],
                valid=result['valid'],
                metadata=result['metadata']
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch_generate", response_model=BatchSQLGenerationResponse)
    def batch_generate_sql(request: BatchSQLGenerationRequest):
        """
        Generate SQL queries for multiple questions.
        """
        try:
            results = engine.batch_generate_sql(
                questions=request.questions,
                schemas=request.schemas,
                batch_size=request.batch_size,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature
            )

            responses = [
                SQLGenerationResponse(
                    sql=r['sql'],
                    raw_output=r['raw_output'],
                    valid=r['valid'],
                    metadata=r['metadata']
                )
                for r in results
            ]

            return BatchSQLGenerationResponse(
                results=responses,
                total_count=len(responses)
            )

        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


def serve_api(
    model_path: str,
    base_model_name: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    load_in_4bit: bool = False
):
    """
    Start API server.

    Args:
        model_path: Path to fine-tuned model
        base_model_name: Base model name (for PEFT models)
        host: Host to bind to
        port: Port to bind to
        load_in_4bit: Use 4-bit quantization
    """
    logger.info(f"Starting API server on {host}:{port}")

    # Initialize engine
    logger.info("Loading model...")
    engine = SQLInferenceEngine(
        model_path=model_path,
        base_model_name=base_model_name,
        load_in_4bit=load_in_4bit
    )

    # Create app
    app = create_app(engine)

    # Run server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL API Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--load-in-4bit", action="store_true")

    args = parser.parse_args()

    serve_api(
        model_path=args.model_path,
        base_model_name=args.base_model,
        host=args.host,
        port=args.port,
        load_in_4bit=args.load_in_4bit
    )
