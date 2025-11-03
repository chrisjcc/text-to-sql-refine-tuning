"""REST API for text-to-SQL inference.

This module provides a FastAPI-based REST API for generating SQL queries
from natural language questions, with support for single and batch requests.
"""

import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference_engine import SQLInferenceEngine

logger = logging.getLogger(__name__)


# Request models
class SQLGenerationRequest(BaseModel):
    """Request model for single SQL generation.

    Attributes:
        question: Natural language question to convert to SQL.
        schema: Optional database schema (CREATE TABLE statements).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        num_beams: Number of beams for beam search.
        do_sample: Whether to use sampling.
    """

    question: str = Field(..., description="Natural language question")
    schema: str | None = Field(None, description="Database schema")  # type: ignore[assignment]
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    num_beams: int = Field(1, ge=1, le=10)
    do_sample: bool = Field(False)


class BatchSQLGenerationRequest(BaseModel):
    """Request model for batch SQL generation.

    Attributes:
        questions: List of natural language questions.
        schemas: Optional list of database schemas (one per question).
        batch_size: Batch size for processing.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
    """

    questions: list[str] = Field(..., description="List of questions")
    schemas: list[str] | None = Field(None, description="List of schemas")
    batch_size: int = Field(4, ge=1, le=32)
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)


# Response models
class SQLGenerationResponse(BaseModel):
    """Response model for SQL generation.

    Attributes:
        sql: Generated SQL query.
        raw_output: Raw model output before parsing.
        valid: Whether the SQL is valid.
        metadata: Additional metadata about generation.
    """

    sql: str
    raw_output: str
    valid: bool
    metadata: dict


class BatchSQLGenerationResponse(BaseModel):
    """Response model for batch SQL generation.

    Attributes:
        results: List of SQL generation results.
        total_count: Total number of results.
    """

    results: list[SQLGenerationResponse]
    total_count: int


# Create API
def create_app(engine: SQLInferenceEngine) -> FastAPI:
    """Create FastAPI application with SQL generation endpoints.

    Args:
        engine: SQL inference engine for generating queries.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Text-to-SQL API",
        description="REST API for generating SQL queries from natural language",
        version="1.0.0",
    )

    @app.get("/")
    def root() -> dict[str, str | dict[str, str]]:
        """Root endpoint with API information.

        Returns:
            Dictionary with API message and available endpoints.
        """
        return {
            "message": "Text-to-SQL API",
            "endpoints": {
                "generate": "/generate",
                "batch_generate": "/batch_generate",
                "health": "/health",
            },
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            Dictionary with health status.
        """
        return {"status": "healthy"}

    @app.post("/generate", response_model=SQLGenerationResponse)
    def generate_sql(request: SQLGenerationRequest) -> SQLGenerationResponse:
        """Generate SQL query from natural language question.

        Args:
            request: SQL generation request with question and parameters.

        Returns:
            SQL generation response with query and metadata.

        Raises:
            HTTPException: If generation fails.
        """
        try:
            result = engine.generate_sql(
                question=request.question,
                schema=request.schema,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                num_beams=request.num_beams,
                do_sample=request.do_sample,
            )

            return SQLGenerationResponse(
                sql=result["sql"],
                raw_output=result["raw_output"],
                valid=result["valid"],
                metadata=result["metadata"],
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/batch_generate", response_model=BatchSQLGenerationResponse)
    def batch_generate_sql(
        request: BatchSQLGenerationRequest,
    ) -> BatchSQLGenerationResponse:
        """Generate SQL queries for multiple questions.

        Args:
            request: Batch SQL generation request with questions and
                parameters.

        Returns:
            Batch SQL generation response with results and count.

        Raises:
            HTTPException: If batch generation fails.
        """
        try:
            results = engine.batch_generate_sql(
                questions=request.questions,
                schemas=request.schemas,
                batch_size=request.batch_size,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )

            responses = [
                SQLGenerationResponse(
                    sql=r["sql"],
                    raw_output=r["raw_output"],
                    valid=r["valid"],
                    metadata=r["metadata"],
                )
                for r in results
            ]

            return BatchSQLGenerationResponse(results=responses, total_count=len(responses))

        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def serve_api(
    model_path: str,
    base_model_name: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    load_in_4bit: bool = False,
) -> None:
    """Start API server.

    Args:
        model_path: Path to fine-tuned model or HuggingFace model ID.
        base_model_name: Base model name (for PEFT models). Defaults to None.
        host: Host to bind to. Defaults to "0.0.0.0".
        port: Port to bind to. Defaults to 8000.
        load_in_4bit: Use 4-bit quantization for inference.
            Defaults to False.

    Returns:
        None. Runs server until interrupted.
    """
    logger.info(f"Starting API server on {host}:{port}")

    # Initialize engine
    logger.info("Loading model...")
    engine = SQLInferenceEngine(
        model_path=model_path,
        base_model_name=base_model_name,
        load_in_4bit=load_in_4bit,
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
        load_in_4bit=args.load_in_4bit,
    )
