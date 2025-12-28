"""
Gradio interface for text-to-SQL demo.
Deploy on HuggingFace Spaces.
"""
import os

import gradio as gr

from src.inference.inference_engine import SQLInferenceEngine

# Initialize model (loads on startup)
MODEL_PATH = os.getenv("MODEL_PATH", "./outputs/final_model")
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

engine = SQLInferenceEngine(model_path=MODEL_PATH, base_model_name=BASE_MODEL, load_in_4bit=True)


def generate_sql(question: str, schema: str) -> tuple:
    """
    Generate SQL from question and schema.

    Returns:
        (sql_query, is_valid, metadata)
    """
    result = engine.generate_sql(
        question=question,
        schema=schema if schema else None,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
    )

    status = "✅ Valid SQL" if result["valid"] else "⚠️ May contain errors"
    metadata_str = "\n".join([f"{k}: {v}" for k, v in result["metadata"].items()])

    return result["sql"], status, metadata_str


# Example schemas and questions
EXAMPLES = [
    [
        "Show all users who signed up in 2024",
        "CREATE TABLE users (id INT, name VARCHAR(100), email VARCHAR(100), signup_date DATE);",
    ],
    [
        "What is the average order amount by customer?",
        "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL, order_date DATE);\nCREATE TABLE customers (id INT, name VARCHAR(100));",
    ],
    [
        "Find products that are out of stock",
        "CREATE TABLE products (id INT, name VARCHAR(100), stock_quantity INT, price DECIMAL);",
    ],
]

# Create Gradio interface
with gr.Blocks(title="Text-to-SQL Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Text-to-SQL Generator")
    gr.Markdown(
        "Convert natural language questions into SQL queries using a fine-tuned Llama-3.1-8B model."
    )

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Question", placeholder="Enter your question here...", lines=2
            )
            schema_input = gr.Textbox(
                label="Database Schema (CREATE TABLE statements)",
                placeholder="CREATE TABLE ...",
                lines=6,
            )
            generate_btn = gr.Button("Generate SQL", variant="primary")

        with gr.Column():
            sql_output = gr.Code(label="Generated SQL", language="sql", lines=10)
            status_output = gr.Textbox(label="Validation Status", lines=1)
            metadata_output = gr.Textbox(label="Metadata", lines=3)

    gr.Examples(examples=EXAMPLES, inputs=[question_input, schema_input], label="Example Queries")

    generate_btn.click(
        fn=generate_sql,
        inputs=[question_input, schema_input],
        outputs=[sql_output, status_output, metadata_output],
    )

    gr.Markdown(
        """
    ## About

    This model was fine-tuned using GRPO (Group Relative Policy Optimization) on the b-mc2/sql-create-context dataset.

    **Tips for best results:**
    - Provide the database schema when possible
    - Be specific in your questions
    - Use natural language (no SQL keywords needed)

    **Limitations:**
    - May not handle very complex queries
    - Trained on specific SQL patterns
    - Best with schema context provided

    [GitHub Repository](https://github.com/chrisjcc/text-to-sql-refine-tuning) | [Model Card](https://huggingface.co/chrisjcc/text-to-sql-grpo)
    """
    )

if __name__ == "__main__":
    demo.launch()
