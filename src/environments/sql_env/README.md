# Text-to-SQL Environment for GRPO Training

A custom Verifiers environment for text-to-SQL generation that handles prompt formatting, response parsing, and reward computation for GRPO (Group Relative Policy Optimization) training.

## Overview

This environment implements a `SingleTurnEnv` for stateless text-to-SQL generation where each natural language question maps to a single SQL query. It integrates with the Verifiers framework to provide:

- **Flexible prompt formatting** with multiple template options
- **Robust SQL parsing** from various response formats
- **Reward computation** based on syntax, keywords, and format quality
- **Efficient batch processing** for GRPO training
- **Schema-aware validation** to ensure queries reference valid tables

## Architecture

The environment follows Verifiers' protocol-first design:

```
User Question + Schema → Format Prompt → Model → Parse Response → Compute Reward
                              ↓                         ↓               ↓
                    TextToSQLEnvironment    SQLParser         SQLValidationRubric
```

### Why SingleTurnEnv?

We chose `SingleTurnEnv` over `MultiTurnEnv` or `ToolEnv` because:

- Each query is independent (no conversation history)
- Single input (question + schema) → single output (SQL)
- No database execution needed (rubric-based validation)
- Simpler and more efficient for GRPO

## Components

### 1. Environment Class (`environment.py`)

The main `TextToSQLEnvironment` class that orchestrates the entire pipeline.

**Key Methods:**
- `format_prompt()` - Format input with question and schema
- `parse_response()` - Extract SQL from model output
- `compute_reward()` - Score SQL quality (0.0-1.0)
- `batch_compute_rewards()` - Efficient batch scoring
- `prepare_dataset_sample()` - Convert dataset format
- `get_metrics()` - Aggregate evaluation metrics

### 2. Prompt Templates (`prompts.py`)

Five prompt templates for experimentation:

- `default` - Simple, clear structure
- `instructional` - Explicit instructions
- `few_shot` - With example queries
- `chat` - Chat format with special tokens
- `concise` - Minimal template

All templates support schema injection and few-shot examples.

### 3. Utilities (`utils.py`)

Helper functions for:

- **Schema extraction** - Parse CREATE TABLE statements
- **SQL validation** - Check table/column references
- **Schema truncation** - Handle long schemas
- **Dataset preparation** - Format for GRPO

## Installation

The environment is included in the main project. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `verifiers>=0.1.0` - Framework integration
- `sqlparse>=0.4.4` - SQL parsing
- `transformers`, `datasets` - Model and data handling

## Usage

### Standalone Usage

```python
from src.environments.sql_env import TextToSQLEnvironment
from src.rubrics.sql_rubric import SQLValidationRubric
from src.utils.sql_parser import SQLParser

# Initialize components
parser = SQLParser()
rubric = SQLValidationRubric()
env = TextToSQLEnvironment(
    rubric=rubric,
    parser=parser,
    prompt_template="instructional",
    include_schema=True,
    max_schema_length=1024,
)

# Format a prompt
prompt = env.format_prompt(
    question="How many users are there?",
    context={"schema": "CREATE TABLE users (id INT, name VARCHAR(100))"}
)
print(prompt)

# Parse a response
response = "SELECT COUNT(*) FROM users"
parsed = env.parse_response(response)
print(f"Valid: {parsed['valid']}, SQL: {parsed['sql']}")

# Compute reward
reward = env.compute_reward(response)
print(f"Reward: {reward:.3f}")
```

### With GRPO Training

```python
from datasets import load_dataset
from src.environments.sql_env import prepare_for_grpo

# Load dataset
dataset = load_dataset("b-mc2/sql-create-context", split="train")

# Prepare for GRPO
prepared_dataset = prepare_for_grpo(dataset, env)

# Each sample now has 'prompt', 'question', 'schema', 'reference' fields
# Ready for GRPO trainer
```

### Batch Processing

```python
# Process multiple responses efficiently
responses = [
    "SELECT * FROM users",
    "SELECT name FROM products WHERE price > 100",
    "INSERT INTO orders (id, total) VALUES (1, 50.00)"
]

# Compute rewards in batch (>100 samples/sec)
rewards = env.batch_compute_rewards(responses)
print(f"Rewards: {rewards}")

# Get aggregate metrics
metrics = env.get_metrics(responses)
print(f"Valid SQL: {metrics['valid_sql_pct']:.1f}%")
print(f"Avg Reward: {metrics['avg_reward']:.3f}")
```

## Configuration

Add to `config/training/training.yaml`:

```yaml
environment:
  type: text_to_sql
  prompt_template: instructional  # default, instructional, few_shot, chat, concise
  include_schema: true
  max_schema_length: 1024
  few_shot_examples: 0  # Number of examples to include

  # Response handling
  max_response_length: 512
  stop_sequences: ["\n\n", "<|eot_id|>", "Question:"]

  # Optional reward shaping
  reward_shaping:
    enabled: false
    min_reward: 0.0
    max_reward: 1.0
    normalize: false
```

## Prompt Templates

### Default Template
Simple and clear structure suitable for most models.

```
Given the following database schema:

{schema}

Generate a SQL query to answer this question:
Question: {question}

SQL Query:
```

### Instructional Template
Explicit instructions for better model adherence.

```
You are a SQL expert. Convert the natural language question into a valid SQL query.

Database Schema:
{schema}

Question: {question}

Instructions:
- Generate only the SQL query
- Use proper SQL syntax
- Include all necessary clauses
- Do not include explanations

SQL:
```

### Few-Shot Template
Includes examples to guide the model.

```
Generate SQL queries based on the database schema and examples below.

Schema:
{schema}

{examples}

Now generate SQL for:
Question: {question}
SQL:
```

### Chat Template
Formatted for chat models with special tokens.

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that converts natural language questions into SQL queries.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Database Schema:
{schema}

Question: {question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

SQL Query:
```

### Custom Templates

Create your own template with required placeholders:

```python
custom_template = """
### Task
Convert question to SQL.

### Schema
{schema}

### Question
{question}

### SQL
"""

env = TextToSQLEnvironment(
    rubric=rubric,
    parser=parser,
    prompt_template=custom_template
)
```

**Required placeholders:**
- `{question}` - Natural language question
- `{schema}` - Database schema (if include_schema=True)

**Optional placeholders:**
- `{examples}` - Few-shot examples (if max_examples > 0)

## Customization

### Custom Rubric Weights

Adjust scoring weights for your use case:

```python
rubric = SQLValidationRubric(
    syntax_weight=0.5,    # Syntax validity (default: 0.4)
    keyword_weight=0.3,   # Keyword presence (default: 0.3)
    format_weight=0.2,    # Format quality (default: 0.3)
    strict_mode=False,    # Return 0.0 for syntax errors
)
```

### Custom SQL Keywords

Focus on specific SQL operations:

```python
custom_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY"]

rubric = SQLValidationRubric(sql_keywords=custom_keywords)
env = TextToSQLEnvironment(rubric=rubric, parser=parser)
```

### Schema Truncation Strategy

For large schemas, customize truncation:

```python
from src.environments.sql_env.utils import truncate_schema

# Keep first 512 characters
short_schema = truncate_schema(long_schema, max_length=512)

env = TextToSQLEnvironment(
    rubric=rubric,
    parser=parser,
    max_schema_length=512
)
```

### Few-Shot Learning

Add examples to prompts:

```python
examples = [
    {
        "question": "How many users?",
        "answer": "SELECT COUNT(*) FROM users"
    },
    {
        "question": "List product names",
        "answer": "SELECT name FROM products"
    }
]

env = TextToSQLEnvironment(
    rubric=rubric,
    parser=parser,
    prompt_template="few_shot",
    max_examples=3
)

prompt = env.format_prompt(
    question="Get all orders",
    context={"schema": schema, "examples": examples}
)
```

## Testing

### Run Unit Tests

```bash
# Run all environment tests
pytest tests/test_environment.py -v

# Run specific test class
pytest tests/test_environment.py::TestPromptFormatting -v

# Run with coverage
pytest tests/test_environment.py --cov=src/environments/sql_env --cov-report=html
```

### Test with Real Data

```bash
# Run integration test
python scripts/test_environment.py

# Test with different template
python scripts/test_environment.py training.environment.prompt_template=chat

# Test with custom config
python scripts/test_environment.py training.environment.max_schema_length=512
```

## Performance

### Batch Processing Efficiency

The environment is optimized for GRPO training:

- **Target**: >100 samples/sec for batch reward computation
- **Actual**: ~200-500 samples/sec (depending on hardware)
- **Bottleneck**: SQL parsing with `sqlparse`

### Optimization Tips

1. **Use batch methods**: Always prefer `batch_compute_rewards()` over loops
2. **Truncate schemas**: Long schemas slow down formatting
3. **Disable schema validation**: Skip `validate_sql_against_schema()` if not needed
4. **Cache parsed results**: Reuse parsed SQL for multiple reward computations

## Dataset Format

The environment expects datasets in this format (like `b-mc2/sql-create-context`):

```python
{
    "question": "How many users registered in 2024?",
    "context": "CREATE TABLE users (id INT, name VARCHAR(100), registered_date DATE)",
    "answer": "SELECT COUNT(*) FROM users WHERE YEAR(registered_date) = 2024"
}
```

**Fields:**
- `question` (required) - Natural language question
- `context` (optional) - CREATE TABLE statements
- `answer` (optional) - Reference SQL query

## Troubleshooting

### Issue: Low reward scores

**Solutions:**
- Check SQL syntax with `rubric.check_syntax()`
- Verify keyword presence with `rubric.check_keywords()`
- Review format quality with `rubric.check_format()`
- Get detailed scores with `rubric.get_detailed_scores()`

### Issue: Schema not included in prompt

**Solutions:**
- Ensure `include_schema=True`
- Check context has 'schema' key
- Verify schema isn't too long (check `max_schema_length`)

### Issue: SQL parsing fails

**Solutions:**
- Check parser settings: `SQLParser(extract_code_blocks=True)`
- Try different response formats (code blocks, inline, raw)
- Adjust stop sequences in generation config

### Issue: Slow batch processing

**Solutions:**
- Use `batch_compute_rewards()` instead of loops
- Reduce `max_schema_length` to speed up formatting
- Profile with: `python -m cProfile scripts/test_environment.py`

## Integration with GRPO

The environment seamlessly integrates with the Verifiers GRPO trainer:

```python
from verifiers.trainers import GRPOTrainer
from src.environments.sql_env import TextToSQLEnvironment

# Initialize environment
env = TextToSQLEnvironment(rubric=rubric, parser=parser)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    environment=env,
    train_dataset=prepared_dataset,
    **training_args
)

# Train
trainer.train()
```

The environment provides the required methods:
- `format_prompt()` - Prepare model inputs
- `compute_reward()` - Score model outputs
- `batch_compute_rewards()` - Efficient batch scoring

## Future Enhancements

Potential improvements:

1. **Schema-aware rewards** - Reward queries that use relevant tables
2. **Execution-based validation** - Score based on query execution results
3. **Multi-turn support** - Handle follow-up questions
4. **Custom reward shaping** - Configurable reward functions
5. **Few-shot retrieval** - Automatically select relevant examples
6. **Template optimization** - A/B test different prompts

## References

- [Verifiers Documentation](https://verifiers.readthedocs.io/)
- [Verifiers Environments Guide](https://verifiers.readthedocs.io/en/latest/environments.html)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [b-mc2/sql-create-context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)

## License

This environment is part of the text-to-sql-refine-tuning project. See LICENSE for details.

## Contributing

Contributions welcome! Areas for improvement:

- Additional prompt templates
- Better schema truncation strategies
- Alternative reward functions
- Performance optimizations
- Documentation improvements

Please follow the project's contribution guidelines.
