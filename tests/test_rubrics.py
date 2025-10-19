"""Unit tests for SQL validation rubrics and parsers.

Tests the SQLParser, SQLValidationRubric, and BatchSQLScorer components
used for computing rewards during GRPO training.
"""

from src.rubrics.batch_scorer import BatchSQLScorer
from src.rubrics.sql_rubric import SQLValidationRubric
from src.utils.sql_parser import SQLParser


class TestSQLParser:
    """Tests for SQLParser class."""

    def test_extract_from_code_block(self):
        """Test extraction from markdown code blocks."""
        parser = SQLParser()

        # Test with sql tag
        text = "```sql\nSELECT * FROM users\n```"
        result = parser.extract_sql(text)
        assert result == "SELECT * FROM users"

        # Test without language tag
        text = "```\nSELECT name FROM products\n```"
        result = parser.extract_sql(text)
        assert result == "SELECT name FROM products"

        # Test with SQL keyword
        text = "```SQL\nINSERT INTO table VALUES (1, 2)\n```"
        result = parser.extract_sql(text)
        assert result == "INSERT INTO table VALUES (1, 2)"

    def test_extract_raw_sql(self):
        """Test extraction of raw SQL."""
        parser = SQLParser()

        # Test simple SELECT
        text = "SELECT * FROM users WHERE id = 1"
        result = parser.extract_sql(text)
        assert result == "SELECT * FROM users WHERE id = 1"

        # Test with leading text
        text = "Here is the query: SELECT name FROM products"
        result = parser.extract_sql(text)
        assert "SELECT name FROM products" in result

    def test_extract_inline_code(self):
        """Test extraction from inline code."""
        parser = SQLParser()

        # Test inline backticks
        text = "The query is `SELECT * FROM users`"
        result = parser.extract_sql(text)
        assert result == "SELECT * FROM users"

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        parser = SQLParser()

        # Empty string
        assert parser.extract_sql("") is None

        # None
        assert parser.extract_sql(None) is None

        # Non-SQL text
        assert parser.extract_sql("This is not SQL") is None

    def test_malformed_sql(self):
        """Test handling of malformed SQL."""
        parser = SQLParser()

        # SQL with typo (should still extract)
        text = "SELECT * FORM users"  # Typo: FORM instead of FROM
        result = parser.extract_sql(text)
        assert result is not None
        assert "SELECT" in result

    def test_long_queries(self):
        """Test handling of very long queries."""
        parser = SQLParser(max_sql_length=100)

        # Very long query should be truncated
        text = "SELECT " + "column, " * 50 + "FROM users"
        result = parser.extract_sql(text)
        assert result is not None
        assert len(result) <= 100

    def test_queries_with_comments(self):
        """Test SQL queries with comments."""
        parser = SQLParser()

        text = """```sql
        -- This is a comment
        SELECT * FROM users
        /* Another comment */
        WHERE id = 1
        ```"""
        result = parser.extract_sql(text)
        assert result is not None
        assert "SELECT" in result

    def test_multiple_queries(self):
        """Test text with multiple SQL queries."""
        parser = SQLParser()

        # Should extract first query
        text = "SELECT * FROM users; SELECT * FROM products"
        result = parser.extract_sql(text)
        assert result is not None
        assert "SELECT" in result

    def test_clean_sql(self):
        """Test SQL cleaning functionality."""
        parser = SQLParser(clean_whitespace=True)

        # Multiple spaces should be normalized
        sql = "SELECT  *   FROM    users"
        cleaned = parser.clean_sql(sql)
        assert "  " not in cleaned
        assert "SELECT * FROM users" == cleaned

    def test_detect_sql_pattern(self):
        """Test SQL pattern detection."""
        parser = SQLParser()

        # Valid SQL patterns
        assert parser.detect_sql_pattern("SELECT * FROM users")
        assert parser.detect_sql_pattern("INSERT INTO table VALUES (1)")
        assert parser.detect_sql_pattern("UPDATE users SET name = 'test'")
        assert parser.detect_sql_pattern("DELETE FROM users WHERE id = 1")
        assert parser.detect_sql_pattern("CREATE TABLE users (id INT)")

        # Invalid patterns
        assert not parser.detect_sql_pattern("This is plain text")
        assert not parser.detect_sql_pattern("")
        assert not parser.detect_sql_pattern(None)

    def test_is_valid_format(self):
        """Test format validation."""
        parser = SQLParser()

        # Valid formats
        assert parser.is_valid_format("SELECT * FROM users")
        assert parser.is_valid_format("```sql\nSELECT * FROM users\n```")

        # Invalid formats
        assert not parser.is_valid_format("")
        assert not parser.is_valid_format("Short")
        assert not parser.is_valid_format("Not SQL at all")

    def test_parse_batch(self):
        """Test batch parsing."""
        parser = SQLParser()

        texts = [
            "SELECT * FROM users",
            "```sql\nINSERT INTO products VALUES (1, 'test')\n```",
            "Not SQL",
            "UPDATE users SET name = 'test'"
        ]

        results = parser.parse_batch(texts)

        assert len(results) == 4
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None
        assert results[3] is not None


class TestSQLValidationRubric:
    """Tests for SQLValidationRubric class."""

    def test_score_valid_sql(self):
        """Test scoring of valid SQL queries."""
        rubric = SQLValidationRubric()

        # Perfect query
        score = rubric.score("SELECT * FROM users WHERE id = 1")
        assert 0.8 <= score <= 1.0  # Should score high

        # Query with JOIN
        score = rubric.score("SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id")
        assert 0.8 <= score <= 1.0

    def test_score_invalid_sql(self):
        """Test scoring of invalid SQL."""
        rubric = SQLValidationRubric()

        # Not SQL at all
        score = rubric.score("This is not SQL")
        assert score == 0.0

        # SQL with typo
        score = rubric.score("SELECT * FORM users")  # FORM instead of FROM
        assert score < 0.5  # Should score low

    def test_score_empty_input(self):
        """Test scoring of empty inputs."""
        rubric = SQLValidationRubric()

        assert rubric.score("") == 0.0
        assert rubric.score(None) == 0.0

    def test_check_syntax(self):
        """Test syntax validation."""
        rubric = SQLValidationRubric()

        # Valid SQL
        is_valid, score = rubric.check_syntax("SELECT * FROM users")
        assert is_valid is True
        assert score == 1.0

        # Invalid SQL
        is_valid, score = rubric.check_syntax("SELECT FORM users")
        assert is_valid is False
        assert score < 1.0

        # Empty
        is_valid, score = rubric.check_syntax("")
        assert is_valid is False
        assert score == 0.0

    def test_check_keywords(self):
        """Test keyword detection."""
        rubric = SQLValidationRubric()

        # Query with many keywords
        sql = "SELECT * FROM users WHERE id = 1 ORDER BY name LIMIT 10"
        score = rubric.check_keywords(sql)
        assert score >= 0.75  # Has SELECT, FROM, WHERE, ORDER BY, LIMIT

        # Query with few keywords
        sql = "SELECT * FROM users"
        score = rubric.check_keywords(sql)
        assert 0.4 <= score <= 0.7

        # No SQL keywords
        score = rubric.check_keywords("Not SQL")
        assert score == 0.0

    def test_check_format(self):
        """Test format quality checking."""
        rubric = SQLValidationRubric()

        # Good format
        score = rubric.check_format("SELECT * FROM users WHERE id = 1")
        assert score >= 0.7

        # Format with code block
        score = rubric.check_format("```sql\nSELECT * FROM users\n```")
        assert score >= 0.7

        # Poor format
        score = rubric.check_format("Some text and maybe SQL somewhere")
        assert score < 0.5

    def test_get_detailed_scores(self):
        """Test detailed score breakdown."""
        rubric = SQLValidationRubric()

        output = "SELECT * FROM users WHERE id = 1"
        details = rubric.get_detailed_scores(output)

        assert "total" in details
        assert "syntax" in details
        assert "keywords" in details
        assert "format" in details
        assert "extracted_sql" in details
        assert "weights" in details

        assert 0.0 <= details["total"] <= 1.0
        assert 0.0 <= details["syntax"] <= 1.0
        assert 0.0 <= details["keywords"] <= 1.0
        assert 0.0 <= details["format"] <= 1.0

    def test_custom_weights(self):
        """Test rubric with custom weights."""
        rubric = SQLValidationRubric(
            syntax_weight=0.5,
            keyword_weight=0.3,
            format_weight=0.2
        )

        score = rubric.score("SELECT * FROM users")
        assert 0.0 <= score <= 1.0

        # Verify weights in detailed scores
        details = rubric.get_detailed_scores("SELECT * FROM users")
        assert details["weights"]["syntax"] == 0.5
        assert details["weights"]["keywords"] == 0.3
        assert details["weights"]["format"] == 0.2

    def test_strict_mode(self):
        """Test strict mode behavior."""
        rubric = SQLValidationRubric(strict_mode=True)

        # Valid SQL should score normally
        score = rubric.score("SELECT * FROM users")
        assert score > 0.0

        # Invalid SQL should return 0.0 in strict mode
        score = rubric.score("SELECT FORM users")
        assert score == 0.0

    def test_score_batch(self):
        """Test batch scoring."""
        rubric = SQLValidationRubric()

        outputs = [
            "SELECT * FROM users",
            "INSERT INTO products VALUES (1, 'test')",
            "Not SQL",
            "UPDATE users SET name = 'test'"
        ]

        scores = rubric.score_batch(outputs)

        assert len(scores) == 4
        assert scores[0] > 0.5  # Valid SQL
        assert scores[1] > 0.5  # Valid SQL
        assert scores[2] == 0.0  # Not SQL
        assert scores[3] > 0.5  # Valid SQL

    def test_edge_cases(self):
        """Test edge cases."""
        rubric = SQLValidationRubric()

        # Very long query
        long_query = "SELECT " + ", ".join([f"col{i}" for i in range(100)]) + " FROM users"
        score = rubric.score(long_query)
        assert 0.0 <= score <= 1.0

        # Query with special characters
        score = rubric.score("SELECT * FROM users WHERE name = 'O''Brien'")
        assert score > 0.5

        # Multiple statements
        score = rubric.score("SELECT * FROM users; DROP TABLE users;")
        assert score > 0.0  # Should extract first statement

    def test_code_block_variations(self):
        """Test various code block formats."""
        rubric = SQLValidationRubric()

        # Different code block formats
        formats = [
            "```sql\nSELECT * FROM users\n```",
            "```SQL\nSELECT * FROM users\n```",
            "```\nSELECT * FROM users\n```",
            "SELECT * FROM users",  # No code block
        ]

        for fmt in formats:
            score = rubric.score(fmt)
            assert score > 0.5, f"Failed for format: {fmt}"


class TestBatchSQLScorer:
    """Tests for BatchSQLScorer class."""

    def test_score_batch_sequential(self):
        """Test sequential batch scoring."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric, use_parallel=False)

        outputs = [
            "SELECT * FROM users",
            "INSERT INTO products VALUES (1, 'test')",
            "UPDATE users SET name = 'test'"
        ]

        scores = scorer.score_batch(outputs)

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(s > 0.5 for s in scores)  # All valid SQL

    def test_score_batch_parallel(self):
        """Test parallel batch scoring."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric, use_parallel=True)

        # Large batch to trigger parallel processing
        outputs = ["SELECT * FROM users WHERE id = " + str(i) for i in range(20)]

        scores = scorer.score_batch(outputs)

        assert len(scores) == 20
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert all(s > 0.5 for s in scores)

    def test_score_with_metadata(self):
        """Test scoring with metadata."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric)

        outputs = [
            "SELECT * FROM users",
            "Not SQL"
        ]

        metadata = scorer.score_with_metadata(outputs)

        assert len(metadata) == 2

        # First output (valid SQL)
        assert metadata[0]["total"] > 0.5
        assert "syntax" in metadata[0]
        assert "keywords" in metadata[0]
        assert "format" in metadata[0]
        assert "extracted_sql" in metadata[0]

        # Second output (not SQL)
        assert metadata[1]["total"] == 0.0

    def test_caching(self):
        """Test result caching."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric)

        scorer.enable_cache(True)

        outputs = ["SELECT * FROM users"] * 10

        # First call should compute
        scores1 = scorer.score_batch(outputs, use_cache=True)

        # Second call should use cache
        scores2 = scorer.score_batch(outputs, use_cache=True)

        assert scores1 == scores2

        # Clear cache
        scorer.clear_cache()

        # Disable cache
        scorer.enable_cache(False)

    def test_compute_batch_statistics(self):
        """Test batch statistics computation."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric)

        outputs = [
            "SELECT * FROM users",
            "INSERT INTO products VALUES (1, 'test')",
            "Not SQL",
            "UPDATE users SET name = 'test'"
        ]

        stats = scorer.compute_batch_statistics(outputs)

        assert "count" in stats
        assert "mean_score" in stats
        assert "std_score" in stats
        assert "min_score" in stats
        assert "max_score" in stats
        assert "valid_sql_count" in stats
        assert "valid_sql_percentage" in stats

        assert stats["count"] == 4
        assert 0.0 <= stats["mean_score"] <= 1.0
        assert stats["valid_sql_count"] >= 3  # At least 3 valid SQL queries

    def test_empty_batch(self):
        """Test handling of empty batch."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric)

        scores = scorer.score_batch([])
        assert scores == []

        metadata = scorer.score_with_metadata([])
        assert metadata == []

        stats = scorer.compute_batch_statistics([])
        assert stats["count"] == 0
        assert stats["mean_score"] == 0.0

    def test_error_handling(self):
        """Test error handling in batch scoring."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric)

        # Mix of valid and invalid inputs
        outputs = [
            "SELECT * FROM users",
            None,  # Will be handled gracefully
            "",    # Empty string
            "UPDATE users SET name = 'test'"
        ]

        # Should not raise exception
        scores = scorer.score_batch(outputs)
        assert len(scores) == 4


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from parsing to scoring."""
        parser = SQLParser()
        rubric = SQLValidationRubric(parser=parser)
        scorer = BatchSQLScorer(rubric)

        # Simulate model outputs
        model_outputs = [
            "```sql\nSELECT * FROM users WHERE id = 1\n```",
            "The query is: INSERT INTO products (name, price) VALUES ('test', 10.0)",
            "Not a valid SQL query",
            "UPDATE users SET active = true WHERE last_login < '2024-01-01'",
        ]

        # Score batch
        scores = scorer.score_batch(model_outputs)

        assert len(scores) == 4
        assert scores[0] > 0.7  # Code block with valid SQL
        assert scores[1] > 0.6  # Inline valid SQL
        assert scores[2] == 0.0  # Not SQL
        assert scores[3] > 0.7  # Valid SQL

        # Get detailed metadata
        metadata = scorer.score_with_metadata(model_outputs)

        assert len(metadata) == 4
        assert metadata[0]["extracted_sql"] is not None
        assert metadata[2]["extracted_sql"] is None

        # Compute statistics
        stats = scorer.compute_batch_statistics(model_outputs)

        assert stats["count"] == 4
        assert 0.0 < stats["mean_score"] < 1.0
        assert stats["valid_sql_count"] == 3
        assert stats["valid_sql_percentage"] == 75.0

    def test_config_integration(self):
        """Test integration with config values."""
        # Simulate config values
        config_keywords = [
            "SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE"
        ]
        config_weights = {
            "syntax": 0.4,
            "keyword": 0.3,
            "format": 0.3
        }

        rubric = SQLValidationRubric(
            sql_keywords=config_keywords,
            syntax_weight=config_weights["syntax"],
            keyword_weight=config_weights["keyword"],
            format_weight=config_weights["format"],
        )

        score = rubric.score("SELECT * FROM users WHERE id = 1")
        assert 0.8 <= score <= 1.0

    def test_grpo_reward_computation(self):
        """Test reward computation as it would be used in GRPO."""
        rubric = SQLValidationRubric()
        scorer = BatchSQLScorer(rubric)

        # Simulate a batch of generations from GRPO
        batch_size = 4
        num_generations = 3

        all_outputs = []
        for i in range(batch_size):
            # Each input generates multiple candidates
            for j in range(num_generations):
                all_outputs.append(
                    f"SELECT * FROM users WHERE id = {i} LIMIT {j+1}"
                )

        # Score all generations
        rewards = scorer.score_batch(all_outputs)

        assert len(rewards) == batch_size * num_generations
        assert all(0.0 <= r <= 1.0 for r in rewards)

        # Reshape for GRPO (batch_size, num_generations)
        import numpy as np
        rewards_array = np.array(rewards).reshape(batch_size, num_generations)

        # Verify shape
        assert rewards_array.shape == (batch_size, num_generations)

        # All should be high scores (valid SQL)
        assert np.mean(rewards_array) > 0.7
