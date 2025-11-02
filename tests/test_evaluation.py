"""
Unit tests for evaluation module.
"""

import pytest

from src.evaluation.evaluator import SQLEvaluator
from src.evaluation.metrics import ExecutionMetrics, SQLMetrics


class TestSQLMetrics:
    """Test SQL metrics calculations."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return SQLMetrics()

    def test_exact_match_identical(self, metrics):
        """Test exact match with identical queries."""
        sql1 = "SELECT * FROM users WHERE id = 1"
        sql2 = "SELECT * FROM users WHERE id = 1"
        assert metrics.exact_match(sql1, sql2) is True

    def test_exact_match_normalized(self, metrics):
        """Test exact match with normalization."""
        sql1 = "select * from users where id=1"
        sql2 = "SELECT * FROM users WHERE id = 1"
        # Should match after normalization
        assert metrics.exact_match(sql1, sql2) is True

    def test_exact_match_different(self, metrics):
        """Test exact match with different queries."""
        sql1 = "SELECT * FROM users WHERE id = 1"
        sql2 = "SELECT * FROM products WHERE id = 1"
        assert metrics.exact_match(sql1, sql2) is False

    def test_token_accuracy_perfect(self, metrics):
        """Test token accuracy with perfect match."""
        sql1 = "SELECT * FROM users"
        sql2 = "SELECT * FROM users"
        assert metrics.token_level_accuracy(sql1, sql2) == 1.0

    def test_token_accuracy_partial(self, metrics):
        """Test token accuracy with partial match."""
        sql1 = "SELECT name FROM users"
        sql2 = "SELECT id FROM users"
        # 3 out of 4 tokens match (SELECT, FROM, users)
        accuracy = metrics.token_level_accuracy(sql1, sql2)
        assert 0.5 < accuracy < 1.0

    def test_token_accuracy_no_match(self, metrics):
        """Test token accuracy with no match."""
        sql1 = "SELECT * FROM users"
        sql2 = "INSERT INTO products VALUES (1)"
        accuracy = metrics.token_level_accuracy(sql1, sql2)
        assert 0.0 <= accuracy < 0.5

    def test_structural_similarity_identical(self, metrics):
        """Test structural similarity with identical queries."""
        sql1 = "SELECT name FROM users WHERE id = 1"
        sql2 = "SELECT name FROM users WHERE id = 1"
        similarity = metrics.structural_similarity(sql1, sql2)
        assert similarity > 0.8

    def test_structural_similarity_partial(self, metrics):
        """Test structural similarity with partial match."""
        sql1 = "SELECT name FROM users WHERE id = 1"
        sql2 = "SELECT email FROM users WHERE status = 'active'"
        similarity = metrics.structural_similarity(sql1, sql2)
        assert 0.3 < similarity < 0.8

    def test_keyword_f1_perfect(self, metrics):
        """Test keyword F1 with perfect match."""
        sql1 = "SELECT * FROM users WHERE id = 1 ORDER BY name"
        sql2 = "SELECT * FROM users WHERE id = 2 ORDER BY name"
        scores = metrics.keyword_f1(sql1, sql2)
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    def test_keyword_f1_partial(self, metrics):
        """Test keyword F1 with partial match."""
        sql1 = "SELECT * FROM users WHERE id = 1"
        sql2 = "SELECT * FROM users WHERE id = 1 ORDER BY name"
        scores = metrics.keyword_f1(sql1, sql2)
        # sql1 has fewer keywords than sql2
        assert scores["recall"] < 1.0
        assert 0.0 < scores["f1"] < 1.0

    def test_complexity_score_simple(self, metrics):
        """Test complexity score for simple query."""
        sql = "SELECT * FROM users"
        complexity = metrics.complexity_score(sql)
        assert complexity["complexity_level"] == "simple"
        assert complexity["num_joins"] == 0
        assert complexity["has_subquery"] is False

    def test_complexity_score_medium(self, metrics):
        """Test complexity score for medium query."""
        sql = "SELECT u.name, COUNT(*) FROM users u WHERE u.active = 1 GROUP BY u.name"
        complexity = metrics.complexity_score(sql)
        assert complexity["complexity_level"] in ["simple", "medium"]
        assert complexity["has_aggregation"] is True
        assert complexity["has_group_by"] is True

    def test_complexity_score_complex(self, metrics):
        """Test complexity score for complex query."""
        sql = """
        SELECT u.name, o.total
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        LEFT JOIN products p ON o.product_id = p.id
        WHERE u.active = 1 AND o.status = 'completed'
        GROUP BY u.name, o.total
        HAVING COUNT(*) > 5
        ORDER BY o.total DESC
        """
        complexity = metrics.complexity_score(sql)
        assert complexity["complexity_level"] in ["medium", "complex"]
        assert complexity["num_joins"] >= 2
        assert complexity["has_group_by"] is True
        assert complexity["has_having"] is True
        assert complexity["has_order_by"] is True

    def test_edit_distance_identical(self, metrics):
        """Test edit distance with identical queries."""
        sql1 = "SELECT * FROM users"
        sql2 = "SELECT * FROM users"
        distance = metrics.edit_distance(sql1, sql2)
        assert distance == 0

    def test_edit_distance_different(self, metrics):
        """Test edit distance with different queries."""
        sql1 = "SELECT * FROM users"
        sql2 = "SELECT * FROM products"
        distance = metrics.edit_distance(sql1, sql2)
        assert distance > 0

    def test_normalize_sql(self, metrics):
        """Test SQL normalization."""
        sql = "  select  *  from   Users   where   id=1  "
        normalized = metrics._normalize_sql(sql)
        assert "SELECT" in normalized
        assert "FROM" in normalized
        assert "WHERE" in normalized
        assert normalized == normalized.strip()

    def test_tokenize_sql(self, metrics):
        """Test SQL tokenization."""
        sql = "SELECT * FROM users WHERE id = 1"
        tokens = metrics._tokenize_sql(sql)
        assert len(tokens) > 0
        assert "SELECT" in tokens
        assert "FROM" in tokens
        assert "WHERE" in tokens

    def test_extract_keywords(self, metrics):
        """Test keyword extraction."""
        sql = "SELECT * FROM users WHERE id = 1 ORDER BY name"
        keywords = metrics._extract_keywords(sql)
        assert "SELECT" in keywords
        assert "FROM" in keywords
        assert "WHERE" in keywords
        assert "ORDER BY" in keywords

    def test_count_tables(self, metrics):
        """Test table counting."""
        sql = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        count = metrics._count_tables(sql)
        assert count == 2  # users and orders


class TestExecutionMetrics:
    """Test execution metrics."""

    @pytest.fixture
    def exec_metrics(self):
        """Create execution metrics instance."""
        return ExecutionMetrics(db_connection=None)

    def test_execution_accuracy_no_connection(self, exec_metrics):
        """Test execution accuracy without database connection."""
        result = exec_metrics.execution_accuracy("SELECT * FROM users", "SELECT * FROM users")
        assert result["execution_match"] is None
        assert result["error"] == "No database connection"


class TestSQLEvaluator:
    """Test SQL evaluator."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        from unittest.mock import Mock

        from src.inference.inference_engine import SQLInferenceEngine

        # Mock inference engine
        mock_engine = Mock(spec=SQLInferenceEngine)

        evaluator = SQLEvaluator(inference_engine=mock_engine)

        assert evaluator.engine is not None
        assert evaluator.metrics is not None
        assert isinstance(evaluator.metrics, SQLMetrics)

    def test_compute_sample_metrics(self):
        """Test computing metrics for a single sample."""
        from unittest.mock import Mock

        from src.inference.inference_engine import SQLInferenceEngine

        # Mock inference engine
        mock_engine = Mock(spec=SQLInferenceEngine)

        evaluator = SQLEvaluator(inference_engine=mock_engine)

        predicted = "SELECT * FROM users WHERE id = 1"
        reference = "SELECT * FROM users WHERE id = 1"

        metrics = evaluator._compute_sample_metrics(
            predicted=predicted, reference=reference, compute_execution=False
        )

        assert "exact_match" in metrics
        assert "token_accuracy" in metrics
        assert "structural_similarity" in metrics
        assert "keyword_f1" in metrics
        assert metrics["exact_match"] == 1.0

    def test_compute_aggregate_metrics(self):
        """Test computing aggregate metrics."""
        from unittest.mock import Mock

        from src.inference.inference_engine import SQLInferenceEngine

        # Mock inference engine
        mock_engine = Mock(spec=SQLInferenceEngine)

        evaluator = SQLEvaluator(inference_engine=mock_engine)

        # Mock results
        results = [
            {
                "exact_match": True,
                "token_accuracy": 1.0,
                "structural_similarity": 1.0,
                "keyword_f1": 1.0,
                "keyword_precision": 1.0,
                "keyword_recall": 1.0,
                "valid": True,
                "complexity_match": True,
                "edit_distance": 0,
            },
            {
                "exact_match": False,
                "token_accuracy": 0.5,
                "structural_similarity": 0.6,
                "keyword_f1": 0.7,
                "keyword_precision": 0.8,
                "keyword_recall": 0.6,
                "valid": True,
                "complexity_match": False,
                "edit_distance": 10,
            },
        ]

        aggregate = evaluator._compute_aggregate_metrics(results)

        assert "exact_match_rate" in aggregate
        assert "avg_token_accuracy" in aggregate
        assert "valid_sql_rate" in aggregate
        assert aggregate["exact_match_rate"] == 50.0  # 1 out of 2
        assert aggregate["valid_sql_rate"] == 100.0  # both valid

    def test_compute_complexity_metrics(self):
        """Test computing complexity-stratified metrics."""
        from unittest.mock import Mock

        from src.inference.inference_engine import SQLInferenceEngine

        # Mock inference engine
        mock_engine = Mock(spec=SQLInferenceEngine)

        evaluator = SQLEvaluator(inference_engine=mock_engine)

        # Mock results with different complexities
        results = [
            {
                "exact_match": True,
                "token_accuracy": 1.0,
                "structural_similarity": 1.0,
                "valid": True,
                "reference_complexity": "simple",
            },
            {
                "exact_match": False,
                "token_accuracy": 0.5,
                "structural_similarity": 0.6,
                "valid": True,
                "reference_complexity": "complex",
            },
            {
                "exact_match": True,
                "token_accuracy": 0.9,
                "structural_similarity": 0.9,
                "valid": True,
                "reference_complexity": "simple",
            },
        ]

        complexity_metrics = evaluator._compute_complexity_metrics(results)

        assert "simple" in complexity_metrics
        assert "complex" in complexity_metrics
        assert complexity_metrics["simple"]["count"] == 2
        assert complexity_metrics["complex"]["count"] == 1
        assert complexity_metrics["simple"]["exact_match_rate"] == 100.0


# Integration tests
class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_report_generation(self, tmp_path):
        """Test report generation."""
        from unittest.mock import Mock

        from src.inference.inference_engine import SQLInferenceEngine

        # Mock inference engine
        mock_engine = Mock(spec=SQLInferenceEngine)

        evaluator = SQLEvaluator(inference_engine=mock_engine)

        # Mock evaluation results
        results = {
            "aggregate": {
                "exact_match_rate": 80.0,
                "avg_token_accuracy": 85.5,
                "valid_sql_rate": 95.0,
            },
            "by_complexity": {
                "simple": {
                    "count": 10,
                    "exact_match_rate": 90.0,
                },
                "complex": {
                    "count": 5,
                    "exact_match_rate": 60.0,
                },
            },
            "per_sample": [
                {
                    "question": "Test question",
                    "predicted_sql": "SELECT * FROM users",
                    "reference_sql": "SELECT * FROM users",
                    "exact_match": True,
                    "valid": True,
                }
            ],
            "total_samples": 15,
        }

        # Generate report
        output_path = tmp_path / "test_report"
        evaluator.generate_report(results, str(output_path))

        # Check files were created
        assert (output_path / "evaluation_results.json").exists()
        assert (output_path / "per_sample_results.csv").exists()
        assert (output_path / "evaluation_report.md").exists()
