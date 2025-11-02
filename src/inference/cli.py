"""Interactive command-line interface for text-to-SQL generation.

This module provides a Rich-powered CLI for interactive text-to-SQL generation
with schema loading, syntax highlighting, and result visualization.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from .inference_engine import SQLInferenceEngine

logger = logging.getLogger(__name__)


class SQLInteractiveCLI:
    """Interactive CLI for text-to-SQL generation.

    Provides a Rich-powered command-line interface with schema loading,
    SQL syntax highlighting, and result visualization.

    Attributes:
        engine: Inference engine for generating SQL.
        console: Rich console for output.
        schema: Currently loaded database schema.
        logger: Logger instance for this class.
    """

    def __init__(self, engine: SQLInferenceEngine) -> None:
        """Initialize CLI.

        Args:
            engine: SQL inference engine for generating queries.
        """
        self.engine = engine
        self.console = Console()
        self.schema: str | None = None
        self.logger = logging.getLogger(__name__)

    def set_schema(self, schema: str) -> None:
        """Set database schema for queries.

        Args:
            schema: Database schema string (CREATE TABLE statements).

        Returns:
            None. Updates internal schema and displays confirmation.
        """
        self.schema = schema
        self.console.print(
            f"[green]Schema loaded ({len(schema)} characters)[/green]"
        )

    def load_schema_from_file(self, filepath: str) -> None:
        """Load schema from file.

        Args:
            filepath: Path to schema file.

        Returns:
            None. Updates internal schema and displays confirmation.
        """
        with Path(filepath).open() as f:
            self.schema = f.read()
        self.console.print(f"[green]Schema loaded from {filepath}[/green]")

    def generate(self, question: str, **kwargs: Any) -> dict[str, Any]:
        """Generate SQL from question.

        Args:
            question: Natural language question.
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary with generated SQL and metadata.
        """
        return self.engine.generate_sql(
            question=question, schema=self.schema, **kwargs
        )

    def display_result(self, result: dict[str, Any]) -> None:
        """Display generation result.

        Args:
            result: Generation result with SQL and metadata.

        Returns:
            None. Prints formatted result to console.
        """
        # Display SQL
        sql_syntax = Syntax(
            result["sql"], "sql", theme="monokai", line_numbers=True
        )

        self.console.print(
            Panel(sql_syntax, title="Generated SQL", border_style="blue")
        )

        # Display metadata
        valid_status = (
            "[green]✓ Valid[/green]"
            if result["valid"]
            else "[red]✗ Invalid[/red]"
        )
        self.console.print(f"\nStatus: {valid_status}")

        if result["metadata"]:
            self.console.print("\nMetadata:")
            for key, value in result["metadata"].items():
                self.console.print(f"  {key}: {value}")

    def run(self) -> None:
        """Run interactive CLI.

        Main loop that prompts for questions and generates SQL until
        user exits.

        Returns:
            None. Runs until user exits with quit command or Ctrl+C.
        """
        self.console.print(
            Panel.fit(
                "[bold cyan]Text-to-SQL Interactive CLI[/bold cyan]\n"
                "Type 'help' for commands, 'quit' to exit",
                border_style="cyan",
            )
        )

        while True:
            try:
                question = Prompt.ask("\n[yellow]Question[/yellow]")

                if question.lower() in ["quit", "exit", "q"]:
                    self.console.print("[green]Goodbye![/green]")
                    break

                if question.lower() == "help":
                    self.show_help()
                    continue

                if question.lower().startswith("schema "):
                    filepath = question[7:].strip()
                    self.load_schema_from_file(filepath)
                    continue

                if question.lower() == "clear":
                    self.schema = None
                    self.console.print("[green]Schema cleared[/green]")
                    continue

                # Generate SQL
                self.console.print("[dim]Generating SQL...[/dim]")
                result = self.generate(question)
                self.display_result(result)

            except KeyboardInterrupt:
                self.console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    def show_help(self) -> None:
        """Show help message.

        Returns:
            None. Displays help panel with available commands.
        """
        help_text = """
[bold]Commands:[/bold]
  help              - Show this help message
  schema <file>     - Load database schema from file
  clear             - Clear loaded schema
  quit/exit/q       - Exit the CLI

[bold]Usage:[/bold]
  1. Optionally load a schema: schema path/to/schema.sql
  2. Enter your natural language question
  3. View the generated SQL query
        """
        self.console.print(
            Panel(help_text, title="Help", border_style="cyan")
        )


def main() -> None:
    """Main CLI entry point.

    Parses command-line arguments, initializes the inference engine,
    and starts the interactive CLI.

    Returns:
        None. Runs CLI until user exits.
    """
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Interactive CLI"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (for PEFT models)",
    )
    parser.add_argument(
        "--schema", type=str, default=None, help="Path to schema file"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )

    args = parser.parse_args()

    # Initialize engine
    console = Console()
    console.print("[cyan]Loading model...[/cyan]")

    engine = SQLInferenceEngine(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )

    # Initialize CLI
    cli = SQLInteractiveCLI(engine)

    # Load schema if provided
    if args.schema:
        cli.load_schema_from_file(args.schema)

    # Run
    cli.run()


if __name__ == "__main__":
    main()
