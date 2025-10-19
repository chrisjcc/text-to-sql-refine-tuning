"""Interactive command-line interface for text-to-SQL generation.

This module provides a Rich-powered CLI for interactive text-to-SQL generation
with schema loading, syntax highlighting, and result visualization.
"""

import argparse
import logging
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from .inference_engine import SQLInferenceEngine

logger = logging.getLogger(__name__)


class SQLInteractiveCLI:
    """Interactive CLI for text-to-SQL generation."""

    def __init__(self, engine: SQLInferenceEngine):
        """
        Initialize CLI.

        Args:
            engine: Inference engine
        """
        self.engine = engine
        self.console = Console()
        self.schema = None
        self.logger = logging.getLogger(__name__)

    def set_schema(self, schema: str):
        """Set database schema for queries."""
        self.schema = schema
        self.console.print(f"[green]Schema loaded ({len(schema)} characters)[/green]")

    def load_schema_from_file(self, filepath: str):
        """Load schema from file."""
        with open(filepath, "r") as f:
            self.schema = f.read()
        self.console.print(f"[green]Schema loaded from {filepath}[/green]")

    def generate(self, question: str, **kwargs) -> dict:
        """Generate SQL from question."""
        result = self.engine.generate_sql(question=question, schema=self.schema, **kwargs)
        return result

    def display_result(self, result: dict):
        """Display generation result."""
        # Display SQL
        sql_syntax = Syntax(result["sql"], "sql", theme="monokai", line_numbers=True)

        self.console.print(Panel(sql_syntax, title="Generated SQL", border_style="blue"))

        # Display metadata
        valid_status = "[green]✓ Valid[/green]" if result["valid"] else "[red]✗ Invalid[/red]"
        self.console.print(f"\nStatus: {valid_status}")

        if result["metadata"]:
            self.console.print("\nMetadata:")
            for key, value in result["metadata"].items():
                self.console.print(f"  {key}: {value}")

    def run(self):
        """Run interactive CLI."""
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

    def show_help(self):
        """Show help message."""
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
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Text-to-SQL Interactive CLI")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument(
        "--base-model", type=str, default=None, help="Base model name (for PEFT models)"
    )
    parser.add_argument("--schema", type=str, default=None, help="Path to schema file")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization")

    args = parser.parse_args()

    # Initialize engine
    console = Console()
    console.print("[cyan]Loading model...[/cyan]")

    engine = SQLInferenceEngine(
        model_path=args.model_path, base_model_name=args.base_model, load_in_4bit=args.load_in_4bit
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
