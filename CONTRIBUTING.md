# Contributing Guidelines

Thank you for your interest in contributing to text-to-sql-refine-tuning!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/text-to-sql-refine-tuning.git
   cd text-to-sql-refine-tuning
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   make test
   ```

4. Run linting:
   ```bash
   make lint
   ```

5. Format code:
   ```bash
   make format
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use Black for formatting (line length: 100)
- Use type hints where possible
- Write docstrings for all public functions/classes
- Keep functions focused and small

## Testing

- Write unit tests for new features
- Maintain test coverage >80%
- Use pytest for testing
- Mock external dependencies

## Documentation

- Update documentation for new features
- Add examples for new functionality
- Keep README.md up to date
- Document configuration options

## Pull Request Guidelines

- Provide a clear description of changes
- Reference related issues
- Include tests for new features
- Update documentation
- Ensure CI passes
- Keep PRs focused and atomic

## Issue Reporting

When reporting issues, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Relevant logs or error messages

## Feature Requests

- Describe the feature clearly
- Explain the use case
- Provide examples if possible
- Discuss implementation approach

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate openly

## Questions?

Feel free to open an issue for questions or discussions.
