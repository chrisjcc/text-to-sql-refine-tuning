"""Setup configuration for text-to-sql-fine-tuning package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="text-to-sql-fine-tuning",
    version="0.1.0",
    author="chrisjcc",
    description="Fine-tuning LLMs for text-to-SQL using GRPO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chrisjcc/text-to-sql-fine-tuning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "trl>=0.24.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "verifiers>=0.1.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "python-dotenv>=1.0.0",
        "wandb>=0.16.0",
        "sqlparse>=0.4.4",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="text-to-sql, fine-tuning, llm, grpo, reinforcement-learning",
    project_urls={
        "Bug Reports": "https://github.com/chrisjcc/text-to-sql-fine-tuning/issues",
        "Source": "https://github.com/chrisjcc/text-to-sql-fine-tuning",
    },
)
