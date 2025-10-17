# Text-to-SQL Fine-Tuning with GRPO

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)

A comprehensive framework for fine-tuning large language models to convert natural language queries into SQL using **Group Relative Policy Optimization (GRPO)**. This project leverages the TRL library's GRPO Trainer and a verifier-based reward system to achieve robust, accurate text-to-SQL generation.

## Overview

This project implements a state-of-the-art approach to natural language-to-SQL conversion through fine-tuning Llama-3.1-8B and Llama-3.1-8B-Instruct models. By employing GRPO—a reinforcement learning technique that optimizes policy using group-based relative rewards—the framework learns to generate syntactically valid and semantically correct SQL queries.

### What is GRPO?

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that improves upon traditional policy gradient methods by computing relative advantages within groups of samples. This approach:
- Reduces variance in gradient estimates
- Improves sample efficiency during training
- Enables more stable convergence for complex generation tasks

The framework integrates TRL's (Transformer Reinforcement Learning) GRPO Trainer with a sophisticated verifier-based reward system to guide the model toward generating high-quality SQL outputs.

## Key Features

- **Advanced Model Fine-Tuning**: Fine-tune Llama-3.1-8B and Llama-3.1-8B-Instruct models for text-to-SQL tasks
- **Verifier-Based Rewards**: Multi-faceted reward system evaluating SQL validity, syntax correctness, and keyword usage
- **QLoRA Optimization**: Efficient training on NVIDIA A100 GPUs using Quantized Low-Rank Adaptation
- **Hydra Configuration**: Flexible, composable configuration management for experiments
- **WandB Integration**: Comprehensive experiment tracking and visualization
- **Protocol-First Design**: Built on the Verifiers framework for extensibility and reusability

## Architecture Overview

### GRPO Training Pipeline

The training pipeline consists of several interconnected components:

```
Natural Language Query → Model → SQL Generation → Verifiers → Rewards → Policy Update
                           ↑                                                    ↓
                           └────────────────── Gradient Flow ──────────────────┘
```

1. **Input Processing**: Natural language queries are tokenized and fed to the model
2. **Generation**: Model generates SQL query candidates
3. **Verification**: Multiple verifiers evaluate the generated SQL:
   - Syntax validation
   - Semantic correctness
   - Keyword presence
   - Execution validity
4. **Reward Computation**: Verifiers provide scalar rewards based on quality metrics
5. **Policy Optimization**: GRPO algorithm updates model parameters to maximize expected rewards

### Verifiers Framework Integration

The project utilizes the [Verifiers](https://github.com/your-org/verifiers) framework, which provides:

- **Environments**: Standardized interfaces for single-turn and multi-turn interactions
- **Rubrics**: Composable evaluation metrics for SQL quality assessment
- **Parsers**: Structured validation of SQL syntax and format

This protocol-first design enables easy extension with custom verifiers and evaluation metrics.

### Dataset

Training uses the [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset from HuggingFace, which contains:
- Natural language questions
- SQL context (table schemas)
- Ground truth SQL queries
- Database execution contexts

## Project Structure

```
text-to-sql-fine-tuning/
├── configs/                      # Hydra configuration files
│   ├── config.yaml              # Main composite configuration
│   ├── model/                   # Model configurations
│   ├── training/                # Training hyperparameters
│   ├── verifier/                # Verifier configurations
│   └── environment/             # Environment settings
├── src/
│   ├── environments/            # Custom environment implementations
│   ├── verifiers/               # SQL verification logic
│   ├── rubrics/                 # Evaluation rubrics
│   ├── parsers/                 # SQL parsing utilities
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # Model architectures and wrappers
│   └── training/                # Training loops and utilities
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference script
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                       # Unit and integration tests
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Configuration Management

This project uses **Hydra** for flexible, composable configuration:

- **Composite Configs**: Combine multiple YAML files for modular experiment setup
- **Override System**: Easily modify parameters via command line
- **Config Groups**: Organize related settings (model, training, verifier)
- **Environment Variables**: Sensitive data (API keys, tokens) stored in `.env`

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **CUDA**: 12.0+ for GPU acceleration
- **GPU**: NVIDIA A100 (40GB or 80GB) recommended
  - Minimum: 24GB VRAM for QLoRA training
- **Storage**: 100GB+ free space for models and datasets

### Cloud Platform

This project is optimized for [Lambda Cloud](https://lambdalabs.com/) A100 instances:
- Pre-configured CUDA environments
- High-performance NVMe storage
- Cost-effective GPU access
- Easy scaling for multi-GPU training

### Accounts & Authentication

- **HuggingFace Account**: Required for dataset and model access
- **HuggingFace Token**: Set in `.env` file for authentication
- **WandB Account** (optional): For experiment tracking and visualization

## Quick Start

> **Note**: Detailed installation and setup instructions will be added in future updates.

### Installation (Placeholder)

```bash
# Clone the repository
git clone https://github.com/chrisjcc/text-to-sql-fine-tuning.git
cd text-to-sql-fine-tuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and add your HuggingFace token and other credentials
```

### Configuration Setup (Placeholder)

```bash
# Review and customize configuration files in configs/
# Main config: configs/config.yaml
# Model config: configs/model/llama-3.1-8b.yaml
# Training config: configs/training/grpo_default.yaml
```

### Training Example (Placeholder)

```bash
# Basic training with default configuration
python scripts/train.py

# Override specific parameters
python scripts/train.py training.batch_size=16 model.lora_r=64

# Multi-GPU training with DeepSpeed
python scripts/train.py training.accelerator=deepspeed training.num_gpus=4
```

### Inference Example (Placeholder)

```bash
# Run inference on a single query
python scripts/inference.py \
  --checkpoint ./outputs/checkpoint-best \
  --query "Show me all customers who made purchases last month"

# Batch inference from file
python scripts/inference.py \
  --checkpoint ./outputs/checkpoint-best \
  --input queries.txt \
  --output results.json
```

## Configuration Management

### Hydra Composite Configuration

The project uses Hydra's powerful configuration system to manage complex experiment setups:

```yaml
# configs/config.yaml
defaults:
  - model: llama-3.1-8b-instruct
  - training: grpo_default
  - verifier: sql_verifier
  - environment: single_turn
  - _self_

# Global settings
project_name: text-to-sql-grpo
seed: 42
output_dir: ./outputs
```

### Config Directory Structure

```
configs/
├── config.yaml                   # Main configuration
├── model/
│   ├── llama-3.1-8b.yaml        # Base model config
│   └── llama-3.1-8b-instruct.yaml
├── training/
│   ├── grpo_default.yaml        # GRPO hyperparameters
│   └── grpo_lora.yaml           # QLoRA-specific settings
├── verifier/
│   ├── sql_verifier.yaml        # SQL verification config
│   └── rubrics.yaml             # Evaluation rubrics
└── environment/
    ├── single_turn.yaml         # Single-turn environment
    └── multi_turn.yaml          # Multi-turn environment
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# HuggingFace Authentication
HUGGINGFACE_TOKEN=your_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=text-to-sql-grpo

# Training Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3
OUTPUT_DIR=./outputs
CACHE_DIR=./cache
```

## Key Components

### Environment Types

The framework supports different environment configurations for SQL generation:

- **SingleTurnEnv**: One-shot query generation (question → SQL)
  - Simpler, faster training
  - Suitable for most text-to-SQL tasks
  - Used in this project by default

- **MultiTurnEnv**: Interactive query refinement (future consideration)
  - Iterative query improvement
  - Handles clarification and corrections
  - More complex but potentially more robust

### Rubrics

Evaluation rubrics define how SQL quality is assessed:

- **Valid SQL Percentage**: Measures syntactic correctness
  - Parses SQL using standard libraries
  - Checks for common syntax errors
  - Validates against SQL standards

- **SQL Keyword Detection**: Ensures proper SQL structure
  - Identifies required keywords (SELECT, FROM, WHERE, etc.)
  - Validates keyword ordering and context
  - Penalizes missing or incorrect keywords

- **Semantic Correctness** (planned): Evaluates query meaning
  - Compares against ground truth
  - Checks table/column references
  - Validates join conditions

### Parsers

SQL parsers validate and extract structure from generated queries:

- **Syntax Validation**: Uses `sqlparse` library for parsing
- **Format Normalization**: Standardizes SQL formatting
- **Error Detection**: Identifies specific syntax issues
- **Structure Extraction**: Extracts tables, columns, and operations

## Training Infrastructure

### Lambda Cloud Platform

This project is designed to run on Lambda Cloud GPU instances:

- **Instance Types**:
  - 1x A100 (40GB): Single-GPU training with QLoRA
  - 1x A100 (80GB): Larger batch sizes, full fine-tuning option
  - 4x A100 (40GB): Multi-GPU distributed training

- **Advantages**:
  - Pre-installed CUDA toolkit and drivers
  - Optimized PyTorch and transformers libraries
  - High-bandwidth NVMe storage
  - Jupyter Lab access for development

### NVIDIA A100 GPU Specifications

- **Architecture**: Ampere
- **Memory**: 40GB or 80GB HBM2e
- **Tensor Cores**: 3rd generation (FP64, FP32, FP16, BF16, INT8)
- **Memory Bandwidth**: 1.6-2.0 TB/s
- **Ideal For**: Large language model fine-tuning with QLoRA

### DeepSpeed and Accelerate Support

The framework supports distributed training through:

- **Hugging Face Accelerate**: Simple multi-GPU training
- **DeepSpeed ZeRO**: Memory-efficient distributed training
  - ZeRO Stage 2: Optimizer state partitioning
  - ZeRO Stage 3: Full parameter partitioning
  - Mixed precision training (FP16/BF16)

## References

### Documentation & Libraries

- **Verifiers Framework**: [Documentation](https://github.com/your-org/verifiers) - Protocol-first evaluation framework
- **TRL (Transformer Reinforcement Learning)**: [GitHub](https://github.com/huggingface/trl) | [Docs](https://huggingface.co/docs/trl)
- **Hydra Configuration**: [Documentation](https://hydra.cc/) - Flexible configuration management
- **Weights & Biases**: [Documentation](https://docs.wandb.ai/) - Experiment tracking

### Research Papers

- **GRPO**: Group Relative Policy Optimization (reference to be added)
- **QLoRA**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Llama 3.1**: [Meta Llama 3.1 Technical Report](https://arxiv.org/abs/2407.21783)
- **Text-to-SQL**: [Recent advances in text-to-SQL](https://arxiv.org/abs/2208.13629)

### Related Resources

- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **SQL Dataset**: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
- **Lambda Cloud**: [Platform](https://lambdalabs.com/) - GPU cloud provider

## Roadmap

This project is under active development. Planned milestones:

- [x] **Project Setup**
  - [x] Repository structure
  - [x] Configuration system design
  - [x] Documentation framework

- [ ] **Environment Setup**
  - [ ] Implement SingleTurnEnv for SQL generation
  - [ ] Create data loaders for sql-create-context dataset
  - [ ] Build tokenization and preprocessing pipeline

- [ ] **Training Pipeline**
  - [ ] Integrate TRL GRPO Trainer
  - [ ] Implement QLoRA configuration
  - [ ] Add DeepSpeed support for multi-GPU training
  - [ ] Create training monitoring and logging

- [ ] **Evaluation Framework**
  - [ ] Implement SQL syntax verifier
  - [ ] Add keyword detection rubric
  - [ ] Create semantic correctness evaluation
  - [ ] Build execution-based validation

- [ ] **Model Deployment**
  - [ ] Fine-tune Llama-3.1-8B baseline
  - [ ] Experiment with different reward functions
  - [ ] Optimize hyperparameters
  - [ ] Deploy to HuggingFace Hub

- [ ] **Documentation & Examples**
  - [ ] Complete installation guide
  - [ ] Add training tutorials
  - [ ] Create inference examples
  - [ ] Write contribution guidelines

## Contributing

We welcome contributions to improve this project! Areas where you can help:

- Implementing new verifiers and rubrics
- Adding support for additional SQL dialects
- Improving documentation and examples
- Optimizing training performance
- Creating evaluation benchmarks

### Guidelines (Detailed guidelines to be added)

- Follow PEP 8 style guide for Python code
- Add unit tests for new functionality
- Update documentation for user-facing changes
- Use meaningful commit messages

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### Apache 2.0 Summary

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Patent use allowed
- ✅ Private use allowed
- ⚠️ Must include license and copyright notice
- ⚠️ Must state changes made to the code
- ❌ No trademark rights granted
- ❌ No liability or warranty

## Acknowledgments

- **Meta AI** for the Llama 3.1 models
- **HuggingFace** for the TRL library and transformers ecosystem
- **Lambda Labs** for accessible GPU infrastructure
- **Verifiers Framework** for protocol-first evaluation design

---

**Note**: This project is in active development. Features, APIs, and documentation are subject to change. For questions, issues, or contributions, please open an issue or pull request on GitHub.

---

*Built with the verifier-based approach for reliable, high-quality text-to-SQL generation*
