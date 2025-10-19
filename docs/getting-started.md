# Getting Started

## Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 40GB+ GPU memory (NVIDIA A100 recommended)
- 100GB+ disk space for models and data

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/chrisjcc/text-to-sql-refine-tuning.git
cd text-to-sql-refine-tuning
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Flash Attention (Optional but Recommended)
```bash
pip install flash-attn --no-build-isolation
```

### 5. Setup Environment Variables

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your tokens:
```
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

## Quick Start

### 1. Prepare Dataset
```bash
python scripts/prepare_data.py
```

This will:
- Download b-mc2/sql-create-context dataset
- Preprocess and clean samples
- Create train/validation splits
- Save processed data

### 2. Test Model Loading
```bash
python scripts/test_model.py
```

### 3. Test Environment
```bash
python scripts/test_environment.py
```

### 4. Run Training (Quick Test)
```bash
python scripts/train.py \
  dataset.split.train="train[:100]" \
  training.max_steps=10
```

### 5. Run Full Training
```bash
python scripts/train.py
```

### 6. Run Inference
```bash
python scripts/inference.py \
  inference.model_path=./outputs/final_model
```

## Next Steps

- Read the [Training Guide](training.md) for detailed training instructions
- Check [Configuration Guide](configuration.md) for customization options
- See [Examples](examples/) for common use cases

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.
