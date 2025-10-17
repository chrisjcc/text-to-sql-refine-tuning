# Data Preparation Guide

This guide explains how to prepare the SQL dataset for GRPO training using the `prepare_data.py` script.

## Basic Usage

### Full Dataset

Load and process the complete dataset:

```bash
python scripts/prepare_data.py
```

This will:
1. Load the full `b-mc2/sql-create-context` dataset
2. Create train/validation/test splits (80/10/10)
3. Clean and preprocess all samples
4. Filter invalid samples
5. Compute statistics
6. Save to `./data_cache/processed/`

### Limited Dataset (for Testing)

To test the pipeline with a subset of data, use the `limit` parameter:

```bash
# Load only first 100 training samples
python scripts/prepare_data.py dataset.limit.train=100

# Load first 500 training samples and 50 validation samples
python scripts/prepare_data.py dataset.limit.train=500 dataset.limit.validation=50
```

**Note**: The `limit` parameter accepts an integer (e.g., `100`) to load the first N samples. Do NOT use HuggingFace slicing syntax like `train[:100]` as it contains special characters that conflict with Hydra's configuration parser.

## Configuration Options

### Dataset Parameters

Override dataset configuration:

```bash
# Use different dataset
python scripts/prepare_data.py dataset.name="your-org/your-dataset"

# Change cache directory
python scripts/prepare_data.py dataset.cache_dir="./custom_cache"

# Adjust split sizes
python scripts/prepare_data.py \
  dataset.create_splits.train_size=0.7 \
  dataset.create_splits.val_size=0.15 \
  dataset.create_splits.test_size=0.15
```

### Preprocessing Parameters

Adjust preprocessing settings:

```bash
# Change max lengths
python scripts/prepare_data.py \
  dataset.preprocessing.max_question_length=256 \
  dataset.preprocessing.max_sql_length=256

# Disable SQL normalization
python scripts/prepare_data.py dataset.preprocessing.normalize_sql=false

# Change number of parallel workers
python scripts/prepare_data.py dataset.num_workers=8
```

### Multiple Overrides

Combine multiple configuration overrides:

```bash
python scripts/prepare_data.py \
  dataset.limit.train=1000 \
  dataset.preprocessing.max_sql_length=256 \
  dataset.num_workers=8 \
  dataset.cache_dir="./test_cache"
```

## Output

The script produces:

1. **Processed Dataset**: Saved to `{cache_dir}/processed/`
   - `dataset_dict.json`: Metadata
   - `train/`: Training split
   - `validation/`: Validation split
   - `test/`: Test split (if enabled)

2. **Logs**: Located in `logs/data_preparation.log`
   - Loading progress
   - Preprocessing statistics
   - Validation results
   - Final dataset statistics

3. **Statistics**: Printed to console and log file
   - Sample counts per split
   - Complexity distribution
   - Length statistics (question, SQL, schema)
   - SQL keyword distribution
   - Validation rates

## Common Issues

### Hydra Configuration Errors

**Problem**: `mismatched input '[' expecting <EOF>`

**Cause**: Using HuggingFace slicing syntax with special characters

**Solution**: Use the `limit` parameter instead:
```bash
# ❌ Wrong (contains special characters)
python scripts/prepare_data.py dataset.split.train="train[:100]"

# ✅ Correct (use limit parameter)
python scripts/prepare_data.py dataset.limit.train=100
```

### Memory Issues

**Problem**: Out of memory during preprocessing

**Solution**: Reduce the number of workers or use dataset limiting:
```bash
# Reduce workers
python scripts/prepare_data.py dataset.num_workers=2

# Process smaller subset first
python scripts/prepare_data.py dataset.limit.train=1000
```

### Authentication Errors

**Problem**: `Error loading dataset: 401 Unauthorized`

**Solution**: Set your HuggingFace token:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
# or
huggingface-cli login
```

## Data Quality Analysis

After preparation, analyze the data quality:

```bash
python scripts/analyze_data.py
```

This generates a comprehensive report including:
- Complexity distribution charts
- SQL pattern analysis
- Length distribution statistics
- Quality issue detection
- Example queries for each complexity level

## Next Steps

After preparing the data:

1. Review the statistics and quality report
2. Adjust preprocessing parameters if needed
3. Use the processed dataset for GRPO training:
   ```bash
   python scripts/train.py dataset.processed_path="./data_cache/processed"
   ```

## Configuration Reference

For complete configuration options, see:
- `config/dataset/dataset.yaml`: Dataset configuration
- `config/config.yaml`: Main configuration file

## Tips

- **Start small**: Test with limited data first (`dataset.limit.train=100`)
- **Monitor logs**: Check `logs/data_preparation.log` for detailed information
- **Cache wisely**: Processed datasets are cached, reuse them when possible
- **Validate quality**: Always run `analyze_data.py` after preparation
