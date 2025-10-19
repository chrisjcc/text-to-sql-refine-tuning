# Troubleshooting Guide

## Common Issues

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `training.per_device_train_batch_size=1`
2. Increase gradient accumulation: `training.gradient_accumulation_steps=16`
3. Enable gradient checkpointing (should be on by default)
4. Reduce sequence length: `dataset.preprocessing.max_length=256`
5. Use QLoRA (should be enabled by default)

### Model Loading Fails

**Problem:** Model fails to load or takes too long

**Solutions:**
1. Check HuggingFace token: `echo $HF_TOKEN`
2. Clear cache: `rm -rf ./cache/*`
3. Check disk space: `df -h`
4. Try loading base model first: `python scripts/test_model.py`

### Dataset Loading Errors

**Problem:** Dataset download fails or is corrupted

**Solutions:**
1. Clear dataset cache: `rm -rf ./data_cache/*`
2. Check internet connection
3. Try manual download from HuggingFace Hub
4. Use streaming mode: `dataset.streaming=true`

### Training Not Improving

**Problem:** Metrics not improving or training unstable

**Solutions:**
1. Check learning rate (try 1e-6 to 1e-5)
2. Verify rubric scores are reasonable
3. Increase num_generations for better gradient estimates
4. Monitor KL divergence (should be < 1.0)
5. Check data quality and preprocessing

### Inference is Slow

**Problem:** SQL generation takes too long

**Solutions:**
1. Merge PEFT adapters (done automatically in inference engine)
2. Use lower temperature: `temperature=0.1`
3. Disable sampling: `do_sample=false`
4. Reduce max_new_tokens
5. Use batch inference
6. Consider 4-bit quantization: `load_in_4bit=true`

### API Server Issues

**Problem:** API fails to start or crashes

**Solutions:**
1. Check port availability: `lsof -i :8000`
2. Verify model path is correct
3. Check memory available
4. Use 4-bit quantization for lower memory
5. Check logs: `python -m src.inference.api --model-path <path> 2>&1 | tee api.log`

### WandB Not Logging

**Problem:** Metrics not appearing in WandB

**Solutions:**
1. Check API key: `echo $WANDB_API_KEY`
2. Login manually: `wandb login`
3. Verify wandb.enabled=true in config
4. Check internet connection
5. Try offline mode: `wandb offline`

## Performance Optimization

### Training Speed

- Enable Flash Attention 2
- Use bf16 instead of fp16
- Increase batch size (if memory allows)
- Use multiple GPUs with DeepSpeed
- Reduce evaluation frequency

### Memory Usage

- Use QLoRA (enabled by default)
- Reduce batch size
- Increase gradient accumulation
- Enable gradient checkpointing
- Reduce sequence length

### Generation Quality

- Tune temperature (0.1-0.7)
- Adjust top_p (0.9-0.95)
- Try beam search (num_beams=3-5)
- Experiment with prompts
- Fine-tune longer (more epochs)

## Getting Help

1. Check [GitHub Issues](https://github.com/chrisjcc/text-to-sql-refine-tuning/issues)
2. Review [documentation](index.md)
3. Check Verifiers documentation
4. Open a new issue with:
   - Error message and full stack trace
   - Configuration used
   - System info (GPU, CUDA version, etc.)
   - Steps to reproduce
