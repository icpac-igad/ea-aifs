# GPU Memory Profiling for AIFS ENS v1

This document describes how to profile GPU memory usage during AIFS ensemble inference using PyTorch's built-in memory profiling tools.

## Overview

The AIFS-ENS model requires approximately **50GB VRAM** at full precision (FP32). Understanding memory allocation patterns is essential for:

1. Identifying optimization opportunities
2. Debugging out-of-memory (OOM) errors
3. Comparing baseline vs. optimized configurations
4. Planning infrastructure requirements

## Profiling Tools

### 1. PyTorch Memory Snapshot (`pytorch_profile.py`)

The `pytorch_profile.py` script profiles GPU memory usage during inference and generates:

- **Memory snapshot** (`.pickle`): Full allocation history for visualization
- **Per-step CSV log**: Detailed memory statistics per inference step
- **Summary report**: Human-readable profiling results

### 2. PyTorch Memory Viz

Upload the memory snapshot to [pytorch.org/memory_viz](https://pytorch.org/memory_viz) for interactive visualization:

- Allocation timeline
- Memory fragmentation analysis
- Stack traces for allocations
- Tensor size breakdown

### 3. Anemoi Benchmark Profiler (Optional)

For full anemoi-training configs with Hydra, memory profiling can be enabled via YAML configuration.

## Quick Start

### Basic Profiling (Single Member, 72h Forecast)

```bash
# Profile with default settings
python pytorch_profile.py

# Output files in /scratch/profile_outputs/:
#   - aifs_ens_memory_snapshot.pickle
#   - aifs_ens_gpu_mem_per_member.csv
#   - profile_summary.txt
```

### Extended Profiling Options

```bash
# Profile multiple members
python pytorch_profile.py --members 1 2 3

# Longer forecast lead time
python pytorch_profile.py --members 1 --lead-time 144

# Custom output directory
python pytorch_profile.py --output-dir /scratch/my_profile

# Specify forecast date
python pytorch_profile.py --date 20251127

# Use live data download instead of pickle files
python pytorch_profile.py --no-pickle
```

## Understanding the Output

### Memory Snapshot (`.pickle`)

The snapshot file contains detailed allocation history that can be visualized at [pytorch.org/memory_viz](https://pytorch.org/memory_viz):

1. Navigate to https://pytorch.org/memory_viz
2. Drag and drop `aifs_ens_memory_snapshot.pickle`
3. Explore:
   - **Timeline**: See when allocations occur during inference
   - **Memory blocks**: Identify largest tensors
   - **Fragmentation**: Understand memory layout issues
   - **Stack traces**: Trace allocations to specific code

### Per-Step CSV Log

The `aifs_ens_gpu_mem_per_member.csv` file contains:

| Column | Description |
|--------|-------------|
| `timestamp` | UTC timestamp of measurement |
| `member` | Ensemble member number |
| `step` | Inference step (or "init"/"final") |
| `allocated_GB` | Peak allocated memory |
| `reserved_GB` | Total reserved memory by PyTorch |
| `current_GB` | Currently allocated memory |

Example:
```csv
timestamp,member,step,allocated_GB,reserved_GB,current_GB
2025-11-27T10:00:00,1,init,2.500,4.000,2.500
2025-11-27T10:00:30,1,1,45.200,48.000,45.200
2025-11-27T10:01:00,1,2,48.500,50.000,48.500
...
```

### Summary Report

The `profile_summary.txt` provides:

- GPU device information
- Configuration used
- Per-member peak memory statistics
- Aggregate statistics across all members
- GPU utilization percentage

## PyTorch Memory Profiling API

### Key Functions Used

```python
import torch

# Start recording allocation history
torch.cuda.memory._record_memory_history(max_entries=100000)

# Reset peak stats before a run
torch.cuda.reset_peak_memory_stats()

# Get memory statistics
allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
reserved = torch.cuda.max_memory_reserved() / 1024**3    # GB
current = torch.cuda.memory_allocated() / 1024**3        # GB

# Dump snapshot for visualization
torch.cuda.memory._dump_snapshot("snapshot.pickle")

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)

# Clear cache (helps reduce fragmentation)
torch.cuda.empty_cache()
```

### Integration Points in `pytorch_profile.py`

1. **Before model loading**: Initialize recording
2. **Per inference step**: Log memory stats
3. **After member completion**: Reset peak stats
4. **After all members**: Dump snapshot

## Anemoi Benchmark Profiler Configuration

When using the full anemoi-training framework with Hydra configs, memory profiling can be enabled in the YAML configuration:

```yaml
# In your config.yaml
diagnostics:
  benchmark_profiler:
    memory:
      enabled: True
      steps: 5        # Number of steps to profile
      warmup: 2       # Warmup steps before profiling
    snapshot:
      enabled: True
      steps: 4        # Steps to include in snapshot
      warmup: 0
```

This generates profiler outputs in:
```
$OUTPUT_DIR/profiler/<run_id>/memory_snapshot.pickle
```

**Note**: The anemoi config-based profiler is for full training/inference configs. For SimpleRunner-based inference (as in this project), use `pytorch_profile.py` instead.

## Comparing FP32 vs FP16

### Baseline Profiling (FP32)

```bash
# Run FP32 baseline profile (requires ~50GB VRAM)
python pytorch_profile.py --members 1 --lead-time 72
```

### FP16 Optimized Profiling (`pytorch_profile_fp16.py`)

A dedicated script is provided for testing 24GB GPU compatibility:

```bash
# Quick test with default settings (FP16 + 16 chunks)
python pytorch_profile_fp16.py

# Test with more aggressive chunking
python pytorch_profile_fp16.py --chunks 32

# Test longer forecast
python pytorch_profile_fp16.py --members 1 --lead-time 144

# Custom memory threshold
python pytorch_profile_fp16.py --threshold 22.0
```

**Output files** (in `/scratch/profile_fp16_outputs/`):
- `aifs_ens_fp16_memory_snapshot.pickle` - Memory snapshot
- `aifs_ens_fp16_gpu_mem.csv` - Per-step memory log with pass/fail status
- `fp16_profile_summary.txt` - Summary with PASS/FAIL verdict

### FP16 Script Features

The `pytorch_profile_fp16.py` script includes:

1. **24GB Threshold Checking**: Each step is checked against 23GB limit (1GB safety margin)
2. **Pass/Fail Verdict**: Clear output indicating if inference fits within 24GB
3. **OOM Handling**: Graceful handling of out-of-memory errors with recommendations
4. **Exit Codes**: Returns 0 on pass, 1 on fail (useful for CI/CD)

### Example Output

```
============================================================
TEST PASSED: FP16 inference fits within 24GB VRAM
============================================================
  Peak memory: 21.45 GB
  Safety margin: 1.55 GB

  Safe for: A10G, RTX 4090, RTX 3090
```

### Expected Results Comparison

| Configuration | Peak Memory | Notes |
|---------------|-------------|-------|
| FP32 baseline | ~50 GB | Full precision, A100-80GB required |
| FP16 only | ~25-30 GB | Half precision weights + activations |
| FP16 + 16 chunks | ~20-24 GB | Fits on A10G/RTX 4090 |
| FP16 + 32 chunks | ~18-22 GB | More aggressive chunking |

### Recommended Testing Workflow

1. **Baseline (A100-80GB)**:
   ```bash
   python pytorch_profile.py --members 1 --lead-time 72
   ```

2. **FP16 Test (24GB target)**:
   ```bash
   python pytorch_profile_fp16.py --chunks 16
   ```

3. **If test fails, increase chunks**:
   ```bash
   python pytorch_profile_fp16.py --chunks 32
   ```

4. **Compare snapshots** at https://pytorch.org/memory_viz

## Troubleshooting

### Out of Memory During Profiling

The profiling script uses FP32 mode intentionally to measure baseline memory. If you encounter OOM:

1. Use a GPU with at least 50GB VRAM (A100-80GB recommended)
2. Or modify to use FP16 for reduced-memory profiling
3. Reduce `--lead-time` for shorter forecasts

### Empty or Corrupted Snapshot

If the snapshot file is empty or won't load:

1. Ensure the script completed without errors
2. Check that `torch.cuda.memory._dump_snapshot()` was called
3. Try with `max_entries=1000000` for longer runs

### Memory Viz Won't Load File

If pytorch.org/memory_viz doesn't load the file:

1. Check file size (should be several MB)
2. Ensure the file was saved with `.pickle` extension
3. Try a different browser

## Best Practices

1. **Profile before optimizing**: Always establish a baseline first
2. **Use short forecasts**: 72h is sufficient for memory profiling
3. **Profile single member**: Memory patterns are consistent across members
4. **Compare configurations**: Profile FP32, FP16, and chunked versions
5. **Check fragmentation**: High reserved vs. allocated indicates fragmentation

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Memory Viz Tool](https://pytorch.org/memory_viz)
- [HuggingFace AIFS Discussion #17](https://huggingface.co/ecmwf/aifs-ens-1.0/discussions/17)
- [Anemoi Inference Documentation](https://anemoi-inference.readthedocs.io/)
