# Quick Start Guide - ECMWF Parquet to PKL

## 🚀 One-Command Setup

```bash
# 1. Start Coiled Notebook
coiled notebook start --name p2-aifs-etl-20251016 \
  --vm-type n2-standard-2 --software aifs-etl-v2 \
  --workspace=gcp-sewaa-nka

# 2. Install Dependencies
micromamba install -c conda-forge fastparquet pyarrow obstore kerchunk cfgrib

# 3. Done! Ready to process
```

---

## 📋 Three-Step Workflow

### Step 1: Create Parquet Files (One-time per forecast)

```bash
python ecmwf_ensemble_par_creator_efficient.py
```

**Output**: 51 parquet files (control + ens_01 to ens_50)
**Time**: ~15 minutes
**Efficiency**: 51x faster than old approach

### Step 2: Extract Variables

```bash
python extract_hybrid_refs.py
```

**Input**: Parquet file (1182 references)
**Output**: PKL file (7.92 MB for t2m)
**Time**: ~4 seconds with fsspec, ~1.2 seconds with obstore

### Step 3: Use in AI Model

```python
import pickle

with open('t2m_instant_heightAboveGround_t2m.pkl', 'rb') as f:
    data = pickle.load(f)

# data.shape = (1, 2, 721, 1440)  # time, step, lat, lon
# data.dtype = float32
```

---

## ⚡ Performance Cheat Sheet

| Operation | Old Way | New Way | Speedup |
|-----------|---------|---------|---------|
| Scan GRIB | 51 scans | 1 scan | **51x** |
| Total scan time | 11.5 hrs | 13.6 min | **51x** |
| S3 fetch (fsspec) | 50ms/chunk | 50ms/chunk | 1x |
| S3 fetch (obstore) | - | 15ms/chunk | **3.3x** |
| Process 51 members | ~3 min | ~54s | **3.3x** |

---

## 🐛 Quick Troubleshooting

### Error: "obstore not found"
```bash
micromamba install -c conda-forge obstore
```

### Error: "Region redirect"
Already fixed! Script auto-detects EU region for ECMWF buckets.

### Error: "GRIB2 decode failed"
```bash
micromamba install -c conda-forge cfgrib eccodes
```

### Validation warnings (non-blocking)
These are expected - parquet files are created correctly!

---

## 📊 Quick Stats

- **Ensemble members**: 51 (control + 50 members)
- **Variables per member**: 41 (t2m, tp, msl, etc.)
- **References per member**: 1,182
- **S3 chunks per member**: 70
- **Time saved per year**: ~49 hours (with obstore)

---

## 🔗 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `ecmwf_ensemble_par_creator_efficient.py` | Creates parquet | ✅ Working |
| `extract_hybrid_refs.py` | Extracts to PKL | ✅ Working (fsspec) |
| `README.md` | Full documentation | 📖 Complete |

---

**Ready to process? Start with Step 1!**
