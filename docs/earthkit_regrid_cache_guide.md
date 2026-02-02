# Earthkit-Regrid Cache Management Guide

## Overview

This document explains the earthkit-regrid caching mechanism, the SQLite database corruption issue encountered during GRIB-to-NetCDF processing, and the solution implemented to ensure reliable multi-member ensemble processing.

---

## Table of Contents

1. [What is Earthkit-Regrid?](#what-is-earthkit-regrid)
2. [The Regrid Matrix Files (.npz)](#the-regrid-matrix-files-npz)
3. [Cache Architecture](#cache-architecture)
4. [The SQLite Corruption Problem](#the-sqlite-corruption-problem)
5. [Solution: Proper Cache Management](#solution-proper-cache-management)
6. [Pre-caching Matrix Files](#pre-caching-matrix-files)
7. [Configuration Reference](#configuration-reference)
8. [Troubleshooting](#troubleshooting)

---

## What is Earthkit-Regrid?

Earthkit-regrid is an ECMWF library for interpolating meteorological data between different grid systems. In our pipeline, it converts AIFS ensemble forecasts from the native **N320 Gaussian grid** to a regular **1.5-degree latitude-longitude grid**.

```python
import earthkit.regrid as ekr

# Regrid from N320 to 1.5 degree regular grid
regridded_data = ekr.interpolate(
    field_list,
    in_grid={"grid": "N320"},
    out_grid={"grid": [1.5, 1.5]}
)
```

The interpolation requires a **transformation matrix** that defines how each point in the source grid maps to points in the target grid.

---

## The Regrid Matrix Files (.npz)

### What Are They?

The `.npz` files are **sparse matrix files** (NumPy compressed format) containing the interpolation weights. For our N320 → 1.5° transformation:

| Property | Value |
|----------|-------|
| Source Grid | N320 Gaussian (~542,080 points) |
| Target Grid | 1.5° regular (121 lat × 240 lon = 29,040 points) |
| Matrix Size | ~367 KB (compressed sparse) |
| File Hash | `1ec0ab77831d12c5058ea56f59c9cd19abe0420714511fba9a458d7aec2f1929` |

### Where Do They Come From?

The matrix files are downloaded from ECMWF's servers on first use:

```
https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/<hash>.npz
```

The hash is computed from the grid specifications, ensuring each unique transformation has its own cached matrix.

### Why Cache Them?

1. **Download time**: Even though the file is small (~367 KB), ECMWF servers can be slow
2. **Network reliability**: Downloads can fail or timeout
3. **Efficiency**: The same matrix is needed for every ensemble member

---

## Cache Architecture

Earthkit-regrid uses a **SQLite database** to manage its cache. The architecture consists of:

```
~/.cache/earthkit-regrid/
├── cache-2.db                    # SQLite database (cache metadata)
└── url-<sha256hash>.npz          # Cached matrix files
```

### The SQLite Database Schema

```sql
CREATE TABLE cache (
    path          TEXT PRIMARY KEY,   -- Full path to cached file
    owner         TEXT NOT NULL,      -- Cache owner (e.g., "url")
    args          TEXT NOT NULL,      -- JSON: URL and parameters
    creation_date TEXT NOT NULL,      -- When file was cached
    flags         INTEGER DEFAULT 0,
    owner_data    TEXT,               -- Additional metadata
    last_access   TEXT NOT NULL,      -- Last access timestamp
    type          TEXT,               -- "file" or "directory"
    parent        TEXT,               -- Parent cache entry
    replaced      TEXT,
    extra         TEXT,
    expires       INTEGER,
    accesses      INTEGER,            -- Access count
    size          INTEGER             -- File size in bytes
);
```

### Cache Filename Computation

The cached filename is computed using SHA-256:

```python
import hashlib
import json

def compute_cache_filename(url):
    owner = "url"
    extension = ".npz"

    m = hashlib.sha256()
    m.update(owner.encode("utf-8"))
    m.update(json.dumps({"url": url, "parts": None}, sort_keys=True).encode("utf-8"))
    m.update(json.dumps(None, sort_keys=True).encode("utf-8"))  # hash_extra
    m.update(json.dumps(extension, sort_keys=True).encode("utf-8"))

    return f"{owner.lower()}-{m.hexdigest()}{extension}"

# Example:
url = "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/1ec0ab77831d12c5058ea56f59c9cd19abe0420714511fba9a458d7aec2f1929.npz"
filename = compute_cache_filename(url)
# Result: url-1491fa2f50c9bb0f53d0ee0d71ee2d2fc709641acebedab69b7b6e74e85dba47.npz
```

### The Cache Singleton

Earthkit-regrid uses a **singleton pattern** for cache management:

```python
# In earthkit/regrid/utils/caching.py
CACHE = Cache()  # Global singleton - created once at module import
```

This singleton:
- Maintains a persistent SQLite connection
- Manages all cache operations
- Cannot be recreated within a Python process

---

## The SQLite Corruption Problem

### Symptoms

When processing multiple ensemble members, the script would:

1. Successfully process member 001
2. Fail on member 002+ with the error:

```
attempt to write a readonly database
Could not download matrix file=https://get.ecmwf.int/...
```

### Root Cause Analysis

The problem occurred due to improper cache cleanup between members:

```
Timeline of Failure:
───────────────────────────────────────────────────────────────────────

Member 001 Processing:
  ├─ CACHE singleton created with SQLite connection
  ├─ Matrix downloaded and cached successfully
  ├─ Processing completes
  └─ cleanup_earthkit_dirs() called
       └─ shutil.rmtree(~/.cache/earthkit-regrid/)  ← DELETES DIRECTORY
       └─ Directory recreated (empty)

Member 002 Processing:
  ├─ CACHE singleton still holds OLD SQLite connection
  │   └─ Connection points to deleted/recreated database
  ├─ Attempt to download matrix
  │   └─ SQLite write fails: "readonly database"
  └─ Processing fails
```

### Why "Readonly Database"?

When the cache directory is deleted and recreated:

1. The SQLite connection in the singleton becomes **stale**
2. The connection references a file that no longer exists (or is a new empty file)
3. SQLite operations fail with "readonly database" or "unable to open database file"
4. The singleton cannot be reset without restarting the Python process

### The Cascade Effect

Once the SQLite cache fails:
1. Matrix downloads fail (can't register in cache)
2. Regridding fails (no matrix available)
3. All subsequent members fail
4. Disk space may fill up with partial downloads

---

## Solution: Proper Cache Management

### Key Principles

1. **Never delete the regrid cache during processing** - The singleton connection must remain valid
2. **Use persistent "user" cache policy** - Matrix files are reused across all members
3. **Pre-cache matrix files when possible** - Avoid slow downloads during processing

### Configuration Before Import

Settings must be configured **before** importing earthkit.regrid:

```python
import os
from pathlib import Path

# Set environment variables FIRST
os.environ["TQDM_DISABLE"] = "1"  # Disable progress bars that can hang

# Configure regrid cache BEFORE importing earthkit.regrid
from earthkit.regrid.utils import caching as regrid_caching

regrid_caching.SETTINGS["cache-policy"] = "user"  # Persistent cache
regrid_caching.SETTINGS["user-cache-directory"] = str(Path.home() / ".cache/earthkit-regrid")
regrid_caching.SETTINGS["url-download-timeout"] = 300  # 5 minutes
regrid_caching.SETTINGS["matrix-memory-cache-policy"] = "lru"
regrid_caching.SETTINGS["maximum-matrix-memory-cache-size"] = 2 * 1024 * 1024 * 1024  # 2GB

# NOW import earthkit.regrid
import earthkit.regrid as ekr
```

### Safe Cleanup Function

```python
def cleanup_earthkit_dirs(verbose=True):
    """Clean up earthkit caches WITHOUT corrupting the regrid singleton."""

    # Safe to clean: earthkit-data cache and temp directories
    safe_to_clean = [
        Path("/scratch/notebook/.cache/earthkit-data"),
        Path("/scratch/notebook/earthkit-tmp"),
        Path("/scratch/notebook/tmp"),
    ]

    for dir_path in safe_to_clean:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Cleared: {dir_path}")

    # NEVER clean during processing:
    # - ~/.cache/earthkit-regrid/  (SQLite singleton corruption)

    # The regrid cache contains:
    # - cache-2.db (SQLite database)
    # - url-*.npz (matrix files - expensive to re-download)
```

---

## Pre-caching Matrix Files

To avoid download issues during processing, you can pre-cache the matrix files.

### Step 1: Identify Required Matrix

For N320 → 1.5° transformation:

```
URL: https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/1ec0ab77831d12c5058ea56f59c9cd19abe0420714511fba9a458d7aec2f1929.npz
Cache filename: url-1491fa2f50c9bb0f53d0ee0d71ee2d2fc709641acebedab69b7b6e74e85dba47.npz
```

### Step 2: Download the Matrix File

```bash
mkdir -p ~/.cache/earthkit-regrid

curl -L --connect-timeout 30 --max-time 300 \
  -o ~/.cache/earthkit-regrid/url-1491fa2f50c9bb0f53d0ee0d71ee2d2fc709641acebedab69b7b6e74e85dba47.npz \
  "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/1ec0ab77831d12c5058ea56f59c9cd19abe0420714511fba9a458d7aec2f1929.npz"
```

### Step 3: Register in SQLite Database

```python
import sqlite3
import json
import os
from datetime import datetime

def register_matrix_in_cache(matrix_url, cache_dir="~/.cache/earthkit-regrid"):
    """Register a pre-downloaded matrix file in the earthkit-regrid cache."""

    cache_dir = os.path.expanduser(cache_dir)
    db_path = os.path.join(cache_dir, "cache-2.db")

    # Compute the expected filename
    import hashlib
    owner = "url"
    extension = ".npz"

    m = hashlib.sha256()
    m.update(owner.encode("utf-8"))
    m.update(json.dumps({"url": matrix_url, "parts": None}, sort_keys=True).encode("utf-8"))
    m.update(json.dumps(None, sort_keys=True).encode("utf-8"))
    m.update(json.dumps(extension, sort_keys=True).encode("utf-8"))

    filename = f"{owner.lower()}-{m.hexdigest()}{extension}"
    file_path = os.path.join(cache_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Matrix file not found: {file_path}")

    file_size = os.path.getsize(file_path)

    # Connect to database
    conn = sqlite3.connect(db_path)

    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            path          TEXT PRIMARY KEY,
            owner         TEXT NOT NULL,
            args          TEXT NOT NULL,
            creation_date TEXT NOT NULL,
            flags         INTEGER DEFAULT 0,
            owner_data    TEXT,
            last_access   TEXT NOT NULL,
            type          TEXT,
            parent        TEXT,
            replaced      TEXT,
            extra         TEXT,
            expires       INTEGER,
            accesses      INTEGER,
            size          INTEGER
        )
    """)

    # Check if already registered
    cursor = conn.execute("SELECT path FROM cache WHERE path=?", (file_path,))
    if cursor.fetchone():
        print(f"Already registered: {filename}")
        conn.close()
        return

    # Register the file
    now = datetime.now().isoformat()
    args = json.dumps({"url": matrix_url, "parts": None})

    conn.execute("""
        INSERT INTO cache (path, owner, args, creation_date, last_access, type, accesses, size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_path, owner, args, now, now, "file", 1, file_size))

    conn.commit()
    conn.close()

    print(f"Registered: {filename} ({file_size} bytes)")

# Usage:
register_matrix_in_cache(
    "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/1ec0ab77831d12c5058ea56f59c9cd19abe0420714511fba9a458d7aec2f1929.npz"
)
```

### Step 4: Verify Cache

```python
from earthkit.regrid.utils import caching as regrid_caching

# Check cache status
print(f"Cache directory: {regrid_caching.CACHE.directory()}")
print(f"Cache policy: {regrid_caching.CACHE.policy.name}")

entries = regrid_caching.CACHE.entries()
print(f"Cached entries: {len(entries) if entries else 0}")

for entry in entries or []:
    print(f"  - {os.path.basename(entry['path'])} ({entry.get('size', 0)} bytes)")
```

---

## Configuration Reference

### Cache Policy Settings

| Setting | Default | Recommended | Description |
|---------|---------|-------------|-------------|
| `cache-policy` | `"user"` | `"user"` | Use persistent cache in user directory |
| `user-cache-directory` | `~/.cache/earthkit-regrid` | Keep default | Where cache files are stored |
| `url-download-timeout` | `30` | `300` | Download timeout in seconds |
| `maximum-cache-size` | `5GB` | `5GB` | Maximum cache size |
| `maximum-cache-disk-usage` | `99` | `99` | Max disk usage percentage |

### Memory Cache Settings

| Setting | Default | Recommended | Description |
|---------|---------|-------------|-------------|
| `matrix-memory-cache-policy` | `"off"` | `"lru"` | In-memory matrix caching |
| `maximum-matrix-memory-cache-size` | `500MB` | `2GB` | Memory cache size limit |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `TQDM_DISABLE=1` | Disable progress bars (can hang in some terminals) |
| `XDG_CACHE_HOME` | Override default cache location |
| `TMPDIR` | Override temporary directory |

---

## Troubleshooting

### Error: "attempt to write a readonly database"

**Cause**: Cache directory was deleted while SQLite connection was open.

**Solution**:
1. Stop the script
2. Delete the entire cache directory: `rm -rf ~/.cache/earthkit-regrid`
3. Restart the script (fresh cache will be created)

### Error: Download stalls at 0%

**Cause**: ECMWF server is slow or tqdm progress bar is hanging.

**Solutions**:
1. Set `TQDM_DISABLE=1` environment variable
2. Increase `url-download-timeout` to 300+ seconds
3. Pre-cache the matrix file manually (see above)

### Error: "No space left on device"

**Cause**: Disk filled up, possibly from failed download attempts.

**Solutions**:
1. Check disk usage: `df -h /scratch`
2. Clean temporary directories:
   ```bash
   rm -rf /scratch/notebook/tmp/*
   rm -rf /scratch/notebook/.cache/earthkit-data/*
   rm -rf /scratch/notebook/earthkit-tmp/*
   ```
3. Do NOT delete `~/.cache/earthkit-regrid` during processing

### Verifying Cache Health

```python
import sqlite3
import os

cache_dir = os.path.expanduser("~/.cache/earthkit-regrid")
db_path = os.path.join(cache_dir, "cache-2.db")

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Check database integrity
try:
    result = conn.execute("PRAGMA integrity_check").fetchone()
    print(f"Database integrity: {result[0]}")
except Exception as e:
    print(f"Database error: {e}")

# List all cached files
cursor = conn.execute("SELECT path, size, accesses FROM cache")
for row in cursor:
    exists = os.path.exists(row['path'])
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {os.path.basename(row['path'])} - {row['size']} bytes, {row['accesses']} accesses")

conn.close()
```

---

## Summary

The earthkit-regrid cache system is powerful but requires careful management:

1. **The SQLite singleton cannot be reset** - Never delete the cache directory during processing
2. **Matrix files are reusable** - They only depend on the grid transformation, not the data
3. **Pre-caching is recommended** - Avoid network issues during critical processing
4. **Use "user" cache policy** - Enables persistent caching across runs

By following these guidelines, you can reliably process large ensemble forecasts without cache corruption or download failures.

---

## References

- [Earthkit-Regrid Documentation](https://earthkit-regrid.readthedocs.io/)
- [ECMWF Earthkit Repository](https://github.com/ecmwf/earthkit-regrid)
- [SQLite Python Documentation](https://docs.python.org/3/library/sqlite3.html)
