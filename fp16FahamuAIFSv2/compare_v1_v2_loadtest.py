#!/usr/bin/env python3
"""AIFS-ENS load-test comparison on NVIDIA L4 (22 GB), FP16:
  - v2 (aifs-ens-2.0): measured here across a chunk sweep {1,8,16,32}.
  - v1 (aifs-ens-1.0): published reference from HF aifs-ens-1.0 Discussion #17
    (FP16 + NUM_CHUNKS=16, 24 GB GPU). v1 could not be re-run on this env: its
    checkpoint is a fully-pickled module tied to the v1-era torch_geometric
    (Inspector API changed in tg 2.6.x), so it needs a separate v1 software env.

Renders a peak-VRAM-vs-chunks PNG and prints a comparison table."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "/scratch/chunk_sweep_results.csv"
OUT_DIR = "/scratch/profile_compare"
os.makedirs(OUT_DIR, exist_ok=True)
GPU_TOTAL = 22.04
# HF aifs-ens-1.0 Discussion #17, FP16 + NUM_CHUNKS=16 on a 24 GB GPU:
DISC17_V1 = {"chunks": 16, "peak_alloc": 20.0, "peak_reserved": 23.0}

rows = []
with open(SWEEP) as f:
    for r in csv.DictReader(f):
        try:
            rows.append({"chunks": int(r["chunks"]),
                         "alloc": float(r["peak_alloc_gb"]),
                         "reserved": float(r["peak_reserved_gb"]),
                         "sps": float(r["s_per_step"]),
                         "result": r["result"]})
        except ValueError:
            print("skip row:", r)
rows.sort(key=lambda x: x["chunks"])
ch = [x["chunks"] for x in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: peak VRAM vs chunks (v2 measured) + v1 Disc#17 reference
ax1.plot(ch, [x["reserved"] for x in rows], "o-", color="tab:green",
         label="v2 peak-reserved (measured)")
ax1.plot(ch, [x["alloc"] for x in rows], "^--", color="tab:green", alpha=0.7,
         label="v2 peak-alloc (measured)")
ax1.scatter([DISC17_V1["chunks"]], [DISC17_V1["peak_reserved"]], color="tab:red",
            marker="*", s=220, zorder=5, label="v1 peak-reserved (Disc#17)")
ax1.scatter([DISC17_V1["chunks"]], [DISC17_V1["peak_alloc"]], color="tab:red",
            marker="P", s=120, zorder=5, label="v1 peak-alloc (Disc#17)")
ax1.axhline(GPU_TOTAL, color="gray", lw=0.9, ls="-", label=f"L4 total {GPU_TOTAL:.0f} GB")
ax1.axhline(23.0, color="orange", lw=0.9, ls=":", label="24 GB-class threshold 23 GB")
ax1.set_xscale("log", base=2); ax1.set_xticks(ch); ax1.set_xticklabels(ch)
ax1.set_xlabel("ANEMOI_INFERENCE_NUM_CHUNKS")
ax1.set_ylabel("Peak GPU memory (GB)")
ax1.set_ylim(0, 26)
ax1.set_title("Peak VRAM vs chunks — FP16, 72h, L4")
ax1.grid(True, alpha=0.3); ax1.legend(fontsize=7, loc="center right")

# Right: speed vs chunks (the cost of more chunking)
ax2.plot(ch, [x["sps"] for x in rows], "s-", color="tab:blue")
ax2.set_xscale("log", base=2); ax2.set_xticks(ch); ax2.set_xticklabels(ch)
ax2.set_xlabel("ANEMOI_INFERENCE_NUM_CHUNKS")
ax2.set_ylabel("seconds / 6h-step")
ax2.set_title("Inference speed vs chunks — v2, FP16, L4")
ax2.grid(True, alpha=0.3)

fig.suptitle("AIFS-ENS v2 (aifs-ens-2.0) load test on NVIDIA L4 — vs v1 Discussion #17",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(OUT_DIR, "v1_vs_v2_loadtest.png")
fig.savefig(png, dpi=130)
plt.close(fig)

print("\n=== AIFS-ENS load test — v2 measured (L4, FP16, 72h) vs v1 Disc#17 ===")
print(f"{'chunks':>7s} {'v2 peak_alloc':>14s} {'v2 peak_reserved':>17s} {'s/step':>8s} {'result':>7s}")
for x in rows:
    print(f"{x['chunks']:>7d} {x['alloc']:>13.2f}G {x['reserved']:>16.2f}G "
          f"{x['sps']:>8.2f} {x['result']:>7s}")
print(f"\n v1 (Disc#17, aifs-ens-1.0, FP16, 16 chunks, 24GB GPU): "
      f"peak_alloc ~{DISC17_V1['peak_alloc']:.0f}G, peak_reserved ~{DISC17_V1['peak_reserved']:.0f}G")
v2_16 = next((x for x in rows if x["chunks"] == 16), None)
if v2_16:
    print(f" v2 @16 chunks measured: peak_reserved {v2_16['reserved']:.2f}G "
          f"-> {DISC17_V1['peak_reserved'] - v2_16['reserved']:.1f}G LOWER than v1 Disc#17")
print(f"\nComparison PNG: {png}")
