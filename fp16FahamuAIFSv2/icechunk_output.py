#!/usr/bin/env python3
"""
Icechunk ensemble output writer for AIFS-ENS-2.0 (Path A — **local** variant).

Implements the writer specified in ``ICECHUNK_PATH_A.md`` §2.1, but targets a
**local filesystem** Icechunk store (``icechunk.local_filesystem_storage``) instead
of GCS. Everything else — the ``(member, time, values)`` schema, the ``(1, n_steps,
n_values)`` chunking that lets members own disjoint chunks, the one-commit-per-member
ACID model — is identical, so this store is a drop-in for the GCS one and reads back
the same way with ``xarray.open_zarr``.

Use for on-box runs where the GPU inference writes straight into a store on a local
disk (e.g. ``/tank/projects/<date>/icechunk_v2``) — no GRIB staging, no upload.

Schema (native N320 reduced-Gaussian cells, no regrid at write time)::

    coords: latitude(values) f4, longitude(values) f4,
            time(time) i4 (seconds since reference_date), member(member) i4
    data:   one (member, time, values) f4/f2 array per model output field
            (ALL fields kept, incl. cos_mwd/sin_mwd — a gain over the GRIB path)

Typical use (single process, sequential members)::

    repo = open_repo_local("/tank/projects/20260625_0000/icechunk_v2")
    init_schema(repo, n_members=50, n_steps=12, n_values=542080,
                var_names=first_state_field_names,
                latitudes=lats, longitudes=lons,
                ref_date=ref_date, timestep_s=6*3600)      # once, before any writer
    w = IcechunkMemberWriter(repo, member_index=member-1)  # member 1 -> index 0
    for state in runner.run(...):
        w.write_step(state)
    w.commit(member)
"""

import numpy as np
import zarr
import icechunk


# ---------------------------------------------------------------------------
# Repository handles
# ---------------------------------------------------------------------------
def open_repo_local(path):
    """Open (or create) a local-filesystem Icechunk repository at ``path``.

    ``path`` is a directory on a local disk; it is created if missing.
    """
    storage = icechunk.local_filesystem_storage(str(path))
    return icechunk.Repository.open_or_create(storage)


def open_repo_gcs(bucket, prefix, service_account_file):
    """Open (or create) a GCS-backed Icechunk repository (parity with Path A doc)."""
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix,
                                   service_account_file=service_account_file)
    return icechunk.Repository.open_or_create(storage)


# ---------------------------------------------------------------------------
# Schema (create once, before any member is written)
# ---------------------------------------------------------------------------
def init_schema(repo, n_members, n_steps, n_values, var_names,
                latitudes, longitudes, ref_date, timestep_s,
                float_size="f4", time_chunk=1, commit_msg=None):
    """Create + commit the (member, time, values) schema on branch ``main``.

    Must run once, committed BEFORE any :class:`IcechunkMemberWriter` opens a
    session. ``var_names`` are the model output field names (derive them from the
    first yielded state so the set always matches the checkpoint exactly).
    ``latitudes``/``longitudes`` are the native N320 cell coordinates (length
    ``n_values``). Returns the commit snapshot id.

    ``time_chunk`` sets how many timesteps live in one chunk along the time axis.
    **Set it equal to the commit cadence** (``1`` = per model step / 6 h, the
    default): a step-by-step writer that commits every ``k`` steps into a chunk of
    ``time_chunk = k`` steps writes each chunk once (~no write amplification). A big
    ``time_chunk`` (e.g. ``n_steps``) written step-by-step read-modify-writes the
    whole chunk every step → severe amplification — see ``ICECHUNK_COMMIT_CADENCE.md``.
    """
    s = repo.writable_session("main")
    root = zarr.group(s.store, overwrite=True)

    root.create_array("latitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = np.asarray(latitudes, "f4")
    root.create_array("longitude", shape=(n_values,), chunks=(n_values,),
                      dtype="f4", dimension_names=("values",))[:] = np.asarray(longitudes, "f4")

    t = root.create_array("time", shape=(n_steps,), chunks=(n_steps,),
                          dtype="i4", dimension_names=("time",))
    t[:] = (np.arange(1, n_steps + 1) * int(timestep_s)).astype("i4")
    t.attrs["units"] = f"seconds since {ref_date}"
    t.attrs["calendar"] = "gregorian"

    root.create_array("member", shape=(n_members,), chunks=(n_members,),
                      dtype="i4", dimension_names=("member",))[:] = \
        np.arange(1, n_members + 1, dtype="i4")

    tc = max(1, min(int(time_chunk), n_steps))
    for v in var_names:
        # fill_value=NaN so members/steps not yet written read as missing, not 0
        # (a 0 fill would masquerade as a real value, e.g. 0 K temperature).
        root.create_array(v, shape=(n_members, n_steps, n_values),
                          chunks=(1, tc, n_values), dtype=float_size,
                          fill_value=float("nan"),
                          dimension_names=("member", "time", "values"))

    msg = commit_msg or (f"init schema: {n_members}x{n_steps}x{n_values}, "
                         f"{len(var_names)} vars ({float_size}), time_chunk={tc}")
    return s.commit(msg)


def schema_exists(repo, sentinel="time"):
    """True if the schema group is already present on ``main`` (idempotent init)."""
    try:
        root = zarr.open_group(repo.readonly_session("main").store, mode="r")
        return sentinel in root
    except Exception:
        return False


def member_written(repo, member_index, probe_index, branch="main", sentinel="2t"):
    """True if ``member_index`` already has data at absolute time ``probe_index``.

    Used for idempotent resume of a multi-member run: pass the **last index the run
    would write** (with a write-window this is not ``n_steps - 1``), so a member that
    finished reads back finite there and can be skipped. ``sentinel`` should be a
    variable that is *not* NaN-masked (``2t`` is a safe default; falls back to the
    first data var).
    """
    try:
        root = zarr.open_group(repo.readonly_session(branch).store, mode="r")
        keys = list(root.array_keys())
        var = sentinel if sentinel in keys else next(
            k for k in keys if k not in ("latitude", "longitude", "time", "member"))
        return bool(np.isfinite(root[var][int(member_index), int(probe_index), 0]))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-member writer (one ACID commit per member)
# ---------------------------------------------------------------------------
class IcechunkMemberWriter:
    """Writes ONE member's rollout into the (member, time, values) arrays, committing
    at a configurable step cadence.

    ``member_index`` is 0-based (ensemble member N -> index N-1). With
    ``commit_every=1`` (the default) it commits **after every model step** — i.e.
    every 6 h for aifs-ens-2.0, the natural inference cadence — so each step becomes a
    durable ACID snapshot as the rollout streams. Pair with ``time_chunk=commit_every``
    in :func:`init_schema` so each committed window is a whole chunk written once
    (no write amplification; see ``ICECHUNK_COMMIT_CADENCE.md``).

    Because each member owns disjoint chunks (member-chunk size 1), sequential members
    never share a chunk, so commits stay conflict-free.
    """

    def __init__(self, repo, member_index, member_number=None,
                 commit_every=1, branch="main", regrid=None, var_filter=None,
                 time_chunk=None):
        """``regrid`` maps a stacked ``(n_fields, n_src_values)`` block to
        ``(n_fields, n_dst_values)`` — pass ``o96_grid.regrid_block`` to write an O96
        store straight off the rollout. ``var_filter`` restricts which fields are
        written (the tier-B N320 sidecar passes ``{"msl", "tp", "2t"}``). ``time_chunk``
        is the store's time-chunk length; steps are buffered and flushed one whole chunk
        at a time, because writing step by step into a multi-step chunk makes zarr
        read-modify-write it once per step and Icechunk (copy-on-write) keeps every
        version — measured at 4.9 GB instead of 1.1 GB for one member. Defaults to
        ``commit_every``, which is what ``init_schema`` uses; at the production
        ``commit_every=1`` a chunk is one step and buffering is a no-op.
        """
        self.repo = repo
        self.branch = branch
        self.m = int(member_index)
        self.member_number = int(member_number) if member_number is not None else self.m + 1
        self.commit_every = max(1, int(commit_every))
        self.regrid = regrid
        self.var_filter = set(var_filter) if var_filter else None
        self.time_chunk = max(1, int(time_chunk if time_chunk is not None
                                     else self.commit_every))
        if self.time_chunk > self.commit_every:
            # A commit always flushes, so committing more often than the chunk length
            # writes the same chunk once per commit -- the amplification the buffering
            # exists to avoid. init_schema uses time_chunk == commit_every, so this only
            # fires if the two were set apart by hand.
            print(f"    [WARN] time_chunk={self.time_chunk} > commit_every="
                  f"{self.commit_every}: each chunk will be rewritten "
                  f"{self.time_chunk // self.commit_every}x. Set them equal.")
        self.n = 0                       # total steps WRITTEN this member
        self._since_commit = 0           # steps written since last commit
        self._pending = []               # absolute time indices in the open session
        self._buf = {}                   # absolute time index -> {var: 1-D values}
        self._buf_chunk = None           # which time chunk _buf belongs to
        self._var_names = None
        self.snapshots = []              # snapshot id per committed window
        self.last_index = None           # last absolute time index written
        self._open()

    def _open(self):
        """Start a fresh writable session (called after each commit)."""
        self.session = self.repo.writable_session(self.branch)
        self.root = zarr.open_group(self.session.store, mode="r+")
        if self._var_names is None:
            self._var_names = set(self.root.array_keys())

    def write_step(self, state, time_index=None):
        """Write one yielded inference state (all fields), committing on cadence.

        ``time_index`` is the **absolute** 0-based index on the time axis. Pass it
        explicitly when some steps are skipped (e.g. only the downstream window is
        stored) — otherwise the written count would silently shift every later step
        into the wrong slot. Defaults to the written count (contiguous writes).
        Skipped steps allocate no chunks and read back as NaN.
        """
        idx = self.n if time_index is None else int(time_index)
        fields = state["fields"]
        names = [n for n in fields
                 if n in self._var_names
                 and (self.var_filter is None or n in self.var_filter)]

        if self.regrid is None:
            values = {n: np.asarray(fields[n]).reshape(-1) for n in names}
        else:
            block = np.stack([np.asarray(fields[n]).reshape(-1) for n in names])
            values = dict(zip(names, self.regrid(block)))

        # Buffer until the step crosses into the next time chunk, so each chunk is
        # written exactly once (see __init__).
        chunk = idx // self.time_chunk
        if self._buf and chunk != self._buf_chunk:
            self._flush_buffer()
        self._buf_chunk = chunk
        self._buf[idx] = values

        self.n += 1
        self._since_commit += 1
        self._pending.append(idx)
        self.last_index = idx
        if self._since_commit >= self.commit_every:
            self._commit_window()

    def _flush_buffer(self):
        """Write the buffered chunk as one slab per variable."""
        if not self._buf:
            return
        idxs = sorted(self._buf)
        lo = self._buf_chunk * self.time_chunk
        hi = min(lo + self.time_chunk, self.root[next(iter(self._buf[idxs[0]]))].shape[1])
        contiguous = idxs == list(range(lo, hi))
        for name in self._buf[idxs[0]]:
            if contiguous:                          # whole chunk present -> one write
                self.root[name][self.m, lo:hi, :] = np.stack(
                    [self._buf[i][name] for i in idxs])
            else:                                   # partial chunk (skipped steps)
                for i in idxs:
                    self.root[name][self.m, i, :] = self._buf[i][name]
        self._buf = {}

    def _commit_window(self):
        """Commit the steps written since the last commit; reopen a fresh session."""
        if not self._pending:                      # nothing staged -> no empty commit
            return None
        self._flush_buffer()
        first, last = self._pending[0], self._pending[-1]
        snap = self.session.commit(
            f"member {self.member_number:03d} steps {first:03d}-{last:03d}")
        self.snapshots.append(snap)
        self._since_commit = 0
        self._pending = []
        self._open()
        return snap

    def finalize(self):
        """Flush any steps not yet committed (when n_steps % commit_every != 0).

        Returns the last snapshot id for this member.
        """
        if self._since_commit > 0:
            self._commit_window()
        return self.snapshots[-1] if self.snapshots else None

    # Backwards-compatible alias: previously commit() sealed the whole member.
    def commit(self, member_number=None):
        if member_number is not None:
            self.member_number = int(member_number)
        return self.finalize()
