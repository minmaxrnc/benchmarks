# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...definitions import SAVED_DATASET_OUTPUT_DIR, SAVED_DATASETS_THRESHOLD

from ...utils.misc import bytes_to_mb_or_gb

PathLike = Union[str, Path]


class SavedDatasetsManager:

    @staticmethod
    def _prune_dir_lru(d: PathLike,
                       f: PathLike,
                       threshold_bytes: int,
                       *,
                       policy: str = "atime",   # "atime" for LRU (default), or "mtime"
                       dry_run: bool = False) -> Dict[str, object]:
        """
        Reduce the total size of files in directory `d` to <= `threshold_bytes`
        by deleting least-recently-used files. The file `f` (if it exists) is
        protected from deletion.

        Deletion order:
          - Oldest by access time when policy="atime" (LRU semantics).
          - Oldest by modification time when policy="mtime".
          - Ties are broken by the other time (mtime/atime).

        Parameters
        ----------
        d : str | Path
            Target directory (non-recursive).
        f : str | Path
            A file path to protect from deletion (commonly the file you just wrote).
        threshold_bytes : int
            Maximum allowed total size (in bytes) of files in `d`.
        policy : {"atime","mtime"}, optional
            Which timestamp to use to decide least-recently-used. Default "atime".
            Note: some systems mount with 'noatime', reducing atime accuracy.
        dry_run : bool, optional
            If True, no files are deleted; a report is returned of what would happen.

        Returns
        -------
        dict
            {
              "deleted": [list of deleted file paths as str],
              "freed_bytes": int,
              "initial_size": int,
              "final_size": int,
              "threshold": int,
              "policy": str
            }

        Raises
        ------
        FileNotFoundError
            If `d` does not exist.
        NotADirectoryError
            If `d` is not a directory.
        ValueError
            If `threshold_bytes` < 0 or `policy` is invalid.
        RuntimeError
            If the directory cannot be reduced to the threshold without
            deleting the protected file `f` (e.g., `f` alone exceeds the threshold),
            or if deletions fail to achieve the threshold.
        """
        d = Path(d)
        f = Path(f)

        if not d.exists():
            raise FileNotFoundError(f"Directory does not exist: {d}")
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        if threshold_bytes < 0:
            raise ValueError("threshold_bytes must be >= 0")
        if policy not in {"atime", "mtime"}:
            raise ValueError("policy must be 'atime' or 'mtime'")

        # Resolve the protected filepath if it exists; fall back to absolute path string match otherwise.
        protected_resolved: Path | None = None
        try:
            if f.exists():
                protected_resolved = f.resolve()
        except Exception:
            protected_resolved = None  # if we can't resolve, we simply won't match on resolve()

        def is_protected(p: Path) -> bool:
            # Protect if it matches the resolved path (best effort) or exact path string
            if protected_resolved is not None:
                try:
                    return p.resolve() == protected_resolved
                except FileNotFoundError:
                    return False
            # If f doesn't exist, we can't reliably protect; fall back to string equality
            return str(p) == str(f)

        # Gather file entries (non-recursive). Follow symlinks for size/time (typical expectation).
        files: List[Tuple[Path, int, float, float]] = []  # (path, size, atime, mtime)
        with os.scandir(d) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=True):
                        st = entry.stat(follow_symlinks=True)
                        files.append((Path(entry.path), st.st_size, st.st_atime, st.st_mtime))
                except FileNotFoundError:
                    # File may have disappeared between scandir and stat; skip it.
                    continue

        initial_total = sum(size for _, size, _, _ in files)
        if initial_total <= threshold_bytes:
            return {
                "deleted": [],
                "freed_bytes": 0,
                "initial_size": initial_total,
                "final_size": initial_total,
                "threshold": threshold_bytes,
                "policy": policy,
            }

        # Sort deletion candidates by the chosen policy (oldest first).
        idx = 2 if policy == "atime" else 3  # 2: atime, 3: mtime
        # exclude protected file
        candidates = [(p, size, at, mt) for (p, size, at, mt) in files if not is_protected(p)]
        candidates.sort(key=lambda t: (t[idx], t[3] if idx == 2 else t[2]))

        deleted: List[str] = []
        freed = 0

        # Perform deletions until we are under/at threshold or run out
        for p, size, _, _ in candidates:
            if initial_total - freed <= threshold_bytes:
                break
            try:
                if not dry_run:
                    p.unlink()
                deleted.append(str(p))
                freed += size
            except FileNotFoundError:
                # Already gone; treat as freed 0
                continue
            except PermissionError as e:
                # Skip and continue trying other files
                continue

        final_total = initial_total - freed
        if final_total > threshold_bytes:
            # If the only way to reach the threshold is to delete the protected file,
            # or if permission issues prevented enough deletions, fail clearly.
            raise RuntimeError(
                "Unable to reduce directory size to the requested threshold without "
                f"deleting the protected file {f!s} or due to permission issues. "
                f"final_size={final_total}, threshold={threshold_bytes}, deleted={deleted}"
            )

        return {
            "deleted":      deleted,
            "freed_bytes":  freed,
            "initial_size": initial_total,
            "final_size":   final_total,
            "threshold":    threshold_bytes,
            "policy":       policy,
        }

    @staticmethod
    def run(protected_file):
        meta = SavedDatasetsManager._prune_dir_lru(
                SAVED_DATASET_OUTPUT_DIR,
                protected_file,
                SAVED_DATASETS_THRESHOLD
                )
        max_size = bytes_to_mb_or_gb(SAVED_DATASETS_THRESHOLD)
        if len(meta['deleted']) == 0:
            cur_size = bytes_to_mb_or_gb(meta['final_size'])
            print(f"[Saved datasets manager] Space usage: {cur_size} / {max_size}")
        else:
            freed = bytes_to_mb_or_gb(meta['freed_bytes'])
            prev_size = bytes_to_mb_or_gb(meta['initial_size'])
            new_size = bytes_to_mb_or_gb(meta['final_size'])
            print(f"[Saved datasets manager] Freed {freed}, New size {new_size}, Prev size {prev_size}, Allocated {max_size}")




