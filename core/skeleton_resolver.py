"""Shared skeleton (.pab) resolver for mesh + animation FBX export.

Why this module exists
----------------------
Pearl Abyss character skeletons are **shared at the class level**,
not per-mesh. Every Kliff armour/cloak/boot PAC ships with its own
geometry but points at the single ``phm_01.pab`` rig. Likewise every
Damiane asset points at ``phw_01.pab``.

Pre-v1.22.4 we had two resolvers:

  * A **correct** prefix-based one in ``ui/tab_explorer.py`` used by
    the PAA animation FBX export path.
  * A **broken** sibling-basename one I added in v1.22.3 to the PAC
    mesh FBX export path. That one searches for a ``.pab`` whose
    filename matches the PAC's basename — which is guaranteed to
    miss for character meshes.

Reports from real users (character-mesh FBX export failing every
time) made the mesh-path oversight obvious. This module lifts the
correct logic into one place so both FBX export paths use it.

Prefix ecosystem
----------------
The prefix after ``cd_`` identifies the **rig family**, not the
individual asset. Verified against real game archives:

  ================   =======================================
  Prefix             Typical skeleton / notes
  ================   =======================================
  cd_phm_*           phm_01.pab   hero male (Kliff, 178 bones)
  cd_phw_*           phw_01.pab   hero female (Damiane)
  cd_ptm_*           ptm_01.pab   template male (169 bones)
  cd_ptw_*           ptw_01.pab   template female (rare)
  cd_pfm_*           pfm_01.pab   face male
  cd_pfw_*           pfw_01.pab   face female
  cd_ppdm_*          ppdm_01.pab  pair-detail male (eye variants)
  cd_ppdw_*          ppdw_01.pab  pair-detail female
  cd_pgm_*           pgm_01.pab   gear male
  cd_pgw_*           pgw_01.pab   gear female
  cd_prh_*           prh_01.pab   player ride horse
  cd_rd_*            rd_*.pab     ride/mount variants
  nhm_*              nhm_01.pab   NPC human male   (no cd_)
  nhw_*              nhw_01.pab   NPC human female (no cd_)
  cd_ngm_*           ngm_01.pab   NPC goblin male
  ================   =======================================

Animation files often embed the rig prefix mid-filename
(``cd_seq_*_phm1_*`` / ``*_phw_*``) rather than at the start. The
resolver accepts an explicit list of sub-patterns for those cases.

API shape
---------
The resolver is a pure module — no Qt, no UI, no disk I/O except
through the small ``SkeletonVfs`` protocol that wraps the existing
``VfsManager.load_pamt`` / ``read_entry_data`` pair. That keeps the
core logic unit-testable against synthetic fixtures.

Top-level entry points:

  detect_rig_prefix(filename) -> str | None
      Pure string match. Given an asset filename, return the
      canonical 3- or 4-letter rig prefix (``'phm'``, ``'ppdm'``,
      ``'rd'``, …) or ``None`` when no pattern matches.

  rank_skeleton_candidates(rig_prefix, pab_paths, asset_path) -> list[str]
      Order a list of ``.pab`` paths from best to worst candidate
      for the asset under consideration. Does not touch the disk;
      ranking is purely lexical + structural.

  resolve_skeleton(asset_path, vfs, manual_override=None) -> SkeletonResolution
      Full resolution: runs the prefix detect, ranks candidates
      found through the VFS, loads the chosen PAB, and returns a
      dataclass with the parsed skeleton plus enough metadata for
      the UI to explain what happened.

UI layer notes
--------------
The UI should:
  1. Always call :func:`resolve_skeleton` first.
  2. If ``resolution.skeleton`` is populated, use it directly.
  3. If it's None, surface ``resolution.reason`` to the user and
     offer a manual-browse fallback that calls
     :func:`load_skeleton_from_path` with the user-picked path.
  4. Remember the manual override via ``rig_prefix`` in config so
     the next export for the same class skips the dialog.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol, Sequence


# ── Prefix detection ─────────────────────────────────────────────────

# Ordered list of (canonical_prefix, substrings_that_trigger_it).
# Order matters — more specific patterns go first so 'pgm' doesn't
# eat a string that really meant 'pgmX' for some future rig.
#
# Each pattern is matched as a **substring** of the lowered filename
# (not a regex) to keep the logic fast and the false-positive surface
# small. A hyphen-style boundary (`_phm_`, `_phm1_`, …) is required
# on either side when the prefix appears mid-string so we don't
# match 'phm' inside 'symphm_' or similar.
_PREFIX_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 4-letter prefixes have to come before the 3-letter ones that
    # are substrings of them — 'ppdm' must win over 'pdm' if we ever
    # add 'pdm', and 'ppdm' before 'ppd'.
    ("ppdm", ("cd_ppdm_", "_ppdm_", "_ppdm1_")),
    ("ppdw", ("cd_ppdw_", "_ppdw_", "_ppdw1_")),
    # 3-letter prefixes ordered roughly by how common they are.
    ("phm",  ("cd_phm_",  "_phm_",  "_phm1_",  "_phm2_",  "_phm3_",  "_phm8_")),
    ("phw",  ("cd_phw_",  "_phw_",  "_phw1_",  "_phw2_",  "_phw3_",  "_phw8_")),
    ("ptm",  ("cd_ptm_",  "_ptm_",  "_ptm1_",  "_ptm2_")),
    ("ptw",  ("cd_ptw_",  "_ptw_",  "_ptw1_")),
    ("pfm",  ("cd_pfm_",  "_pfm_",  "_pfm1_")),
    ("pfw",  ("cd_pfw_",  "_pfw_",  "_pfw1_")),
    ("pgm",  ("cd_pgm_",  "_pgm_",  "_pgm1_")),
    ("pgw",  ("cd_pgw_",  "_pgw_",  "_pgw1_")),
    ("prh",  ("cd_prh_",  "_prh_",  "cd_rd_prh_")),
    ("ngm",  ("cd_ngm_",  "_ngm_")),
    ("ngw",  ("cd_ngw_",  "_ngw_")),
    # NPC-family prefixes that appear without the 'cd_' wrapper.
    ("nhm",  ("nhm_",)),
    ("nhw",  ("nhw_",)),
    # Ride/mount catch-all.
    ("rd",   ("cd_rd_",)),
)


# Expose the ordered prefix list for tests and for the manual-browse
# dialog (which shows an ordered dropdown of rig classes).
KNOWN_RIG_PREFIXES: tuple[str, ...] = tuple(p for p, _ in _PREFIX_PATTERNS)


def detect_rig_prefix(filename: str) -> Optional[str]:
    """Detect the rig class prefix from an asset filename.

    Accepts both the basename and full paths. Case-insensitive.
    Returns the canonical prefix (``'phm'``, ``'ppdm'``, ...) or
    ``None`` when no pattern matches.

    Pattern matching is in two phases:

    1. **Bare-rig start match** — when the filename *starts with*
       ``<prefix>_`` (e.g. ``phm_01.pab``, ``nhm_guard.pac``) we
       return that prefix directly. This is the canonical form
       used for shared class rigs.

    2. **Substring match** — asset filenames usually embed the
       prefix with a boundary (``cd_phm_``, ``_phm_``, …). The
       ordered pattern list handles these, with more specific
       (4-letter) prefixes checked before shorter ones so ``ppdm``
       doesn't get out-muscled by a hypothetical ``pdm``.

    This function is a pure string operation — no disk access.
    """
    if not filename:
        return None
    name = os.path.basename(filename).lower()

    # Phase 1: bare start-of-name match. Iterate in the same order as
    # _PREFIX_PATTERNS so 4-letter prefixes (ppdm, ppdw) get first
    # refusal over 3-letter ones (ptm, pgm).
    for prefix, _ in _PREFIX_PATTERNS:
        if name.startswith(prefix + "_"):
            return prefix

    # Phase 2: substring match for embedded forms like cd_phm_* or
    # mid-string _phm_.
    for prefix, patterns in _PREFIX_PATTERNS:
        for pattern in patterns:
            if pattern in name:
                return prefix
    return None


# ── Candidate ranking ────────────────────────────────────────────────

_PAB_EXT_RE = re.compile(r"\.pab$", re.IGNORECASE)


def _same_directory(asset_path: str, pab_path: str) -> bool:
    """True when the asset and PAB share the same archive directory."""
    a = asset_path.replace("\\", "/").rsplit("/", 1)
    p = pab_path.replace("\\", "/").rsplit("/", 1)
    if len(a) < 2 or len(p) < 2:
        return False
    return a[0].lower() == p[0].lower()


def rank_skeleton_candidates(
    rig_prefix: Optional[str],
    pab_paths: Iterable[str],
    asset_path: str = "",
) -> list[str]:
    """Rank candidate .pab paths from best to worst for this asset.

    Ranking rules (ties broken by the next rule down):

    1.  Filename starts with ``<rig_prefix>_`` (exact class match).
        Within this bucket, prefer the shortest basename — ``phm_01.pab``
        wins over ``phm_01_lod2.pab`` which wins over
        ``phm_01_experimental.pab``. This is the key rule that
        picks the canonical shared rig.
    2.  Lives in the same archive directory as the asset.
    3.  Shortest overall filename (rough "most generic" heuristic).
    4.  Lexical order (deterministic tiebreak so tests stay stable).

    PAB paths that aren't valid strings are silently dropped.
    Duplicates are collapsed preserving first-occurrence order.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for p in pab_paths:
        if not isinstance(p, str) or not p:
            continue
        key = p.replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p.replace("\\", "/"))

    def _rank_key(path: str) -> tuple:
        base = os.path.basename(path).lower()
        prefix_match = bool(
            rig_prefix and base.startswith(rig_prefix.lower() + "_")
        )
        same_dir = bool(asset_path and _same_directory(asset_path, path))
        return (
            0 if prefix_match else 1,
            0 if same_dir else 1,
            len(base),
            base,
        )

    return sorted(unique, key=_rank_key)


# ── VFS protocol ─────────────────────────────────────────────────────

class SkeletonVfs(Protocol):
    """Minimum VFS surface required to resolve a skeleton.

    ``VfsManager`` from :mod:`core.vfs_manager` already satisfies
    this protocol. The tests use a tiny fake that wraps a path→bytes
    dict, so production code and tests share the same resolver.
    """

    def list_pab_paths(self) -> list[str]:
        """Return every ``.pab`` path visible through the VFS."""
        ...

    def read_pab_bytes(self, path: str) -> bytes:
        """Return the raw bytes of the PAB at *path*.

        Raises ``LookupError`` when the path isn't known.
        """
        ...


class VfsManagerAdapter:
    """Wrap a real ``VfsManager`` so it conforms to :class:`SkeletonVfs`.

    The adapter scans every loaded PAMT for ``.pab`` entries once
    and caches the result. A second call with a different VfsManager
    instance would need its own adapter (the cache is per-instance).
    """

    def __init__(self, vfs):
        self._vfs = vfs
        self._pab_index: dict[str, object] | None = None   # path -> entry

    def _ensure_index(self) -> dict[str, object]:
        if self._pab_index is not None:
            return self._pab_index

        index: dict[str, object] = {}
        pamt_cache = getattr(self._vfs, "_pamt_cache", None) or {}
        for _group, pamt_data in pamt_cache.items():
            for entry in getattr(pamt_data, "file_entries", []):
                path = getattr(entry, "path", "")
                if path and path.lower().endswith(".pab"):
                    index[path.replace("\\", "/")] = entry
        self._pab_index = index
        return index

    def list_pab_paths(self) -> list[str]:
        return list(self._ensure_index().keys())

    def read_pab_bytes(self, path: str) -> bytes:
        index = self._ensure_index()
        key = path.replace("\\", "/")
        entry = index.get(key)
        if entry is None:
            # Also accept basename-only queries for convenience.
            base_key = os.path.basename(key).lower()
            for k, v in index.items():
                if os.path.basename(k).lower() == base_key:
                    entry = v
                    break
        if entry is None:
            raise LookupError(f"PAB not found in VFS: {path}")
        return self._vfs.read_entry_data(entry)


# ── Resolution entry point ───────────────────────────────────────────

@dataclass
class SkeletonResolution:
    """Result of a full skeleton resolution attempt.

    ``skeleton`` is the parsed :class:`core.skeleton_parser.Skeleton`
    when resolution succeeded, ``None`` otherwise. ``source`` is a
    short string the UI uses to explain where the rig came from:

      * ``"prefix_match"``   — rig found via prefix auto-detect
      * ``"manual"``         — user picked the .pab by hand
      * ``"sibling_path"``   — rare non-character path: the PAC had a real sibling .pab
      * ``"fallback_scan"``  — last-resort nearest-basename scan

    ``reason`` carries a human-readable diagnostic when
    ``skeleton is None``, e.g.
    ``"no .pab matching prefix 'phm' found in VFS"``.

    ``pab_path`` is the VFS-relative path of the chosen rig, empty
    when nothing was picked. Useful for logging and for the config
    persistence (remembering per-prefix choices).
    """
    skeleton: object = None                # core.skeleton_parser.Skeleton
    pab_path: str = ""
    source: str = ""
    reason: str = ""
    rig_prefix: Optional[str] = None
    candidates_tried: list[str] = field(default_factory=list)


def _parse_pab(raw: bytes, source_path: str):
    """Thin wrapper around ``core.skeleton_parser.parse_pab``.

    Importing lazily so :mod:`core.skeleton_resolver` stays cheap
    to import (unit tests that only exercise the string logic don't
    pay the cost of loading the PAB parser).
    """
    from core.skeleton_parser import parse_pab   # noqa: WPS433 — lazy import
    return parse_pab(raw, source_path)


def resolve_skeleton(
    asset_path: str,
    vfs: SkeletonVfs,
    manual_override: Optional[str] = None,
) -> SkeletonResolution:
    """Best-effort resolve the rig for an asset path.

    Resolution order:

      1. If ``manual_override`` is set and readable, use it.
      2. Detect the rig prefix from the asset basename.
      3. Ask the VFS for every ``.pab`` it knows.
      4. Rank candidates via :func:`rank_skeleton_candidates`.
      5. Take the top candidate, load + parse it.

    Any step that fails produces a populated ``reason`` so the UI
    can show it to the user and decide whether to fall back to a
    mesh-only export or open a manual picker.

    Never raises — all exceptions are captured and surfaced through
    ``reason``. This lets the caller treat the resolver as a
    black box.
    """
    resolution = SkeletonResolution(rig_prefix=detect_rig_prefix(asset_path))

    # 1) Manual override wins unconditionally.
    if manual_override:
        try:
            raw = vfs.read_pab_bytes(manual_override)
        except Exception as e:
            resolution.reason = (
                f"manual override {manual_override!r} could not be read: {e}"
            )
            return resolution
        try:
            parsed = _parse_pab(raw, manual_override)
        except Exception as e:
            resolution.reason = (
                f"manual override {manual_override!r} failed to parse: {e}"
            )
            return resolution
        if not getattr(parsed, "bones", None):
            resolution.reason = (
                f"manual override {manual_override!r} has zero bones"
            )
            return resolution
        resolution.skeleton = parsed
        resolution.pab_path = manual_override
        resolution.source = "manual"
        return resolution

    # 2) Auto-resolve via prefix match.
    try:
        all_pabs = vfs.list_pab_paths()
    except Exception as e:
        resolution.reason = f"VFS enumeration failed: {e}"
        return resolution

    if not all_pabs:
        resolution.reason = "no .pab files visible through the VFS"
        return resolution

    ordered = rank_skeleton_candidates(
        resolution.rig_prefix, all_pabs, asset_path=asset_path,
    )
    resolution.candidates_tried = list(ordered)

    for candidate in ordered:
        try:
            raw = vfs.read_pab_bytes(candidate)
        except Exception:
            continue
        try:
            parsed = _parse_pab(raw, candidate)
        except Exception:
            continue
        if not getattr(parsed, "bones", None):
            continue
        resolution.skeleton = parsed
        resolution.pab_path = candidate
        base = os.path.basename(candidate).lower()
        if (
            resolution.rig_prefix
            and base.startswith(resolution.rig_prefix + "_")
        ):
            resolution.source = "prefix_match"
        elif asset_path and _same_directory(asset_path, candidate):
            resolution.source = "sibling_path"
        else:
            resolution.source = "fallback_scan"
        return resolution

    resolution.reason = (
        f"no usable .pab found (prefix={resolution.rig_prefix!r}, "
        f"{len(all_pabs)} PAB(s) searched)"
    )
    return resolution


def load_skeleton_from_path(path: str, read_bytes: Callable[[], bytes]):
    """Convenience helper for the manual-browse flow.

    Given a user-picked path and a lazy reader that returns the raw
    bytes (from disk or VFS), parse and return the skeleton. Returns
    ``None`` when parsing fails or the skeleton has no bones.
    """
    try:
        raw = read_bytes()
    except Exception:
        return None
    try:
        parsed = _parse_pab(raw, path)
    except Exception:
        return None
    if not getattr(parsed, "bones", None):
        return None
    return parsed
