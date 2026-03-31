"""PAM / PAMLOD / PAC mesh parser for Crimson Desert.

Parses Pearl Abyss 3D mesh files from PAZ archives into an intermediate
representation (vertices, UVs, normals, faces, materials, bones, weights)
that can be exported to OBJ, FBX, or rendered in the 3D preview.

Format overview (all share the 'PAR ' magic):
  PAM     — static meshes (objects, props, world geometry)
  PAMLOD  — LOD variants (5 quality levels per mesh)
  PAC     — skinned character meshes (with bone indices + weights)

Vertex positions are uint16-quantized and dequantized using the per-file
bounding box.  UVs are stored as float16 at vertex offset +8/+10.  Bone
weights (PAC only) follow the UV data.
"""

from __future__ import annotations

import os
import re
import struct
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger("core.mesh_parser")

# ── Constants ────────────────────────────────────────────────────────

PAR_MAGIC = b"PAR "

# PAM header offsets
HDR_MESH_COUNT = 0x10
HDR_BBOX_MIN = 0x14
HDR_BBOX_MAX = 0x20
HDR_GEOM_OFF = 0x3C

# Submesh table
SUBMESH_TABLE = 0x410
SUBMESH_STRIDE = 0x218
SUBMESH_TEX_OFF = 0x10
SUBMESH_MAT_OFF = 0x110

# Global-buffer prefab constants
GLOBAL_VERT_BASE = 3068
PAM_IDX_OFF = 0x19840

# PAMLOD header offsets
PAMLOD_LOD_COUNT = 0x00
PAMLOD_GEOM_OFF = 0x04
PAMLOD_BBOX_MIN = 0x10
PAMLOD_BBOX_MAX = 0x1C
PAMLOD_ENTRY_TABLE = 0x50

# Stride candidates for auto-detection
STRIDE_CANDIDATES = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64]


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class MeshVertex:
    """Single vertex with position, UV, and optional bone data."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    u: float = 0.0
    v: float = 0.0
    nx: float = 0.0
    ny: float = 1.0
    nz: float = 0.0
    bone_indices: tuple[int, ...] = ()
    bone_weights: tuple[float, ...] = ()


@dataclass
class SubMesh:
    """A submesh within a PAM/PAC file."""
    name: str = ""
    material: str = ""
    texture: str = ""
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    bone_indices: list[tuple[int, ...]] = field(default_factory=list)
    bone_weights: list[tuple[float, ...]] = field(default_factory=list)
    vertex_count: int = 0
    face_count: int = 0


@dataclass
class ParsedMesh:
    """Complete parsed mesh file."""
    path: str = ""
    format: str = ""  # "pam", "pamlod", "pac"
    bbox_min: tuple[float, float, float] = (0, 0, 0)
    bbox_max: tuple[float, float, float] = (0, 0, 0)
    submeshes: list[SubMesh] = field(default_factory=list)
    lod_levels: list[list[SubMesh]] = field(default_factory=list)  # PAMLOD only
    total_vertices: int = 0
    total_faces: int = 0
    has_uvs: bool = False
    has_bones: bool = False


# ── Utility ──────────────────────────────────────────────────────────

def _dequant_u16(v: int, mn: float, mx: float) -> float:
    """uint16 → float: bbox_min + (v / 65535) * (bbox_max - bbox_min)."""
    return mn + (v / 65535.0) * (mx - mn)


def _dequant_i16(v: int, mn: float, mx: float) -> float:
    """int16 → float (legacy global-buffer format)."""
    return mn + ((v + 32768) / 65536.0) * (mx - mn)


def _compute_face_normal(v0, v1, v2):
    """Compute face normal from 3 vertex positions."""
    ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 1e-8:
        return (nx / length, ny / length, nz / length)
    return (0.0, 1.0, 0.0)


def _compute_smooth_normals(vertices, faces):
    """Compute per-vertex smooth normals by averaging adjacent face normals."""
    normals = [[0.0, 0.0, 0.0] for _ in range(len(vertices))]
    for a, b, c in faces:
        if a < len(vertices) and b < len(vertices) and c < len(vertices):
            fn = _compute_face_normal(vertices[a], vertices[b], vertices[c])
            for idx in (a, b, c):
                normals[idx][0] += fn[0]
                normals[idx][1] += fn[1]
                normals[idx][2] += fn[2]
    result = []
    for n in normals:
        length = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if length > 1e-8:
            result.append((n[0] / length, n[1] / length, n[2] / length))
        else:
            result.append((0.0, 1.0, 0.0))
    return result


# ── Stride detection ─────────────────────────────────────────────────

def _find_local_stride(data: bytes, geom_off: int, voff: int, n_verts: int, n_idx: int):
    """Detect vertex stride for per-mesh layout where indices follow vertex data."""
    for stride in STRIDE_CANDIDATES:
        vert_start = geom_off + voff
        idx_off = vert_start + n_verts * stride
        if idx_off + n_idx * 2 > len(data):
            continue
        # Validate: all index values must be < n_verts
        valid = True
        for j in range(min(n_idx, 100)):  # sample first 100 for speed
            val = struct.unpack_from("<H", data, idx_off + j * 2)[0]
            if val >= n_verts:
                valid = False
                break
        if valid:
            # Full validation on remaining
            if n_idx > 100:
                valid = all(
                    struct.unpack_from("<H", data, idx_off + j * 2)[0] < n_verts
                    for j in range(100, n_idx)
                )
            if valid:
                return stride, idx_off
    return None, None


# ── PAM Parser ───────────────────────────────────────────────────────

def parse_pam(data: bytes, filename: str = "") -> ParsedMesh:
    """Parse a .pam static mesh file."""
    if len(data) < 0x40 or data[:4] != PAR_MAGIC:
        raise ValueError(f"Not a valid PAM file: bad magic {data[:4]!r}")

    result = ParsedMesh(path=filename, format="pam")
    result.bbox_min = struct.unpack_from("<fff", data, HDR_BBOX_MIN)
    result.bbox_max = struct.unpack_from("<fff", data, HDR_BBOX_MAX)
    geom_off = struct.unpack_from("<I", data, HDR_GEOM_OFF)[0]
    mesh_count = struct.unpack_from("<I", data, HDR_MESH_COUNT)[0]
    bmin, bmax = result.bbox_min, result.bbox_max

    # Read submesh table
    raw_entries = []
    for i in range(mesh_count):
        off = SUBMESH_TABLE + i * SUBMESH_STRIDE
        if off + SUBMESH_STRIDE > len(data):
            break
        nv = struct.unpack_from("<I", data, off)[0]
        ni = struct.unpack_from("<I", data, off + 4)[0]
        ve = struct.unpack_from("<I", data, off + 8)[0]
        ie = struct.unpack_from("<I", data, off + 12)[0]
        tex = data[off + SUBMESH_TEX_OFF:off + SUBMESH_TEX_OFF + 256].split(b"\x00")[0].decode("ascii", "replace")
        mat = data[off + SUBMESH_MAT_OFF:off + SUBMESH_MAT_OFF + 256].split(b"\x00")[0].decode("ascii", "replace")
        raw_entries.append({"i": i, "nv": nv, "ni": ni, "ve": ve, "ie": ie, "tex": tex, "mat": mat})

    # Detect combined-buffer layout
    is_combined = False
    if mesh_count > 1:
        ve_acc = ie_acc = 0
        is_combined = True
        for r in raw_entries:
            if r["ve"] != ve_acc or r["ie"] != ie_acc:
                is_combined = False
                break
            ve_acc += r["nv"]
            ie_acc += r["ni"]

    if is_combined:
        _parse_combined_buffer(data, raw_entries, geom_off, bmin, bmax, result)
    else:
        _parse_independent_meshes(data, raw_entries, geom_off, bmin, bmax, result)

    # Fallback: if no vertices found, try scanning for vertex+index blocks
    # This handles "breakable" PAMs and other extended layouts that have
    # extra data (physics/destruction metadata) before the actual geometry
    if result.total_vertices == 0 and mesh_count > 0:
        _parse_scan_fallback(data, raw_entries, geom_off, bmin, bmax, result)

    # Compute normals for all submeshes
    for sm in result.submeshes:
        sm.normals = _compute_smooth_normals(sm.vertices, sm.faces)

    result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
    result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
    result.has_uvs = any(sm.uvs for sm in result.submeshes)

    logger.info("Parsed PAM %s: %d submeshes, %d verts, %d faces",
                filename, len(result.submeshes), result.total_vertices, result.total_faces)
    return result


def _parse_independent_meshes(data, entries, geom_off, bmin, bmax, result):
    """Parse PAM with per-submesh or global vertex buffers."""
    idx_avail = (len(data) - PAM_IDX_OFF) // 2

    for r in entries:
        i, nv, ni, voff, ioff = r["i"], r["nv"], r["ni"], r["ve"], r["ie"]
        tex, mat = r["tex"], r["mat"]

        # Try local layout first
        stride, idx_off = _find_local_stride(data, geom_off, voff, nv, ni)

        if stride is not None:
            verts, uvs, faces = _extract_local_mesh(data, geom_off, voff, stride, idx_off, nv, ni, bmin, bmax)
        elif ioff + ni <= idx_avail:
            verts, uvs, faces = _extract_global_mesh(data, geom_off, ni, ioff, bmin, bmax)
        else:
            continue

        sm = SubMesh(
            name=f"mesh_{i:02d}_{mat or str(i)}",
            material=mat, texture=tex,
            vertices=verts, uvs=uvs, faces=faces,
            vertex_count=len(verts), face_count=len(faces),
        )
        result.submeshes.append(sm)


def _parse_scan_fallback(data, entries, geom_off, bmin, bmax, result):
    """Fallback parser: scan for vertex+index blocks in extended-layout PAMs.

    Breakable/destructible PAMs often have extra metadata (physics, destruction
    fragments) between the header and the actual geometry. This scanner probes
    the region after geom_off to locate the real vertex positions (uint16
    quantized) and matching index block.
    """
    total_v = sum(r["nv"] for r in entries)
    total_i = sum(r["ni"] for r in entries)
    if total_v < 3 or total_i < 3:
        return

    search_limit = min(len(data) - 100, geom_off + min(len(data) // 2, 2000000))

    # Scan for a block of u16 values that look like quantized vertex positions
    # (spread across the 0-65535 range), followed by valid indices.
    # Step by 2 in small files, step by 4 in large files for speed.
    step = 2 if (search_limit - geom_off) < 500000 else 4
    for scan_start in range(geom_off, search_limit, step):
        # Quick check: read 10 potential XYZ triples (stride 6)
        if scan_start + 60 > len(data):
            break
        vals = [struct.unpack_from("<H", data, scan_start + j * 2)[0] for j in range(30)]
        spread = max(vals) - min(vals)
        if spread < 5000:
            continue

        # Found candidate vertex data. Try common strides
        for try_stride in [6, 8, 10, 12, 14, 16, 20, 24, 28, 32]:
            test_idx_off = scan_start + total_v * try_stride
            if test_idx_off + total_i * 2 > len(data):
                continue

            # Validate: first 50 indices must be < total_v
            valid = True
            for j in range(min(50, total_i)):
                v = struct.unpack_from("<H", data, test_idx_off + j * 2)[0]
                if v >= total_v:
                    valid = False
                    break
            if not valid:
                continue

            # Full validation on a larger sample
            valid = all(
                struct.unpack_from("<H", data, test_idx_off + j * 2)[0] < total_v
                for j in range(min(total_i, 500))
            )
            if not valid:
                continue

            # Found valid layout! Parse as combined buffer from this offset
            logger.info("Scan fallback: found vertex data at 0x%X stride=%d for %s",
                        scan_start, try_stride, entries[0].get("tex", ""))

            has_uv = try_stride >= 12
            idx_base = test_idx_off

            for r in entries:
                nv, ni = r["nv"], r["ni"]
                vert_base = scan_start + r["ve"] * try_stride
                idx_off = idx_base + r["ie"] * 2

                indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0]
                           for j in range(ni)]
                if not indices:
                    continue

                unique = sorted(set(indices))
                idx_map = {gi: li for li, gi in enumerate(unique)}

                verts, uvs = [], []
                for gi in unique:
                    foff = vert_base + gi * try_stride
                    if foff + 6 > len(data):
                        break
                    xu, yu, zu = struct.unpack_from("<HHH", data, foff)
                    verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                                  _dequant_u16(yu, bmin[1], bmax[1]),
                                  _dequant_u16(zu, bmin[2], bmax[2])))
                    if has_uv and foff + 12 <= len(data):
                        u = struct.unpack_from("<e", data, foff + 8)[0]
                        v = struct.unpack_from("<e", data, foff + 10)[0]
                        uvs.append((u, v))

                faces = []
                for j in range(0, ni - 2, 3):
                    a, b, c = indices[j], indices[j + 1], indices[j + 2]
                    if a in idx_map and b in idx_map and c in idx_map:
                        faces.append((idx_map[a], idx_map[b], idx_map[c]))

                sm = SubMesh(
                    name=f"mesh_{r['i']:02d}_{r['mat'] or str(r['i'])}",
                    material=r["mat"], texture=r["tex"],
                    vertices=verts, uvs=uvs, faces=faces,
                    vertex_count=len(verts), face_count=len(faces),
                )
                result.submeshes.append(sm)

            result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
            result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
            result.has_uvs = any(sm.uvs for sm in result.submeshes)
            return  # Done

    # Second pass: scan BACKWARD from end of file for the index block
    # This handles files where extra per-vertex data creates non-integer strides
    for scan_end_off in range(len(data) - 2, geom_off + total_v * 6, -2):
        test_start = scan_end_off - total_i * 2 + 2
        if test_start < geom_off:
            break

        # Quick check first index
        first_val = struct.unpack_from("<H", data, test_start)[0]
        if first_val >= total_v:
            continue

        # Check first 30 indices
        valid = True
        for j in range(min(30, total_i)):
            v = struct.unpack_from("<H", data, test_start + j * 2)[0]
            if v >= total_v:
                valid = False
                break
        if not valid:
            continue

        # Deeper validation
        valid = all(
            struct.unpack_from("<H", data, test_start + j * 2)[0] < total_v
            for j in range(min(total_i, 300))
        )
        if not valid:
            continue

        # Full validation
        valid = all(
            struct.unpack_from("<H", data, test_start + j * 2)[0] < total_v
            for j in range(total_i)
        )
        if not valid:
            continue

        # Found index block! Calculate vertex region
        vert_region = test_start - geom_off
        # Try common strides that fit
        best_stride = None
        for try_stride in [6, 8, 10, 12, 14, 16, 20, 24, 28, 32]:
            expected_end = geom_off + total_v * try_stride
            # Allow up to 16KB padding between vertex data and index data
            if expected_end <= test_start and (test_start - expected_end) < 16384:
                best_stride = try_stride
                break

        if best_stride is None:
            # Use floor division of vert_region / total_v
            best_stride = vert_region // total_v
            if best_stride < 6:
                best_stride = 6

        has_uv = best_stride >= 12
        idx_base = test_start
        logger.info("Backward scan: found idx at 0x%X stride=%d for %d verts",
                    test_start, best_stride, total_v)

        for r in entries:
            nv, ni = r["nv"], r["ni"]
            vert_base = geom_off + r["ve"] * best_stride
            idx_off = idx_base + r["ie"] * 2

            indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0]
                       for j in range(ni)]
            if not indices:
                continue

            unique = sorted(set(indices))
            idx_map = {gi: li for li, gi in enumerate(unique)}

            verts, uvs = [], []
            for gi in unique:
                foff = vert_base + gi * best_stride
                if foff + 6 > len(data):
                    break
                xu, yu, zu = struct.unpack_from("<HHH", data, foff)
                verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                              _dequant_u16(yu, bmin[1], bmax[1]),
                              _dequant_u16(zu, bmin[2], bmax[2])))
                if has_uv and foff + 12 <= len(data):
                    u = struct.unpack_from("<e", data, foff + 8)[0]
                    v = struct.unpack_from("<e", data, foff + 10)[0]
                    uvs.append((u, v))

            faces = []
            for j in range(0, ni - 2, 3):
                a, b, c = indices[j], indices[j + 1], indices[j + 2]
                if a in idx_map and b in idx_map and c in idx_map:
                    faces.append((idx_map[a], idx_map[b], idx_map[c]))

            sm = SubMesh(
                name=f"mesh_{r['i']:02d}_{r['mat'] or str(r['i'])}",
                material=r["mat"], texture=r["tex"],
                vertices=verts, uvs=uvs, faces=faces,
                vertex_count=len(verts), face_count=len(faces),
            )
            result.submeshes.append(sm)

        result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
        result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
        result.has_uvs = any(sm.uvs for sm in result.submeshes)
        return

    logger.debug("Scan fallback: no valid vertex block found after 0x%X", geom_off)


def _parse_combined_buffer(data, entries, geom_off, bmin, bmax, result):
    """Parse PAM with shared vertex + index buffer."""
    total_verts = sum(r["nv"] for r in entries)
    total_idx = sum(r["ni"] for r in entries)
    avail = len(data) - geom_off

    target = (avail - total_idx * 2) / total_verts if total_verts else 0
    stride = min(STRIDE_CANDIDATES, key=lambda s: abs(s - target))
    if geom_off + total_verts * stride + total_idx * 2 > len(data):
        return

    idx_base = geom_off + total_verts * stride

    for r in entries:
        nv, ni = r["nv"], r["ni"]
        vert_base = geom_off + r["ve"] * stride
        idx_off = idx_base + r["ie"] * 2
        tex, mat = r["tex"], r["mat"]

        indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0] for j in range(ni)]
        if not indices:
            continue

        unique = sorted(set(indices))
        idx_map = {gi: li for li, gi in enumerate(unique)}
        has_uv = stride >= 12

        verts, uvs = [], []
        for gi in unique:
            foff = vert_base + gi * stride
            if foff + 6 > len(data):
                break
            xu, yu, zu = struct.unpack_from("<HHH", data, foff)
            verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                          _dequant_u16(yu, bmin[1], bmax[1]),
                          _dequant_u16(zu, bmin[2], bmax[2])))
            if has_uv and foff + 12 <= len(data):
                u = struct.unpack_from("<e", data, foff + 8)[0]
                v = struct.unpack_from("<e", data, foff + 10)[0]
                uvs.append((u, v))

        faces = []
        for j in range(0, ni - 2, 3):
            a, b, c = indices[j], indices[j + 1], indices[j + 2]
            if a in idx_map and b in idx_map and c in idx_map:
                faces.append((idx_map[a], idx_map[b], idx_map[c]))

        sm = SubMesh(
            name=f"mesh_{r['i']:02d}_{mat or str(r['i'])}",
            material=mat, texture=tex,
            vertices=verts, uvs=uvs, faces=faces,
            vertex_count=len(verts), face_count=len(faces),
        )
        result.submeshes.append(sm)


def _extract_local_mesh(data, geom_off, voff, stride, idx_off, nv, ni, bmin, bmax):
    """Extract vertices/uvs/faces from local (per-mesh) layout."""
    indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0] for j in range(ni)]
    unique = sorted(set(indices))
    idx_map = {gi: li for li, gi in enumerate(unique)}
    has_uv = stride >= 12

    verts, uvs = [], []
    for gi in unique:
        foff = geom_off + voff + gi * stride
        if foff + 6 > len(data):
            break
        xu, yu, zu = struct.unpack_from("<HHH", data, foff)
        verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                      _dequant_u16(yu, bmin[1], bmax[1]),
                      _dequant_u16(zu, bmin[2], bmax[2])))
        if has_uv and foff + 12 <= len(data):
            u = struct.unpack_from("<e", data, foff + 8)[0]
            v = struct.unpack_from("<e", data, foff + 10)[0]
            uvs.append((u, v))

    faces = []
    for j in range(0, ni - 2, 3):
        a, b, c = indices[j], indices[j + 1], indices[j + 2]
        if a in idx_map and b in idx_map and c in idx_map:
            faces.append((idx_map[a], idx_map[b], idx_map[c]))

    return verts, uvs, faces


def _extract_global_mesh(data, geom_off, ni, ioff, bmin, bmax):
    """Extract vertices/uvs/faces from global (prefab) layout."""
    indices = [struct.unpack_from("<H", data, PAM_IDX_OFF + (ioff + j) * 2)[0] for j in range(ni)]
    unique = sorted(set(indices))
    idx_map = {gi: li for li, gi in enumerate(unique)}

    verts = []
    for gi in unique:
        li = gi - GLOBAL_VERT_BASE
        foff = geom_off + li * 6
        if foff + 6 > len(data):
            break
        xi, yi, zi = struct.unpack_from("<hhh", data, foff)
        verts.append((_dequant_i16(xi, bmin[0], bmax[0]),
                      _dequant_i16(yi, bmin[1], bmax[1]),
                      _dequant_i16(zi, bmin[2], bmax[2])))

    faces = []
    for j in range(0, ni - 2, 3):
        a, b, c = indices[j], indices[j + 1], indices[j + 2]
        if a in idx_map and b in idx_map and c in idx_map:
            faces.append((idx_map[a], idx_map[b], idx_map[c]))

    return verts, [], faces


# ── PAMLOD Parser ────────────────────────────────────────────────────

def parse_pamlod(data: bytes, filename: str = "", lod_level: int = 0) -> ParsedMesh:
    """Parse a .pamlod LOD mesh file. lod_level=0 is highest quality."""
    result = ParsedMesh(path=filename, format="pamlod")

    lod_count = struct.unpack_from("<I", data, PAMLOD_LOD_COUNT)[0]
    geom_off = struct.unpack_from("<I", data, PAMLOD_GEOM_OFF)[0]
    if lod_count == 0 or geom_off == 0 or geom_off >= len(data):
        return result

    result.bbox_min = struct.unpack_from("<fff", data, PAMLOD_BBOX_MIN)
    result.bbox_max = struct.unpack_from("<fff", data, PAMLOD_BBOX_MAX)
    bmin, bmax = result.bbox_min, result.bbox_max

    # Locate LOD entries by scanning for .dds texture strings
    entries = []
    search_region = data[PAMLOD_ENTRY_TABLE:geom_off]
    for m in re.finditer(rb"[^\x00]{1,255}\.dds\x00", search_region):
        tex_start = PAMLOD_ENTRY_TABLE + m.start()
        nv_off = tex_start - 0x10
        if nv_off < PAMLOD_ENTRY_TABLE:
            continue
        nv = struct.unpack_from("<I", data, nv_off)[0]
        ni = struct.unpack_from("<I", data, nv_off + 4)[0]
        if not (1 <= nv <= 131072 and ni > 0 and ni % 3 == 0):
            continue
        voff = struct.unpack_from("<I", data, tex_start - 0x08)[0]
        ioff = struct.unpack_from("<I", data, tex_start - 0x04)[0]
        tex = data[tex_start:tex_start + 256].split(b"\x00")[0].decode("ascii", "replace")
        mat_start = tex_start + 0x100
        mat = data[mat_start:mat_start + 256].split(b"\x00")[0].decode("ascii", "replace") if mat_start < geom_off else ""
        entries.append({"nv": nv, "ni": ni, "voff": voff, "ioff": ioff,
                        "tex_start": tex_start, "tex": tex, "mat": mat})

    entries.sort(key=lambda e: e["tex_start"])

    # Group into LOD levels
    lod_groups = []
    cur_group, ve_acc, ie_acc = [], 0, 0
    for e in entries:
        if e["voff"] == ve_acc and e["ioff"] == ie_acc:
            cur_group.append(e)
            ve_acc += e["nv"]
            ie_acc += e["ni"]
        else:
            if cur_group:
                lod_groups.append(cur_group)
            cur_group = [e]
            ve_acc = e["nv"]
            ie_acc = e["ni"]
    if cur_group:
        lod_groups.append(cur_group)
    lod_groups = lod_groups[:lod_count]

    if not lod_groups:
        return result

    # Parse each LOD level
    cur = geom_off
    for lod_i, group in enumerate(lod_groups):
        total_nv = sum(e["nv"] for e in group)
        total_ni = sum(e["ni"] for e in group)

        # Find stride with padding scan
        found_base = found_stride = found_idx_off = None
        for pad in range(0, 64, 2):
            base = cur + pad
            for stride in STRIDE_CANDIDATES:
                cand = base + total_nv * stride
                if cand + total_ni * 2 > len(data):
                    continue
                if all(struct.unpack_from("<H", data, cand + j * 2)[0] < total_nv
                       for j in range(min(total_ni, 100))):
                    found_base = base
                    found_stride = stride
                    found_idx_off = cand
                    break
            if found_base is not None:
                break

        if found_base is None:
            result.lod_levels.append([])
            cur += 2
            continue

        # Parse submeshes for this LOD
        lod_submeshes = []
        vert_offset = 0
        has_uv = found_stride >= 12

        all_verts, all_uvs, all_faces = [], [], []
        for e in group:
            nv_e, ni_e = e["nv"], e["ni"]
            vert_base_e = found_base + e["voff"] * found_stride
            idx_off_e = found_idx_off + e["ioff"] * 2

            indices = [struct.unpack_from("<H", data, idx_off_e + j * 2)[0] for j in range(ni_e)]
            unique = sorted(set(indices))
            idx_map = {gi: li + vert_offset for li, gi in enumerate(unique)}

            for gi in unique:
                foff = vert_base_e + gi * found_stride
                if foff + 6 > len(data):
                    break
                xu, yu, zu = struct.unpack_from("<HHH", data, foff)
                all_verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                                  _dequant_u16(yu, bmin[1], bmax[1]),
                                  _dequant_u16(zu, bmin[2], bmax[2])))
                if has_uv and foff + 12 <= len(data):
                    u = struct.unpack_from("<e", data, foff + 8)[0]
                    v = struct.unpack_from("<e", data, foff + 10)[0]
                    all_uvs.append((u, v))

            for j in range(0, ni_e - 2, 3):
                a, b, c = indices[j], indices[j + 1], indices[j + 2]
                if a in idx_map and b in idx_map and c in idx_map:
                    all_faces.append((idx_map[a], idx_map[b], idx_map[c]))

            vert_offset += len(unique)

        mat_name = group[0]["mat"] or f"lod{lod_i}"
        sm = SubMesh(
            name=f"lod{lod_i:02d}_{mat_name}",
            material=mat_name,
            texture=group[0]["tex"],
            vertices=all_verts, uvs=all_uvs, faces=all_faces,
            normals=_compute_smooth_normals(all_verts, all_faces),
            vertex_count=len(all_verts), face_count=len(all_faces),
        )
        lod_submeshes.append(sm)
        result.lod_levels.append(lod_submeshes)
        cur = found_idx_off + total_ni * 2

    # Use requested LOD level as the main submeshes
    if lod_level < len(result.lod_levels) and result.lod_levels[lod_level]:
        result.submeshes = result.lod_levels[lod_level]
    elif result.lod_levels:
        # Fallback to first non-empty LOD
        for lod in result.lod_levels:
            if lod:
                result.submeshes = lod
                break

    result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
    result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
    result.has_uvs = any(sm.uvs for sm in result.submeshes)

    logger.info("Parsed PAMLOD %s: %d LODs, using LOD %d (%d verts, %d faces)",
                filename, len(result.lod_levels), lod_level,
                result.total_vertices, result.total_faces)
    return result


# ── PAC Parser (skinned mesh) ────────────────────────────────────────

def parse_pac(data: bytes, filename: str = "") -> ParsedMesh:
    """Parse a .pac skinned character mesh.

    PAC format (reverse-engineered from binary analysis):
      Header: 80 bytes
        [0x00] 4B: 'PAR ' magic
        [0x04] 4B: version (0x01000903)
        [0x10] 4B: zero
        [0x14] N×8B or N×4B: section sizes (u64 or u32, variable count)

      Section 0: Metadata
        - u32 flags, u8 n_lods
        - n_lods × u32: section start offsets (LOD0 first)
        - n_lods × u32: vertex/index split offsets per section
        - Per submesh descriptor:
            [u8 len][mesh_name] [u8 len][mat_name]
            [u8 flag][2B pad] [8 floats: pivot(2) + bbox(6)]
            [u8 bone_count][bone_indices...]
            [n_lods × u16: vert counts] [n_lods × u32: idx counts]

      Sections 1..N: LOD levels (1=lowest, N=highest/LOD0)
        Part A: 40-byte vertex records (up to split offset)
        Part B: uint16 triangle list indices (after split offset)

      40-byte vertex record:
        [0-5]  3×uint16: quantized XYZ position
        [6-7]  uint16: packed data (normal/tangent)
        [8-11] 2×float16: UV coordinates
        [12-15] constant (0x3C000000)
        [16-19] 4 bytes data
        [20-27] zeros
        [28-31] bone index bytes (0xFF=none)
        [32-35] bone weight bytes
        [36-39] FFFFFFFF terminator

      Per-submesh bounding box for dequantization:
        bbox_min = (float[2], float[3], float[4])
        bbox_max = (float[5], float[6], float[7])
        pivot    = (float[0], float[1])  (bone attachment point)
    """
    if len(data) < 0x50 or data[:4] != PAR_MAGIC:
        raise ValueError(f"Not a valid PAC file: bad magic {data[:4]!r}")

    result = ParsedMesh(path=filename, format="pac")

    # ── Parse section layout using section offset table in section 0 ──
    # Section 0 always starts at byte 80. Its first bytes contain:
    #   [u32 flags] [u8 n_lods] [n_lods × u32 section_offsets] [n_lods × u32 split_offsets]
    # Section offsets are absolute file positions (LOD0 first = largest, descending).
    # This is the most reliable way to determine section boundaries.
    header_size = 80

    if len(data) < header_size + 5:
        return _pac_fallback_pam(data, filename)

    s0_start = header_size
    off = s0_start
    flags = struct.unpack_from("<I", data, off)[0]
    n_lods = data[off + 4]
    off += 5

    if n_lods == 0 or n_lods > 10:
        return _pac_fallback_pam(data, filename)

    # Read section offsets (absolute file positions, LOD0 first = descending)
    lod_offsets = [struct.unpack_from("<I", data, off + i * 4)[0] for i in range(n_lods)]
    off += n_lods * 4
    # Skip split offsets (we compute splits from vertex counts)
    off += n_lods * 4

    # Compute section boundaries from offsets:
    #   sec0: header_size to min(lod_offsets)
    #   LOD sections: between sorted offsets, last one ends at file_end
    sorted_offsets = sorted(lod_offsets)
    boundaries = [header_size] + sorted_offsets + [len(data)]
    sections = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Validate: sec0 must have positive size
    if sections[0][1] <= sections[0][0]:
        return _pac_fallback_pam(data, filename)

    s0_end = sections[0][1]

    # ── Find and parse submesh descriptors ──
    # Scan forward for first length-prefixed ASCII string
    scan = off
    while scan < s0_end - 10:
        b = data[scan]
        if 4 < b < 100:
            test = data[scan + 1:scan + 1 + b]
            if len(test) == b and all(32 <= c < 127 for c in test):
                break
        scan += 1
    off = scan

    pac_submeshes = []
    while off < s0_end - 20:
        name_len = data[off]
        if name_len == 0 or name_len > 200 or off + 1 + name_len >= s0_end:
            break
        mesh_name = data[off + 1:off + 1 + name_len].decode("ascii", "replace")
        off += 1 + name_len
        if not all(32 <= ord(c) < 127 for c in mesh_name):
            break

        mat_len = data[off]
        mat_name = data[off + 1:off + 1 + mat_len].decode("ascii", "replace") if mat_len > 0 else ""
        off += 1 + mat_len

        # flag(1) + pad(2) + 8 floats(32) + bone data
        off += 3
        bbox_floats = [struct.unpack_from("<f", data, off + i * 4)[0] for i in range(8)]
        off += 32

        # Bone data: [u8 bone_count] [bone_count × u8 indices]
        # Bone indices are padded to even byte count (odd bc gets +1 pad byte).
        bone_count = data[off]
        off += 1
        bones_size = bone_count + (bone_count % 2)  # round up to even
        off += bones_size

        # Per-LOD vertex counts (n_lods × u16) + index counts (n_lods × u32)
        # Some files have fewer idx_counts than n_lods — validate and truncate.
        vert_counts = [struct.unpack_from("<H", data, off + i * 2)[0] for i in range(n_lods)]
        off += n_lods * 2

        idx_counts = []
        max_reasonable_idx = 10_000_000  # no single submesh has 10M indices
        for i in range(n_lods):
            if off + 4 > s0_end:
                break
            val = struct.unpack_from("<I", data, off)[0]
            if val > max_reasonable_idx:
                break  # hit garbage — stop reading idx_counts
            idx_counts.append(val)
            off += 4
        # Pad missing LODs with 0
        while len(idx_counts) < n_lods:
            idx_counts.append(0)

        bmin = (bbox_floats[2], bbox_floats[3], bbox_floats[4])
        bmax = (bbox_floats[5], bbox_floats[6], bbox_floats[7])

        pac_submeshes.append({
            "name": mesh_name, "material": mat_name,
            "bmin": bmin, "bmax": bmax,
            "vert_counts": vert_counts, "idx_counts": idx_counts,
        })

        # Check if next byte starts another submesh name
        if off >= s0_end - 4:
            break
        next_b = data[off]
        if next_b == 0 or next_b > 200:
            break
        peek = data[off + 1:off + 1 + min(next_b, 6)]
        if not all(32 <= c < 127 for c in peek):
            break

    if not pac_submeshes:
        return _pac_fallback_pam(data, filename)

    # ── Extract LOD0 geometry (highest quality = last data section) ──
    lod0_sec_start, lod0_sec_end = sections[-1]
    lod0_sec_size = lod0_sec_end - lod0_sec_start

    # Auto-detect vertex stride from section size:
    #   section = (total_verts × stride) + (total_indices × 2)
    #   stride = (section_size - total_indices × 2) / total_verts
    total_lod0_verts = sum(sm["vert_counts"][0] for sm in pac_submeshes)
    total_lod0_indices = sum(sm["idx_counts"][0] for sm in pac_submeshes)

    if total_lod0_verts == 0:
        return _pac_fallback_pam(data, filename)

    vert_stride = (lod0_sec_size - total_lod0_indices * 2) // total_lod0_verts
    if vert_stride < 6 or vert_stride > 128:
        logger.debug("PAC %s: computed stride %d out of range, trying PAM fallback",
                     filename, vert_stride)
        return _pac_fallback_pam(data, filename)

    lod0_split = lod0_sec_start + total_lod0_verts * vert_stride

    vert_off = lod0_sec_start
    idx_off = lod0_split

    for sm_info in pac_submeshes:
        nv = sm_info["vert_counts"][0]  # LOD0 vertex count
        ni = sm_info["idx_counts"][0]   # LOD0 index count
        bmin = sm_info["bmin"]
        bmax = sm_info["bmax"]

        # Dequantize vertex positions from uint16
        verts = []
        uvs = []
        for i in range(nv):
            rec_off = vert_off + i * vert_stride
            if rec_off + 12 > len(data):
                break
            xu, yu, zu = struct.unpack_from("<HHH", data, rec_off)
            x = bmin[0] + (xu / 65535.0) * (bmax[0] - bmin[0])
            y = bmin[1] + (yu / 65535.0) * (bmax[1] - bmin[1])
            z = bmin[2] + (zu / 65535.0) * (bmax[2] - bmin[2])
            verts.append((x, y, z))

            # UV from float16 at bytes 8-11
            try:
                u = struct.unpack_from("<e", data, rec_off + 8)[0]
                v = struct.unpack_from("<e", data, rec_off + 10)[0]
                if not math.isnan(u) and not math.isnan(v):
                    uvs.append((u, v))
                else:
                    uvs.append((0.0, 0.0))
            except Exception:
                uvs.append((0.0, 0.0))

        # Extract faces (triangle list, NOT strip)
        faces = []
        for i in range(0, ni - 2, 3):
            if idx_off + (i + 2) * 2 + 2 > len(data):
                break
            a = struct.unpack_from("<H", data, idx_off + i * 2)[0]
            b = struct.unpack_from("<H", data, idx_off + (i + 1) * 2)[0]
            c = struct.unpack_from("<H", data, idx_off + (i + 2) * 2)[0]
            if a < nv and b < nv and c < nv:
                faces.append((a, b, c))

        normals = _compute_smooth_normals(verts, faces)

        sm = SubMesh(
            name=sm_info["name"],
            material=sm_info["material"],
            texture="",
            vertices=verts,
            uvs=uvs,
            faces=faces,
            normals=normals,
            vertex_count=len(verts),
            face_count=len(faces),
        )
        result.submeshes.append(sm)

        vert_off += nv * vert_stride
        idx_off += ni * 2

    # Compute overall stats
    if result.submeshes:
        all_verts = [v for sm in result.submeshes for v in sm.vertices]
        if all_verts:
            xs = [v[0] for v in all_verts]
            ys = [v[1] for v in all_verts]
            zs = [v[2] for v in all_verts]
            result.bbox_min = (min(xs), min(ys), min(zs))
            result.bbox_max = (max(xs), max(ys), max(zs))

    result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
    result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
    result.has_uvs = any(sm.uvs for sm in result.submeshes)

    logger.info("Parsed PAC %s: %d submeshes, %d verts, %d faces",
                filename, len(result.submeshes), result.total_vertices, result.total_faces)
    return result


def _pac_fallback_pam(data: bytes, filename: str) -> ParsedMesh:
    """Fallback: try parsing PAC as PAM (works for some small PAC files)."""
    try:
        result = parse_pam(data, filename)
        if result.total_vertices > 0:
            return result
    except Exception:
        pass
    logger.debug("PAC %s: unsupported format variant, skipping", filename)
    return ParsedMesh(path=filename, format="pac")


# ── Auto-detect and parse ────────────────────────────────────────────

def parse_mesh(data: bytes, filename: str = "") -> ParsedMesh:
    """Auto-detect file type and parse accordingly."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pamlod":
        return parse_pamlod(data, filename)
    elif ext == ".pac":
        return parse_pac(data, filename)
    else:
        return parse_pam(data, filename)


def is_mesh_file(path: str) -> bool:
    """Check if a file path is a supported mesh format."""
    ext = os.path.splitext(path.lower())[1]
    return ext in (".pam", ".pamlod", ".pac")
