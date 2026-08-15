from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


class CEdge(ctypes.Structure):
    _fields_ = [("u", ctypes.c_uint32), ("v", ctypes.c_uint32), ("weight", ctypes.c_double)]


class CConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("method", ctypes.c_int),
        ("cda", ctypes.c_int),
        ("batch_size", ctypes.c_uint32),
        ("top_k", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("max_queries", ctypes.c_uint64),
        ("lambda_w", ctypes.c_double),
        ("louvain_resolution", ctypes.c_double),
        ("louvain_threshold", ctypes.c_double),
    ]


class CCommunities(ctypes.Structure):
    _fields_ = [
        ("community_count", ctypes.c_size_t),
        ("offsets", ctypes.POINTER(ctypes.c_size_t)),
        ("nodes", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CBatch(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("len", ctypes.c_size_t),
        ("records", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CStats(ctypes.Structure):
    _fields_ = [
        ("query_count", ctypes.c_uint64),
        ("discovered_matches", ctypes.c_uint64),
        ("candidate_edges", ctypes.c_uint64),
        ("community_batches", ctypes.c_uint64),
        ("current_batches", ctypes.c_uint64),
        ("community_records", ctypes.c_uint64),
        ("temperature", ctypes.c_double),
        ("finished", ctypes.c_int),
    ]


EnginePointer = ctypes.c_void_p


def _candidates() -> list[Path]:
    extension = "dylib" if platform.system() == "Darwin" else "so"
    filename = f"libperbacco.{extension}"
    package_dir = Path(__file__).resolve().parent
    configured = os.environ.get("PERBACCO_LIBRARY")
    paths = [package_dir / filename, package_dir.parents[1] / "build" / filename]
    if configured:
        paths.insert(0, Path(configured).expanduser())
    return paths


def _load() -> ctypes.CDLL:
    errors: list[str] = []
    for path in _candidates():
        if not path.is_file():
            continue
        try:
            return ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    details = f" ({'; '.join(errors)})" if errors else ""
    raise RuntimeError(
        "libperbacco was not found; run `make all` or set PERBACCO_LIBRARY" + details
    )


lib = _load()

lib.pb_config_init.argtypes = [ctypes.POINTER(CConfig)]
lib.pb_config_init.restype = None
lib.pb_engine_create.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(CEdge),
    ctypes.c_size_t,
    ctypes.POINTER(CConfig),
    ctypes.POINTER(CCommunities),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(EnginePointer),
]
lib.pb_engine_create.restype = ctypes.c_int
lib.pb_engine_create_borrowed.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(CEdge),
    ctypes.c_size_t,
    ctypes.POINTER(CConfig),
    ctypes.POINTER(CCommunities),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(EnginePointer),
]
lib.pb_engine_create_borrowed.restype = ctypes.c_int
lib.pb_engine_create_subopt.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(CEdge),
    ctypes.c_size_t,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(EnginePointer),
]
lib.pb_engine_create_subopt.restype = ctypes.c_int
lib.pb_engine_next_batch.argtypes = [
    EnginePointer,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(CBatch),
]
lib.pb_engine_next_batch.restype = ctypes.c_int
lib.pb_engine_submit_partition.argtypes = [
    EnginePointer,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
]
lib.pb_engine_submit_partition.restype = ctypes.c_int
lib.pb_engine_group_members.argtypes = [
    EnginePointer,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
lib.pb_engine_group_members.restype = ctypes.c_int
lib.pb_engine_get_stats.argtypes = [EnginePointer, ctypes.POINTER(CStats)]
lib.pb_engine_get_stats.restype = ctypes.c_int
lib.pb_engine_snapshot_size.argtypes = [EnginePointer, ctypes.POINTER(ctypes.c_size_t)]
lib.pb_engine_snapshot_size.restype = ctypes.c_int
lib.pb_engine_snapshot.argtypes = [
    EnginePointer,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
lib.pb_engine_snapshot.restype = ctypes.c_int
lib.pb_engine_restore.argtypes = [
    ctypes.c_uint32,
    ctypes.POINTER(CEdge),
    ctypes.c_size_t,
    ctypes.POINTER(CConfig),
    ctypes.POINTER(CCommunities),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(EnginePointer),
]
lib.pb_engine_restore.restype = ctypes.c_int
lib.pb_engine_last_error.argtypes = [EnginePointer]
lib.pb_engine_last_error.restype = ctypes.c_char_p
lib.pb_strerror.argtypes = [ctypes.c_int]
lib.pb_strerror.restype = ctypes.c_char_p
lib.pb_engine_free.argtypes = [EnginePointer]
lib.pb_engine_free.restype = None
