# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Persistent content-addressed cache helpers for Numba-CUDA-MLIR.

The existing schema and key bytes are intentionally preserved across this
module move.  Persistence is separate from semantic operation identities.
"""

import hashlib
import json
import os
import tempfile
from base64 import b64decode, b64encode
from functools import wraps

_FALSE_CACHE_VALUES = frozenset(("", "0", "false", "no", "off"))
_CACHE_ENV_VALUE = os.environ.get("CUDA_COOP_ENABLE_CACHE")
_ENABLE_CACHE = (
    _CACHE_ENV_VALUE is not None
    and _CACHE_ENV_VALUE.strip().lower() not in _FALSE_CACHE_VALUES
)
_CACHE_USABLE = _ENABLE_CACHE
_CACHE_SCHEMA_VERSION = 5
_CACHE_MISS = object()
_CACHE_LOCATION = os.path.join(os.path.expanduser("~"), ".cache", "cccl")


def _json_cache_key(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "__cuda_coop_numba_mlir_cache_type__": "bytes",
            "data": b64encode(value).decode("ascii"),
        }
    if isinstance(value, tuple):
        value_type = f"{type(value).__module__}.{type(value).__qualname__}"
        return {
            "__cuda_coop_numba_mlir_cache_type__": value_type,
            "items": [_json_cache_key(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "__cuda_coop_numba_mlir_cache_type__": "builtins.list",
            "items": [_json_cache_key(item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "__cuda_coop_numba_mlir_cache_type__": "builtins.dict",
            "items": [
                (_json_cache_key(key), _json_cache_key(item))
                for key, item in sorted(value.items(), key=lambda entry: repr(entry[0]))
            ],
        }
    raise TypeError(f"Unsupported disk cache key value: {value!r}")


def json_hash(*args, **kwargs):
    hasher = hashlib.sha256()
    hasher.update(f"v{_CACHE_SCHEMA_VERSION}:".encode("utf-8"))
    payload = json.dumps(
        _json_cache_key((args, kwargs)),
        separators=(",", ":"),
        sort_keys=True,
    )
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


def _cache_identity_path(cache_identity):
    identity_hash = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
    path = os.path.join(_CACHE_LOCATION, identity_hash)
    os.makedirs(path, exist_ok=True)
    return path


def _cache_value_type(value):
    if isinstance(value, bytes):
        return "bytes"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _encode_cache_value(value):
    if isinstance(value, bytes):
        return {
            "__cuda_coop_numba_mlir_cache_type__": "bytes",
            "data": b64encode(value).decode("ascii"),
        }
    return value


def _decode_cache_value(value):
    if (
        isinstance(value, dict)
        and value.get("__cuda_coop_numba_mlir_cache_type__") == "bytes"
    ):
        return b64decode(value["data"].encode("ascii"))
    return value


def _read_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            return _CACHE_MISS
        if cached.get("version") != _CACHE_SCHEMA_VERSION:
            return _CACHE_MISS
        value = _decode_cache_value(cached["value"])
        if cached.get("value_type") != _cache_value_type(value):
            return _CACHE_MISS
        return value
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _CACHE_MISS


def _write_cache(path, value):
    cached = {
        "version": _CACHE_SCHEMA_VERSION,
        "value_type": _cache_value_type(value),
        "value": _encode_cache_value(value),
    }
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        dir=os.path.dirname(path),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cached, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def disk_cache(func):
    cache_identity = f"{func.__module__}.{func.__qualname__}"

    @wraps(func)
    def cacher(*args, **kwargs):
        global _CACHE_USABLE

        if not _CACHE_USABLE:
            return func(*args, **kwargs)

        try:
            key = json_hash(cache_identity, *args, **kwargs)
            path = os.path.join(_cache_identity_path(cache_identity), key)
        except (TypeError, ValueError):
            return func(*args, **kwargs)
        except OSError:
            _CACHE_USABLE = False
            return func(*args, **kwargs)

        if os.path.isfile(path):
            cached = _read_cache(path)
            if cached is not _CACHE_MISS:
                return cached

        result = func(*args, **kwargs)
        try:
            _write_cache(path, result)
        except (TypeError, ValueError):
            pass
        except OSError:
            _CACHE_USABLE = False
        return result

    return cacher
