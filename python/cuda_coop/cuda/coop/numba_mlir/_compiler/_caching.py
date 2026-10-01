# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cache results on disk by function identity and serialized arguments.

``CUDA_COOP_ENABLE_CACHE`` enables persistent caching when set to ``"1"`` or
any value other than empty, ``"0"``, ``"false"``, ``"no"``, or ``"off"``.
Comparison ignores surrounding whitespace and letter case. Unset disables the
disk cache. This flag does not control the separate in-memory LRU cache on
NVRTC compilation.

``XDG_CACHE_HOME`` selects the cache parent directory on non-Windows systems;
the default is ``~/.cache``. ``LOCALAPPDATA`` selects it on Windows; the
default is ``~/AppData/Local``. Only the variable for the current platform is
read. Its value must be an absolute path: unset, empty, and relative values
use the platform default. Cache files live under a ``cccl`` subdirectory,
partitioned further by callable identity.

The enable flag and selected cache location are captured when this module is
imported. Set them before backend initialization imports this module; later
environment changes do not reconfigure its wrappers. Directory or write
``OSError`` failures disable disk caching for the rest of the process, while
unreadable entries simply miss. Compilation proceeds without persistent
caching in those cases.
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


def _cache_location():
    if os.name == "nt":
        cache_home = os.environ.get("LOCALAPPDATA", "")
        fallback = ("AppData", "Local")
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME", "")
        fallback = (".cache",)
    if not os.path.isabs(cache_home):
        cache_home = os.path.join(os.path.expanduser("~"), *fallback)
    return os.path.join(cache_home, "cccl")


_CACHE_LOCATION = _cache_location()


def _json_cache_key(value):
    """Convert supported key values into a tagged JSON-serializable tree.

    Plain JSON loses distinctions such as tuples versus lists and cannot
    encode bytes. Tag these containers, preserve a tuple subclass's qualified
    type name, and encode bytes as base64. Represent dictionaries as sorted
    key/value pairs so supported non-string keys can participate without JSON
    coercing them to strings. Scalar values retain JSON's native encoding.

    Parameters
    ----------
    value : object
        A scalar (``None``, bool, int, float, or str), bytes, or a recursively
        supported tuple, list, or dictionary. Dictionary entries are ordered
        by ``repr`` of their keys. Recursive containers are not supported.

    Returns
    -------
    object
        JSON-compatible representation used to hash arguments. This is a key
        encoding, not a general-purpose serialization format for cache values.

    Raises
    ------
    TypeError
        A value has no supported key representation. The disk-cache wrapper
        treats this as a request to compute without caching.
    """

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
                for key, item in sorted(
                    value.items(), key=lambda entry: repr(entry[0])
                )
            ],
        }
    raise TypeError(f"Unsupported disk cache key value: {value!r}")


def json_hash(*args, **kwargs):
    """Hash positional and keyword arguments using the current cache schema.

    Prefix the serialized argument tree with the schema version so changes to
    the cache format invalidate older entries. Keyword names are part of the
    key, and their insertion order does not affect it. This function does not
    add callable identity; ``disk_cache`` passes that identity as an argument.

    Parameters
    ----------
    *args : object
        Positional key components accepted by ``_json_cache_key``.
    **kwargs : object
        Named key components with the same serialization restrictions.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the versioned argument representation.

    Raises
    ------
    TypeError
        An argument cannot be represented by the cache key encoder.
    """

    hasher = hashlib.sha256()
    hasher.update(f"v{_CACHE_SCHEMA_VERSION}:".encode())
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
    """Read one cache entry, treating unusable files as cache misses.

    Validate the schema version and the decoded value's recorded type before
    returning it. This catches stale formats and values whose JSON round trip
    changes their top-level type. It does not authenticate cached data or
    validate the value against a particular compiler invocation.

    Parameters
    ----------
    path : str or path-like
        JSON cache entry to open. The function does not modify or remove it.

    Returns
    -------
    object
        Decoded value on success, otherwise the unique ``_CACHE_MISS`` sentinel
        for an unreadable, malformed, stale, or type-inconsistent entry.
        ``None`` is a valid cached result and is distinct from a miss.
    """

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
    """Serialize a result and atomically replace its cache entry.

    Write a schema and top-level type tag alongside the value, encoding bytes
    as base64. Use a temporary file in the destination directory, flush and
    fsync it, then replace the destination so readers do not see a partially
    written JSON document. Concurrent writers may replace the same entry.
    On a write or replacement failure, attempt to remove the temporary file
    and propagate the error for ``disk_cache`` to handle.

    Parameters
    ----------
    path : str or path-like
        Destination entry. Its parent directory must already exist.
    value : object
        Result to persist: bytes or a JSON-serializable value. A later read
        also requires the decoded top-level type to match the saved type tag.

    Raises
    ------
    OSError
        Creating, writing, syncing, or replacing the entry fails.
    TypeError or ValueError
        The result cannot be serialized as a cache entry.
    """

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
    """Decorate a computation with best-effort persistent result caching.

    Cache files are namespaced by the callable's module and qualified name;
    the entry key includes that identity, arguments, and the cache schema.
    Callers must therefore include every compilation input that affects the
    result in the arguments. Callable source changes do not invalidate entries
    by themselves. The enable flag is read from ``CUDA_COOP_ENABLE_CACHE``
    when this module is imported.

    Unsupported key encodings bypass the cache, and unsupported result
    encodings skip the write. Directory or write ``OSError`` failures disable
    caching for every wrapper in this module for the rest of the process;
    unreadable entries simply miss. Exceptions from the wrapped computation
    propagate without being cached. There is no lock around computation, so
    concurrent misses can compute the same result before atomic publication.

    Parameters
    ----------
    func : callable
        Computation whose result can be reused for the same serialized inputs.
        Arguments must be accepted by ``_json_cache_key`` to use the cache;
        results must round-trip through the cache value encoding.

    Returns
    -------
    callable
        Wrapper preserving ``func`` metadata and forwarding its arguments.
        Returns a decoded cache hit or the newly computed result.
    """

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
