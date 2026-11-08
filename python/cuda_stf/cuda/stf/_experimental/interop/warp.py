# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp interop helpers for ``cuda.stf._experimental``.

This module provides:

* :func:`wrap_stream` -- one cached ``wp.Stream`` per raw ``cudaStream_t``
  (registering the same raw stream twice corrupts Warp's stream bookkeeping).
* :func:`array_arg` and :func:`warp_arguments` -- alias STF task arguments as
  ``wp.array`` views (no copy) through their ``ptr``/``shape``/``dtype``.
* :func:`warp_task` -- context manager that opens an STF task, makes the task
  stream Warp's current stream (no enter/exit synchronisation) and, when that
  stream is under capture, lets Warp adopt the capture
  (``wp.capture_begin(external=True)``) so Warp's allocator, ``is_capturing``
  checks and conditional captures (``capture_if`` / ``capture_while``) see it.
  Yields ``(stream, arrays)``.
* :func:`local_context` -- a stream-backend context bound to Warp's current
  stream, for library code whose caller may already be capturing (e.g. inside
  ``wp.ScopedCapture``): the library's tasks become parallel branches of the
  caller's graph without changing the caller.
* :class:`WarpTaskGraph` -- record a frame once into a stackable context and
  replay it, with a catch-all "frame" task around the recorded body so Warp
  work issued outside explicit tasks is captured too; library code adds its
  own tasks at graph level with :meth:`WarpTaskGraph.suspend_frame` /
  :meth:`WarpTaskGraph.resume_frame` (tasks are never nested) and finds the
  recording graph through :func:`current`.

Two facts drive the design. Warp's ``capture_if`` / ``capture_while`` pause
the capture with ``cudaStreamEndCapture``, which CUDA accepts only on the
capture's **origin** stream: a task placed directly on a context gets such a
stream, a task of a :func:`local_context` runs on a fork of the caller's
stream and cannot host them. And ``wp.capture_end`` returns a ``Graph`` whose
destructor destroys the captured ``cudaGraph_t``, which belongs to STF: the
adapter keeps those handles alive until :func:`release_graph_handles`.

Warp is imported lazily; importing this module does not require ``warp-lang``.
"""

from __future__ import annotations

import contextlib

from .. import context as _context
from .. import stackable_context as _stackable_context

_WARP_INSTALL_HINT = (
    "This functionality requires Warp to be installed (pip install warp-lang). "
    "Install Warp or use ``ctx.task()`` directly for a raw task."
)

_graph_handles = []  # Graph objects from wp.capture_end inside tasks (see module docstring)
_recording = []  # WarpTaskGraph stack while recording


def _import_warp():
    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError(_WARP_INSTALL_HINT) from exc
    return wp


# ---------------------------------------------------------------------------
# streams and arrays
# ---------------------------------------------------------------------------

_wp_stream_cache = {}


def _device_cache_key(device):
    return getattr(device, "ordinal", str(device))


def wrap_stream(raw_ptr, device=None):
    """Return the cached ``wp.Stream`` wrapping raw ``cudaStream_t`` ``raw_ptr``."""
    wp = _import_warp()
    device = wp.get_device(device) if device is not None else wp.get_cuda_device()
    key = (_device_cache_key(device), int(raw_ptr))
    s = _wp_stream_cache.get(key)
    if s is None:
        s = wp.Stream(device, cuda_stream=int(raw_ptr))
        _wp_stream_cache[key] = s
    return s


_TYPESTR_TO_WP = None


def _wp_dtype_from_typestr(typestr):
    global _TYPESTR_TO_WP
    wp = _import_warp()
    if _TYPESTR_TO_WP is None:
        _TYPESTR_TO_WP = {
            "<f4": wp.float32,
            "<f8": wp.float64,
            "<f2": wp.float16,
            "<i1": wp.int8,
            "<i2": wp.int16,
            "<i4": wp.int32,
            "<i8": wp.int64,
            "<u1": wp.uint8,
            "<u2": wp.uint16,
            "<u4": wp.uint32,
            "<u8": wp.uint64,
            "|i1": wp.int8,
            "|u1": wp.uint8,
            "|b1": wp.bool,
        }
    try:
        return _TYPESTR_TO_WP[typestr]
    except KeyError as exc:
        raise TypeError(
            f"no Warp scalar dtype for CUDA array interface typestr {typestr!r}; pass dtype="
        ) from exc


def array_arg(task, index, dtype=None, shape=None, device=None):
    """Alias task argument ``index`` as a ``wp.array`` (no copy, no ownership).

    ``wp.array(data=...)`` rejects objects that only expose the CUDA Array
    Interface, so the view is built from ``ptr``. ``dtype`` defaults to the
    scalar type of the argument; pass a Warp vector/matrix type and ``shape``
    to reinterpret. The view is valid for the duration of the task.
    """
    wp = _import_warp()
    cai = task.get_arg_cai(index)
    iface = cai.__cuda_array_interface__
    if dtype is None:
        dtype = _wp_dtype_from_typestr(iface["typestr"])
    return wp.array(
        ptr=int(iface["data"][0]),
        dtype=dtype,
        shape=tuple(iface["shape"]) if shape is None else tuple(shape),
        device=device if device is not None else wp.get_device(),
    )


def warp_arguments(task, dtypes=None):
    """All task buffer arguments as ``wp.array`` views, as a tuple (empty for
    a token-only task). ``dtypes`` optionally gives one dtype per argument."""
    out = task.args_cai()
    if out is None:
        return ()
    n = len(out) if isinstance(out, tuple) else 1
    dtypes = dtypes or (None,) * n
    return tuple(array_arg(task, i, dtypes[i]) for i in range(n))


# ---------------------------------------------------------------------------
# the task scope
# ---------------------------------------------------------------------------


def warp_task(ctx, *args, symbol=None, capture="auto", dtypes=None):
    """Context manager: ``ctx.task(*args)`` with Warp's current stream set to
    the task stream; yields ``(stream, arrays)``.

    ``capture``: ``"auto"`` (default) adopts the capture as an external Warp
    capture when the task stream is capturing (STF graph scopes, stackable
    record); ``"never"`` issues no Warp capture calls (use for a
    :func:`local_context` under a Warp-owned ``wp.ScopedCapture``: Warp keys
    its capture registry by capture id, shared with the outer capture, and a
    nested ``capture_end`` would unregister the caller's capture); ``"always"``
    requires a capturing stream.

    Example
    -------
    >>> from cuda.stf._experimental.interop.warp import warp_task
    >>> with warp_task(ctx, lX.read(), lY.rw()) as (s, (x, y)):
    ...     wp.launch(axpy, dim=x.shape[0], inputs=[2.0, x, y])   # on the task stream
    """
    wp = _import_warp()
    if capture not in ("auto", "never", "always"):
        raise ValueError("capture must be 'auto', 'never' or 'always'")
    t = ctx.task(*args, symbol=symbol) if symbol is not None else ctx.task(*args)

    class _WarpTaskContext:
        _scope = None
        _captured = False

        def __enter__(self):
            t.start()
            try:
                self.stream = wrap_stream(t.stream_ptr())
                self._scope = wp.ScopedStream(
                    self.stream, sync_enter=False, sync_exit=False
                )
                self._scope.__enter__()
                if capture == "always" or (
                    capture == "auto" and self.stream.is_capturing
                ):
                    wp.capture_begin(stream=self.stream, external=True)
                    self._captured = True
                arrays = warp_arguments(t, dtypes)
            except BaseException as e:
                if self._captured:
                    with contextlib.suppress(Exception):
                        _graph_handles.append(wp.capture_end(stream=self.stream))
                if self._scope is not None:
                    with contextlib.suppress(Exception):
                        self._scope.__exit__(type(e), e, e.__traceback__)
                with contextlib.suppress(Exception):
                    t.end()
                raise
            return self.stream, arrays

        def __exit__(self, exc_type, exc_val, exc_tb):
            # same precedence as pytorch_task: body error, then cleanup errors
            errors = []
            if self._captured:
                try:
                    _graph_handles.append(wp.capture_end(stream=self.stream))
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
            if self._scope is not None:
                try:
                    self._scope.__exit__(exc_type, exc_val, exc_tb)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
            try:
                t.end()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            if exc_type is not None:
                return False
            if errors:
                raise errors[0]
            return False

    return _WarpTaskContext()


def release_graph_handles():
    """Drop the Warp ``Graph`` handles kept alive by :func:`warp_task`. Call
    only after every STF context that captured through them is finalized."""
    _graph_handles.clear()


# ---------------------------------------------------------------------------
# named tokens, local contexts, record-once graphs
# ---------------------------------------------------------------------------


class _Tokens:
    """``tokens[name]`` creates a token on first use and reuses it after."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._tokens = {}

    def __getitem__(self, name):
        t = self._tokens.get(name)
        if t is None:
            t = self._ctx.token()
            t.set_symbol(str(name))
            self._tokens[name] = t
        return t


class _TaskHost:
    """Shared surface of :class:`LocalContext` and :class:`WarpTaskGraph`:
    ``ctx``, ``tokens``, ``task()``, ``frame_token``, ``suspend_frame()``."""

    _capture = "auto"
    frame_token = None

    def task(self, *deps, symbol=None):
        return warp_task(self.ctx, *deps, symbol=symbol, capture=self._capture)

    def suspend_frame(self):
        return False

    def resume_frame(self, was_open):
        pass


class LocalContext(_TaskHost):
    """A stream-backend context bound to Warp's current stream (see
    :func:`local_context`)."""

    _capture = "never"

    def __init__(self, stream=None, capture="never"):
        wp = _import_warp()
        s = stream if stream is not None else wp.get_stream()
        self.ctx = _context(stream=int(s.cuda_stream))
        self.tokens = _Tokens(self.ctx)
        self._capture = capture

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ctx.finalize()
        return False


def local_context(stream=None, capture="never"):
    """Open a context whose tasks fork from Warp's current stream and join it
    again on exit; works eagerly and inside a caller's ``wp.ScopedCapture``.

    Tasks run on forked streams, so bodies must not use Warp's conditional
    captures (``capture_if`` / ``capture_while``: CUDA pauses a capture only on
    its origin stream) and, with ``capture="never"`` (the default, required
    under a Warp-owned capture with stock Warp), Warp allocations inside the
    tasks are not capture-aware. For those bodies use :class:`WarpTaskGraph`.

    >>> with local_context() as lc:
    ...     with lc.task(lc.tokens["x"].rw(), symbol="k1"): wp.launch(k1, ...)
    ...     with lc.task(lc.tokens["x"].read(), lc.tokens["y"].rw()): wp.launch(k2, ...)
    ...     with lc.task(lc.tokens["x"].read(), lc.tokens["z"].rw()): wp.launch(k3, ...)   # parallel to k2
    """
    return LocalContext(stream, capture)


def current():
    """The :class:`WarpTaskGraph` currently recording, or ``None``."""
    return _recording[-1] if _recording else None


class WarpTaskGraph(_TaskHost):
    """Record a frame of Warp work once as an STF graph, replay it many times.

    >>> g = WarpTaskGraph()
    >>> graph = g.record(example.simulate)     # instead of wp.ScopedCapture
    >>> for _ in range(frames): graph.launch()
    >>> g.finalize()

    During :meth:`record` the body runs inside a catch-all task ("frame") so
    launches outside explicit tasks are captured; library code that wants its
    own tasks at graph level (each task stream is a capture origin, so Warp
    conditional captures work there) calls :meth:`suspend_frame` before and
    :meth:`resume_frame` after, and orders itself against the frame through
    ``frame_token`` (read/write it in its barrier tasks).
    """

    def __init__(self):
        self.ctx = _stackable_context()
        self.tokens = _Tokens(self.ctx)
        self.frame_token = self.tokens["frame"]
        self._frame = None
        self._graph = None

    # frame task -----------------------------------------------------------
    def _open_frame(self):
        self._frame = self.task(self.frame_token.rw(), symbol="frame")
        self._frame.__enter__()

    def _close_frame(self):
        f, self._frame = self._frame, None
        f.__exit__(None, None, None)

    def suspend_frame(self):
        """Close the catch-all frame task if it is open; returns whether it was."""
        if self._frame is None:
            return False
        self._close_frame()
        return True

    def resume_frame(self, was_open):
        if was_open:
            self._open_frame()

    # record / replay --------------------------------------------------------
    def record(self, body, frame_task=True):
        """Record ``body()`` once; returns a launchable graph (``.launch()``)."""
        self.ctx.push()
        _recording.append(self)
        try:
            if frame_task:
                self._open_frame()
            try:
                body()
            finally:
                if self._frame is not None:
                    self._close_frame()
        finally:
            _recording.pop()
        self._graph = self.ctx.pop_prologue_shared()
        return self._graph

    def launch(self):
        self._graph.launch()

    def finalize(self):
        self._graph = None
        self.ctx.finalize()


__all__ = [
    "WarpTaskGraph",
    "array_arg",
    "current",
    "local_context",
    "release_graph_handles",
    "warp_arguments",
    "warp_task",
    "wrap_stream",
]
