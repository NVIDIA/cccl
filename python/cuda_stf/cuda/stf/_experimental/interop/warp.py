# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp interop helpers for ``cuda.stf._experimental``.

Warp work enters a cuda.stf task graph through four names, three of which
are plain cuda.stf API:

* **scopes**: ``ctx.graph_scope()`` / ``ctx.launchable_graph_scope()``. A
  library that wants width opens a scope at its entry; the frame owner opens
  a launchable scope and replays it. A scope is a graph for its writer and one
  node for its parent, so library boundaries and graph boundaries coincide.
  :func:`scope` is the thin wrapper this module adds: the same two context
  managers plus a *scope token* (see below).
* :func:`warp_task` -- a leaf phase holding many launches: ``ctx.task(*deps)``
  with Warp's current stream pointed at the task stream, yielding
  ``(stream, arrays)`` where ``arrays`` are ``wp.array`` views of the task's
  logical-data arguments. Tokens for coarse (per-entry) dependencies, logical
  data for fine ones.
* :func:`warp_native` -- plain Warp code inside a scope that is not a phase of
  its own: ordered after every task before it and before every task after it
  in that scope. It is a task holding the scope token read-write; tasks of the
  scope read that token, so two tasks with no native block between them stay
  siblings.
* Plain Warp code inside a scope but outside both is a **record-time error**:
  while a scope is open, Warp's current stream is a poison stream whose handle
  raises on work submission. Without it the launch would run eagerly, outside
  the graph, and silently race.

Everything is a no-op without a context (``ctx is None``): the sequential
version and the task version of a library are the same code. :func:`token`
returns an inert token in that case.

Two CUDA facts drive the design. Warp's ``capture_if`` / ``capture_while``
pause the capture with ``cudaStreamEndCapture``, which CUDA accepts only on
the capture's **origin** stream: task streams of a graph scope are origins,
streams forked inside a Warp-owned capture are not, so the recording
boundary is a graph scope. And ``wp.capture_end`` returns a ``Graph`` whose
destructor destroys the captured ``cudaGraph_t``, which belongs to STF: the
adapter keeps those handles alive until :func:`release_graph_handles`.

Warp is imported lazily; importing this module does not require ``warp-lang``.
"""

from __future__ import annotations

import contextlib
import sys

from .. import AccessMode as _AccessMode

_WARP_INSTALL_HINT = (
    "This functionality requires Warp to be installed (pip install warp-lang). "
    "Install Warp or use ``ctx.task()`` directly for a raw task."
)

_graph_handles = []  # Graph objects from wp.capture_end inside tasks (see module docstring)
_scopes = {}  # id(ctx) -> stack of scope tokens

#: While a scope is open, route Warp's allocations through :class:`_ScopeAllocator`
#: (``wp.set_device_allocator``) to the STF allocator of the current level
#: (``ctx.allocate``): outside the capture, owned by the level, released at its
#: pop. A nested scope records into a CUDA child graph, which cannot hold
#: allocation nodes.
scope_allocator = (
    __import__("os").environ.get("CUDA_STF_WARP_SCOPE_ALLOCATOR", "1") == "1"
)


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
    """Return the cached ``wp.Stream`` wrapping raw ``cudaStream_t`` ``raw_ptr``
    (registering the same raw stream twice corrupts Warp's stream bookkeeping)."""
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
# tokens
# ---------------------------------------------------------------------------


class _NullToken:
    """What :func:`token` returns without a context: its deps are inert."""

    def rw(self):
        return None

    def read(self):
        return None

    def write(self):
        return None

    def set_symbol(self, symbol):
        pass


def token(ctx, symbol=None):
    """``ctx.token()``, or an inert token whose ``rw()`` / ``read()`` return
    ``None`` when ``ctx`` is ``None`` (``None`` deps are dropped by
    :func:`warp_task`), so a library can create its tokens unconditionally."""
    if ctx is None:
        return _NullToken()
    t = ctx.token()
    if symbol is not None:
        t.set_symbol(str(symbol))
    return t


# ---------------------------------------------------------------------------
# the poison stream
# ---------------------------------------------------------------------------

# Handle reads that are not work submission; everything else raises while armed.
_POISON_ALLOWED_CALLERS = frozenset(
    {
        "is_capturing",
        "__del__",
        "set_stream",
        "synchronize",
        "synchronize_device",
        "get_capture_info",
    }
)


class _PoisonStream:
    """While a scope is open and no task or native block is, Warp's current
    stream is a stream whose handle raises on work submission: a launch there
    is a record-time error instead of eager work outside the graph."""

    _stream = None  # one per process (Warp's current stream is per-device state; one recording thread)
    _armed = False
    _saved = []  # armed flags saved by enter(); restored by exit()

    @classmethod
    def _make(cls):
        wp = _import_warp()

        class Stream(wp.Stream):
            # Warp stores the handle as a plain attribute; shadow it with a property
            @property
            def cuda_stream(self):
                if _PoisonStream._armed:
                    caller = sys._getframe(1).f_code.co_name
                    if caller not in _POISON_ALLOWED_CALLERS:
                        raise RuntimeError(
                            f"Warp work ({caller}) inside a cuda.stf scope but outside "
                            "warp_task() / warp_native(): it would run eagerly, outside the graph"
                        )
                return self.__dict__.get("_cs")

            @cuda_stream.setter
            def cuda_stream(self, v):
                self.__dict__["_cs"] = v

        return Stream(wp.get_device())

    @classmethod
    def enter(cls):
        wp = _import_warp()
        if cls._stream is None:
            cls._stream = cls._make()
        cls._saved.append(cls._armed)
        cls._armed = False
        sc = wp.ScopedStream(cls._stream, sync_enter=False, sync_exit=False)
        sc.__enter__()
        cls._armed = True
        return sc

    @classmethod
    def exit(cls, sc):
        cls._armed = False  # restoring the outer stream reads the handle
        try:
            sc.__exit__(None, None, None)
        finally:
            cls._armed = cls._saved.pop()

    @classmethod
    @contextlib.contextmanager
    def suspended(cls):
        was = cls._armed
        cls._armed = False
        try:
            yield
        finally:
            cls._armed = was


# ---------------------------------------------------------------------------
# the scope allocator
# ---------------------------------------------------------------------------


class _ScopeAllocator:
    """Warp allocator installed (``wp.set_device_allocator``) while a scope is
    open. Allocations go to the **STF allocator of the current level**
    (``ctx.allocate`` / ``ctx.deallocate``): a nested graph level allocates on
    its support stream, outside the capture, so no allocation node enters the
    graph, and the level owns the memory (released at pop, handed to the
    parent until the parent's graph has launched). Nothing is hoisted or
    retained by hand.

    Lifetime rule: memory allocated inside a scope belongs to that scope's
    level. When the scope exits, every allocation of the scope that Warp has
    not freed yet is returned to the level (the release is deferred to the
    pop, after the graph has launched); a later free from Warp, e.g. a
    temporary dropped after the scope or at graph destruction, is ignored.
    With a binding that lacks ``allocate`` the fallback is an eager
    ``cudaMalloc`` retained until :func:`release_graph_handles`."""

    _ctxs = []  # (ctx, scope id) of the open scopes, innermost last
    _owner = {}  # ptr -> (ctx, scope id, size) for live STF allocations
    _live = []  # fallback path: retained pointers
    _installed = 0
    _next_id = 0

    @classmethod
    def allocate(cls, size_in_bytes):
        from cuda.bindings import runtime as rt

        size_in_bytes = int(size_in_bytes)
        ctx, sid = cls._ctxs[-1] if cls._ctxs else (None, None)
        # Warp's conditional captures resume the task stream's capture in Warp's own mode,
        # not STF's relaxed one; an allocation is an "unsafe" API there: relax around it
        _, mode = rt.cudaThreadExchangeStreamCaptureMode(
            rt.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed
        )
        try:
            if ctx is not None and hasattr(ctx, "allocate"):
                ptr = int(ctx.allocate(size_in_bytes))
                cls._owner[ptr] = (ctx, sid, size_in_bytes)
                return ptr
            err, ptr = rt.cudaMalloc(size_in_bytes)
        finally:
            rt.cudaThreadExchangeStreamCaptureMode(mode)
        if err != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"scope allocator: cudaMalloc({size_in_bytes}) failed: {err.name}"
            )
        cls._live.append(int(ptr))
        return int(ptr)

    @classmethod
    def deallocate(cls, ptr, size_in_bytes):
        rec = cls._owner.pop(int(ptr), None)
        if rec is not None and any(sid == rec[1] for _, sid in cls._ctxs):
            rec[0].deallocate(
                int(ptr), rec[2]
            )  # the level defers the release to its pop
        # otherwise: already returned at scope exit, or a fallback pointer retained until release

    @classmethod
    def install(cls, ctx):
        wp = _import_warp()
        cls._next_id += 1
        cls._ctxs.append((ctx, cls._next_id))
        if cls._installed == 0:
            wp.set_device_allocator(wp.get_device(), cls)
        cls._installed += 1

    @classmethod
    def uninstall(cls):
        wp = _import_warp()
        ctx, sid = cls._ctxs.pop()
        # return the scope's remaining allocations to its level before the pop
        for ptr, (octx, osid, size) in [
            (p, r) for p, r in cls._owner.items() if r[1] == sid
        ]:
            del cls._owner[ptr]
            if octx is not None:
                octx.deallocate(int(ptr), size)
        cls._installed -= 1
        if cls._installed == 0:
            wp.set_device_allocator(wp.get_device(), None)

    @classmethod
    def release(cls):
        from cuda.bindings import runtime as rt

        for p in cls._live:
            rt.cudaFree(p)
        cls._live.clear()


# ---------------------------------------------------------------------------
# scopes
# ---------------------------------------------------------------------------


def _scope_token(ctx):
    st = _scopes.get(id(ctx))
    return st[-1] if st else None


@contextlib.contextmanager
def scope(ctx, launchable=False, reads=(), writes=()):
    """``ctx.graph_scope()`` (``launchable=False``) or
    ``ctx.launchable_graph_scope()`` (``launchable=True``) plus the scope token.
    Yields the binding's scope object (``.launch()`` / ``.graph`` when
    launchable). No-op yielding ``None`` without a context.

    The token is created at scope entry, never lazily: a task created before
    it would not be ordered against a native block after it. A nested scope
    reads its parent's token at entry (``push(AccessMode.READ)``), so it is
    ordered after the parent's native blocks before it and stays a sibling of
    the parent's tasks. While the scope is open, Warp's current stream is the
    poison stream (see module docstring). ``reads`` / ``writes`` are tokens (or
    logical data) the scope depends on as a whole: they are pushed into the
    level at entry with that access mode, so the scope's node is ordered like a
    task declaring them (a scope with dependencies is a task whose body is a
    graph; use it around a call into code that opens its own scope).

    A scope opened *inside a running task* is a node of the parent level,
    ordered only through the tokens it imports: it is **not** ordered against
    the enclosing task's own launches before or after it (they are siblings).
    Open scopes between tasks / native blocks, never inside a task body that
    also launches work.

    >>> with scope(ctx, launchable=True) as g:
    ...     with warp_native(ctx, "collide"): example.collide()
    ...     solver.step(...)                       # opens its own scope inside
    ...     for _ in range(frames): g.launch()
    """
    if ctx is None:
        yield None
        return
    outer = _scope_token(ctx)
    cm = ctx.launchable_graph_scope() if launchable else ctx.graph_scope()
    with cm as g:
        tok = ctx.token()
        tok.set_symbol("scope")
        if outer is not None:
            outer.push(_AccessMode.READ)
        for d in reads:
            if d is not None:
                d.push(_AccessMode.READ)
        for d in writes:
            if d is not None:
                d.push(_AccessMode.RW)
        _scopes.setdefault(id(ctx), []).append(tok)
        poison = _PoisonStream.enter()
        if scope_allocator:
            _ScopeAllocator.install(ctx)
        try:
            yield g
        finally:
            if scope_allocator:
                _ScopeAllocator.uninstall()
            _PoisonStream.exit(poison)
            st = _scopes[id(ctx)]
            st.pop()
            if not st:
                del _scopes[id(ctx)]


# ---------------------------------------------------------------------------
# tasks and native blocks
# ---------------------------------------------------------------------------


def warp_task(ctx, *args, symbol=None, capture="auto", dtypes=None):
    """Context manager: ``ctx.task(*args)`` with Warp's current stream set to
    the task stream; yields ``(stream, arrays)``. Inside a scope the task also
    reads the scope token. ``None`` deps (from an inert :func:`token`) are
    dropped; with ``ctx=None`` the body runs as is on Warp's current stream and
    ``arrays`` is empty.

    ``capture``: ``"auto"`` (default) adopts the capture as an external Warp
    capture when the task stream is capturing (graph scopes), so Warp's
    allocator, ``is_capturing`` checks and conditional captures see it;
    ``"never"`` issues no Warp capture calls; ``"always"`` requires a capturing
    stream.

    >>> with warp_task(ctx, lX.read(), lY.rw()) as (s, (x, y)):
    ...     wp.launch(axpy, dim=x.shape[0], inputs=[2.0, x, y])   # on the task stream
    """
    wp = _import_warp()
    if capture not in ("auto", "never", "always"):
        raise ValueError("capture must be 'auto', 'never' or 'always'")
    deps = [a for a in args if a is not None]
    if ctx is None:
        return _no_context_task(wp)
    tok = _scope_token(ctx)
    if tok is not None:
        deps.append(tok.read())
    t = ctx.task(*deps, symbol=symbol) if symbol is not None else ctx.task(*deps)

    class _WarpTaskContext:
        _scope = None
        _captured = False
        _suspend = None

        def __enter__(self):
            self._suspend = _PoisonStream.suspended()
            self._suspend.__enter__()
            try:
                t.start()
            except BaseException:
                self._suspend.__exit__(None, None, None)
                raise
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
                self._suspend.__exit__(None, None, None)
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
            self._suspend.__exit__(None, None, None)
            if exc_type is not None:
                return False
            if errors:
                raise errors[0]
            return False

    return _WarpTaskContext()


@contextlib.contextmanager
def _no_context_task(wp):
    yield wp.get_stream(), ()


@contextlib.contextmanager
def warp_native(ctx, symbol="native"):
    """Plain Warp code inside a scope: a task holding the scope token
    read-write, Warp's current stream on its stream; yields the stream. Ordered
    after every task of the scope before it and before every task after it.
    Raises if no scope is open on ``ctx``. No-op without a context.

    >>> with warp_native(ctx, "integrate"):
    ...     example.integrate()      # any number of launches, copies, allocations
    """
    if ctx is None:
        yield _import_warp().get_stream()
        return
    tok = _scope_token(ctx)
    if tok is None:
        raise RuntimeError(
            "warp_native() needs an open scope on this context (see scope())"
        )
    with warp_task(ctx, tok.rw(), symbol=symbol) as (s, _):
        yield s


def release_graph_handles():
    """Drop the Warp ``Graph`` handles kept alive by :func:`warp_task` and free
    the buffers of the scope allocator. Call only after every STF context that
    captured through them is finalized."""
    _graph_handles.clear()
    _ScopeAllocator.release()


__all__ = [
    "array_arg",
    "release_graph_handles",
    "scope",
    "token",
    "warp_arguments",
    "warp_native",
    "warp_task",
    "wrap_stream",
]
