# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp interop helpers for ``cuda.stf._experimental``.

Warp work enters a cuda.stf task graph as ordinary cuda.stf: tasks, scopes,
and tokens or logical data for the order between tasks. This module adds no
ordering rule of its own.

* :func:`warp_task` -- ``ctx.task(*deps)`` with Warp's current stream pointed
  at the task stream, yielding ``(stream, arrays)`` where ``arrays`` are
  ``wp.array`` views of the task's logical-data arguments. A task that
  declares nothing is independent of every other task, as in cuda.stf.
* :func:`scope` -- ``ctx.graph_scope()`` / ``ctx.launchable_graph_scope()``.
  Inside, a library sees a flat context; for its parent the scope is one node,
  ordered by the ``reads`` / ``writes`` it is given.
* :func:`token` -- ``ctx.token()``. When Warp owns the arrays there is no
  logical data to declare, and tokens carry the order.
* :func:`scope_token` -- a token of the current scope, created the first time
  it is asked for and the same for the life of the scope. A convenience for
  "the sequence of this scope": tasks that take it read-write are ordered
  against every task that takes it, tasks that read it may overlap.

Inside a scope **every Warp launch belongs to a task**. Plain Warp code inside
a scope but outside a task is a record-time error: Warp's current stream is
then a poison stream whose handle raises on work submission. Without it the
launch would run eagerly, outside the graph, and silently race.

The context argument is optional. :func:`scope` given a context makes it the
current one while the scope is open; :func:`warp_task`, :func:`scope`,
:func:`token` and :func:`scope_token` called without a context use the current
one (:func:`current_context`). Called that way they do nothing when there is
no current context (the code is not being recorded) and when a task is already
open (the code runs as part of that task), so a library written with them
also runs unchanged from callers that do not record, or that wrap the whole
call in one task:

>>> def step(self):                              # a library
...     s = scope_token()
...     with warp_task(s.rw()): self.prepare()
...     with warp_task(s.read(), symbol="mass"): self.mass()
...     with warp_task(s.read(), symbol="rows"): self.rows()
...     with warp_task(s.rw()): self.solve()
>>> with scope(ctx, launchable=True) as graph:   # the code that records
...     solver.step()
>>> graph.launch()

A library that records tasks is called from a scope, not from inside a task;
its caller gives it a scope of its own (``scope(reads=..., writes=...)``)
where its sequence must not be the caller's.

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
_scopes = []  # the open scopes, innermost last: [ctx, symbol, scope token or None]
_open_tasks = []  # the warp_task blocks currently open (_Task), innermost last


def current_context():
    """The context of the innermost open :func:`scope`, or ``None`` when no
    scope is open (nothing is being recorded)."""
    return _scopes[-1][0] if _scopes else None


def current_task_symbol():
    """Symbol of the innermost open :func:`warp_task` block, or ``None``. For
    tools that attribute Warp events (a conditional capture, an allocation) to
    the task that recorded them."""
    return _open_tasks[-1].symbol if _open_tasks else None


def _is_context(obj):
    return hasattr(obj, "task") and hasattr(obj, "logical_data")


def _in_task(ctx):
    return bool(_open_tasks) and _open_tasks[-1].ctx is ctx


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


def get_arg_warp(task, index, dtype=None, shape=None, device=None):
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
    return tuple(get_arg_warp(task, i, dtypes[i]) for i in range(n))


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


class _NullToken:
    """What :func:`token` and :func:`scope_token` return when nothing is
    recorded: its deps are ``None``, which :func:`warp_task` drops."""

    def rw(self):
        return None

    def read(self):
        return None

    def write(self):
        return None

    def set_symbol(self, symbol):
        pass


def token(*args, symbol=None):
    """``ctx.token()``: ``token(ctx)``, ``token(ctx, "name")``, or
    ``token("name")`` / ``token()`` on the current context. Without a context
    the token is inert (its ``rw()`` / ``read()`` return ``None``, which
    :func:`warp_task` drops), so a library creates its tokens unconditionally.
    """
    if args and (args[0] is None or _is_context(args[0])):
        ctx, args = args[0], args[1:]
    else:
        ctx = None if _open_tasks else current_context()
    if args:
        (symbol,) = args
    if ctx is None:
        return _NullToken()
    t = ctx.token()
    if symbol is not None:
        t.set_symbol(str(symbol))
    return t


def scope_token():
    """A token of the current scope: created the first time it is asked for,
    the same object until the scope closes. Tasks that take it ``rw()`` are
    ordered against every task of the scope that takes it; tasks that take it
    ``read()`` may overlap each other. It is an ordinary token: a shortcut for
    creating one at scope entry and handing it to every function called from
    the scope. Inert without an open scope and inside a task.
    """
    if not _scopes or _open_tasks:
        return _NullToken()
    s = _scopes[-1]
    if s[2] is None:
        s[2] = s[0].token()
        s[2].set_symbol(f"scope:{s[1]}" if s[1] is not None else "scope")
    return s[2]


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
    """While a scope is open and no task is, Warp's current
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
                            "warp_task(): it would run eagerly, outside the graph"
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
    open. Allocations go to the **STF allocator of the current scope**
    (``stackable_context.allocate`` / ``deallocate``): a nested scope allocates
    on its support stream, outside the capture, so no allocation node enters
    the graph (a CUDA child graph cannot hold one), and the scope owns the
    memory (released at pop, handed to the parent until the parent's graph has
    launched). Nothing is hoisted or retained by hand.

    Only allocations made while the current stream is capturing go to the
    scope (the native hook of the Warp patch has the same rule); any other
    allocation while the allocator is installed goes to Warp's own allocator.

    Lifetime rule: memory allocated under capture inside a scope belongs to
    that scope. When the scope exits, every allocation of the scope that Warp
    has not freed yet is returned to it (the release is deferred to the pop,
    after the graph has launched); a later free from Warp, e.g. a temporary
    dropped after the scope or at graph destruction, is ignored."""

    _ctxs = []  # (ctx, scope id) of the open scopes, innermost last
    _owner = {}  # ptr -> (ctx, scope id, size) for live STF allocations
    _builtin = {}  # ptr -> Warp's own allocator, for allocations made outside a capture
    _installed = 0
    _next_id = 0
    stats = {
        "count": 0,
        "bytes": 0,
        "max_size": 0,
    }  # scope allocations since the last reset (measurement aid)

    @classmethod
    def allocate(cls, size_in_bytes):
        from cuda.bindings import runtime as rt

        wp = _import_warp()
        size_in_bytes = int(size_in_bytes)
        # Only an allocation made while recording (the current stream is capturing) is
        # the scope's business: it would otherwise be an allocation node in the graph,
        # forbidden in a child graph and costly in a wide one. Anything else inside a
        # scope (eager work between recordings, a wp.empty outside any block) goes to
        # Warp's own allocator and lives as long as the Python object.
        if not cls._ctxs or not wp.get_stream().is_capturing:
            builtin = wp.get_device().current_allocator
            ptr = int(builtin.allocate(size_in_bytes))
            cls._builtin[ptr] = builtin
            return ptr
        ctx, sid = cls._ctxs[-1]
        # Warp's conditional captures resume the task stream's capture in Warp's own mode,
        # not STF's relaxed one; an allocation is an "unsafe" API there: relax around it.
        # Uncached: the scope's own allocator defers every release to the pop, which is
        # what a temporary freed while the scope records needs.
        _, mode = rt.cudaThreadExchangeStreamCaptureMode(
            rt.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed
        )
        try:
            ptr = int(ctx.allocate(size_in_bytes))
        finally:
            rt.cudaThreadExchangeStreamCaptureMode(mode)
        cls._owner[ptr] = (ctx, sid, size_in_bytes)
        cls.stats["count"] += 1
        cls.stats["bytes"] += size_in_bytes
        cls.stats["max_size"] = max(cls.stats["max_size"], size_in_bytes)
        return ptr

    @classmethod
    def deallocate(cls, ptr, size_in_bytes):
        builtin = cls._builtin.pop(int(ptr), None)
        if builtin is not None:
            builtin.deallocate(int(ptr), int(size_in_bytes))
            return
        rec = cls._owner.pop(int(ptr), None)
        if rec is not None and any(sid == rec[1] for _, sid in cls._ctxs):
            rec[0].deallocate(
                int(ptr), rec[2]
            )  # the scope defers the release to its pop
        # otherwise: already returned at scope exit

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
        # a scope torn down at interpreter shutdown finds Warp already gone: skip the allocator reset then
        wp = sys.modules.get("warp")
        ctx, sid = cls._ctxs.pop()
        # return the scope's remaining allocations to it before the pop
        for ptr, (octx, osid, size) in [
            (p, r) for p, r in cls._owner.items() if r[1] == sid
        ]:
            del cls._owner[ptr]
            if octx is not None:
                octx.deallocate(int(ptr), size)
        cls._installed -= 1
        if cls._installed == 0 and wp is not None:
            wp.set_device_allocator(wp.get_device(), None)


# ---------------------------------------------------------------------------
# scopes
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def scope(ctx=None, launchable=False, reads=(), writes=(), symbol=None):
    """``ctx.graph_scope()`` (``launchable=False``) or
    ``ctx.launchable_graph_scope()`` (``launchable=True``). Yields the binding's
    scope object (``.launch()`` / ``.graph`` when launchable).

    ``reads`` / ``writes`` are tokens (or logical data) of the enclosing scope
    that the scope depends on as a whole: they are pushed into the scope at
    entry with that access mode, so the scope's node is ordered like a task
    declaring them, and tasks inside the scope can use them. A scope that is
    given nothing is independent of the other tasks of its parent. ``symbol``
    names the scope (STF DOT output, diagnostics).

    While the scope is open, ``ctx`` is the current context
    (:func:`current_context`) and Warp's current stream is the poison stream
    (see module docstring). Without ``ctx`` the scope is opened on the current
    context; it is a no-op yielding ``None`` when there is none, and inside a
    task, where the code runs as part of that task.

    >>> with scope(ctx, launchable=True) as graph:
    ...     s = scope_token()
    ...     with warp_task(s.rw(), symbol="collide"): example.collide()
    ...     with scope(writes=(s,), symbol="step"): solver.step(...)  # records its own tasks
    ...     for _ in range(frames): graph.launch()
    """
    explicit = ctx is not None
    if not explicit:
        ctx = current_context()
    if ctx is None:
        yield None
        return
    if _open_tasks and not explicit:
        yield None  # the code runs as part of the open task
        return
    if _in_task(ctx):
        raise RuntimeError(
            f"scope {symbol!r} opened inside task {_open_tasks[-1].symbol!r}: it would "
            "not be ordered against that task's own work. Open it from the enclosing scope"
        )
    cm = ctx.launchable_graph_scope() if launchable else ctx.graph_scope()
    with cm as g:
        if symbol is not None:
            ctx.set_scope_symbol(str(symbol))  # names the scope in STF's DOT output
        # one push per logical data, with the union of the requested modes
        pushes = {}
        for deps, mode in ((reads, _AccessMode.READ), (writes, _AccessMode.RW)):
            for d in deps:
                if d is not None and not isinstance(d, _NullToken):
                    prev = pushes.get(id(d))
                    pushes[id(d)] = (d, int(mode) | (prev[1] if prev else 0))
        for ld, mode in pushes.values():
            ld.push(_AccessMode(mode))
        _scopes.append([ctx, symbol, None])
        poison = _PoisonStream.enter()
        _ScopeAllocator.install(ctx)
        try:
            yield g
        finally:
            _ScopeAllocator.uninstall()
            _PoisonStream.exit(poison)
            _scopes.pop()


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


class _Task:
    """One :func:`warp_task` block: an STF task with Warp's current stream on
    the task stream."""

    def __init__(self, ctx, deps, symbol, capture, dtypes):
        self.ctx = ctx
        self.deps = deps
        self.symbol = symbol
        self.stream = None
        self.arrays = ()
        self._capture = capture
        self._dtypes = dtypes
        self._task = None
        self._wp_scope = None
        self._captured = False
        self._poison = None

    def _start(self):
        wp = _import_warp()
        t = (
            self.ctx.task(*self.deps, symbol=self.symbol)
            if self.symbol is not None
            else self.ctx.task(*self.deps)
        )
        t.start()
        self._task = t
        try:
            self.stream = wrap_stream(t.stream_ptr())
            self._wp_scope = wp.ScopedStream(
                self.stream, sync_enter=False, sync_exit=False
            )
            self._wp_scope.__enter__()
            if self._capture == "always" or (
                self._capture == "auto" and self.stream.is_capturing
            ):
                wp.capture_begin(stream=self.stream, external=True)
                self._captured = True
            self.arrays = warp_arguments(t, self._dtypes)
        except BaseException as e:
            self._end(type(e), e, e.__traceback__)
            raise

    def _end(self, exc_type=None, exc_val=None, exc_tb=None):
        """End the task; returns the cleanup errors (not raised)."""
        wp = _import_warp()
        errors = []
        if self._captured:
            self._captured = False
            try:
                _graph_handles.append(wp.capture_end(stream=self.stream))
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
        if self._wp_scope is not None:
            sc, self._wp_scope = self._wp_scope, None
            try:
                sc.__exit__(exc_type, exc_val, exc_tb)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
        if self._task is not None:
            t, self._task = self._task, None
            try:
                t.end()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
        return errors

    def __enter__(self):
        self._poison = _PoisonStream.suspended()
        self._poison.__enter__()
        try:
            self._start()
        except BaseException:
            self._poison.__exit__(None, None, None)
            raise
        _open_tasks.append(self)
        return self.stream, self.arrays

    def __exit__(self, exc_type, exc_val, exc_tb):
        # same precedence as pytorch_task: body error, then cleanup errors
        errors = self._end(exc_type, exc_val, exc_tb)
        if _open_tasks and _open_tasks[-1] is self:
            _open_tasks.pop()
        self._poison.__exit__(None, None, None)
        if exc_type is not None:
            return False
        if errors:
            raise errors[0]
        return False


def warp_task(*args, symbol=None, capture="auto", dtypes=None):
    """Context manager: ``ctx.task(*deps)`` with Warp's current stream set to
    the task stream; yields ``(stream, arrays)``.

    ``warp_task(ctx, *deps)`` or ``warp_task(*deps)`` on the current context.
    Dependencies are the usual ones (``token.rw()``, ``data.read()``, ...);
    ``None`` deps (from an inert token) are dropped. A task that declares
    nothing is independent of every other task.

    Without ``ctx`` the body runs as is on Warp's current stream, with empty
    ``arrays``, when there is no current context and when a task is already
    open (the code runs as part of that task). With ``ctx``, a task opened
    inside another task raises.

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
    explicit = bool(args) and _is_context(args[0])
    if explicit:
        ctx, args = args[0], args[1:]
    else:
        ctx = current_context()
    deps = [a for a in args if a is not None]
    if ctx is None:
        return _no_task(wp)
    if _open_tasks and not explicit:
        return _no_task(wp)  # the code runs as part of the open task
    if _in_task(ctx):
        raise RuntimeError(
            f"task {symbol!r} opened inside task {_open_tasks[-1].symbol!r}: it would not "
            "be ordered against that task's own work. Open it from the enclosing scope"
        )
    return _Task(ctx, deps, symbol, capture, dtypes)


@contextlib.contextmanager
def _no_task(wp):
    yield wp.get_stream(), ()


def release_graph_handles():
    """Drop the Warp ``Graph`` handles kept alive by :func:`warp_task`. Call
    only after every STF context that captured through them is finalized."""
    _graph_handles.clear()


__all__ = [
    "current_context",
    "current_task_symbol",
    "get_arg_warp",
    "release_graph_handles",
    "scope",
    "scope_token",
    "token",
    "warp_arguments",
    "warp_task",
    "wrap_stream",
]
