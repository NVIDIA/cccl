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

* **A scope opened inside a block continues the block.** A library that is
  called from someone's block marks the parts of its work that may overlap
  with ``scope`` + ``warp_task``; the code before and after stays as written
  and runs in the caller's block, which is cut around the scope and resumed
  after it (see :func:`scope`). The caller does not need to know, and tasks
  that declare nothing are joined by the code that follows them.
  :func:`current_context` gives such a library the context being recorded.

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

#: How a scope opened inside a block is recorded (see :func:`scope`): in the
#: block's own graph (``True``), or as a nested STF scope, which is a CUDA child
#: graph (``False``). Same ordering either way.
fork_in_place = __import__("os").environ.get("CUDA_STF_WARP_FORK_IN_PLACE", "1") == "1"

_graph_handles = []  # Graph objects from wp.capture_end inside tasks (see module docstring)
_scopes = {}  # id(ctx) -> stack of scope tokens
_ctx_stack = []  # contexts of the open scopes, innermost last
_open_tasks = []  # the warp_task / warp_native blocks currently open (_Block), innermost last


def current_context():
    """The context of the innermost open :func:`scope`, or ``None`` when no
    scope is open. A library reads it at its entry instead of being handed a
    context: outside a recording it gets ``None`` and every construct of this
    module is a no-op.

    >>> ctx = current_context()
    >>> with scope(ctx):
    ...     with warp_task(ctx, symbol="mass"): ...
    ...     with warp_task(ctx, symbol="rows"): ...
    """
    return _ctx_stack[-1] if _ctx_stack else None


def current_task_symbol():
    """Symbol of the innermost open :func:`warp_task` / :func:`warp_native`
    block, or ``None``. For tools that attribute Warp events (a conditional
    capture, an allocation) to the task that recorded them."""
    return _open_tasks[-1].symbol if _open_tasks else None


def _open_block(ctx):
    """The innermost block of ``ctx`` that is open and not suspended by a scope."""
    if _open_tasks and _open_tasks[-1].ctx is ctx and _open_tasks[-1]._task is not None:
        return _open_tasks[-1]
    return None


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


def _scope_token(ctx):
    st = _scopes.get(id(ctx))
    return st[-1] if st else None


@contextlib.contextmanager
def scope(ctx, launchable=False, reads=(), writes=(), symbol=None):
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
    scope at entry with that access mode, so the scope's node is ordered like a
    task declaring them (a scope with dependencies is a task whose body is a
    graph). ``symbol`` names the scope (STF DOT output, diagnostics) and its
    token.

    **A scope opened inside a block** (:func:`warp_task` / :func:`warp_native`)
    takes over that block's dependencies, and the block continues after it:
    the block is ended where the scope opens, the scope is entered with the
    block's dependencies and access modes, and a new task with the same
    dependencies and symbol is started when the scope closes. The caller wrote
    one block; the graph holds first part -> scope -> second part, ordered
    among themselves and, as a whole, like the block. Code that records tasks
    can therefore be called from any block, by a caller that does not know:

    >>> with warp_native(ctx, "step"):           # caller
    ...     prepare()                            #   plain: first part
    ...     with scope(ctx):                     #   callee: these may overlap
    ...         with warp_task(ctx, symbol="mass"): ...
    ...         with warp_task(ctx, symbol="rows"): ...
    ...     finish()                             #   plain: second part

    The block's stream changes when it resumes (``wp.get_stream()`` is always
    the right one). Refused: a block that yielded arrays of logical data, and
    a scope opened while Warp's current stream is not the block's own.

    >>> with scope(ctx, launchable=True) as g:
    ...     with warp_native(ctx, "collide"): example.collide()
    ...     with warp_native(ctx, "step"): solver.step(...)   # may open scopes inside
    ...     for _ in range(frames): g.launch()
    """
    if ctx is None:
        yield None
        return
    blk = _open_block(ctx)
    if (
        blk is not None
        and fork_in_place
        and not launchable
        and not reads
        and not writes
    ):
        # Recorded in the block's own graph: no STF scope is pushed. The block's
        # support token plays the scope token (tasks read it, native blocks and
        # the rest of the block hold it read-write), so the tasks are ordered
        # after the block's first part and joined by its second part.
        blk._suspend()
        resumed = False
        try:
            _scopes.setdefault(id(ctx), []).append(blk._support)
            _ctx_stack.append(ctx)
            poison = _PoisonStream.enter()
            try:
                yield None
            finally:
                _PoisonStream.exit(poison)
                _ctx_stack.pop()
                _scopes[id(ctx)].pop()
            blk._start_piece()
            resumed = True
        finally:
            if not resumed:
                blk._abandon()
        return
    if blk is not None:
        blk._suspend()
    resumed = False
    try:
        outer = _scope_token(ctx)
        cm = ctx.launchable_graph_scope() if launchable else ctx.graph_scope()
        with cm as g:
            if symbol is not None:
                ctx.set_scope_symbol(str(symbol))  # names the scope in STF's DOT output
            tok = ctx.token()
            tok.set_symbol(f"scope:{symbol}" if symbol is not None else "scope")
            # one push per logical data, with the union of the requested modes
            pushes = {}

            def want(ld, mode):
                mode = int(mode)
                if mode & int(_AccessMode.WRITE):
                    mode = int(_AccessMode.RW)
                prev = pushes.get(id(ld))
                pushes[id(ld)] = (ld, mode | (prev[1] if prev else 0))

            if outer is not None:
                want(outer, _AccessMode.READ)
            for d in reads:
                if d is not None:
                    want(d, _AccessMode.READ)
            for d in writes:
                if d is not None:
                    want(d, _AccessMode.RW)
            if blk is not None:
                # inherit the enclosing block's dependencies and its support token
                for d in blk.deps:
                    want(d.ld, d.mode)
                if blk._support is not None:
                    want(blk._support, _AccessMode.RW)
            for ld, mode in pushes.values():
                ld.push(_AccessMode(mode))
            _scopes.setdefault(id(ctx), []).append(tok)
            _ctx_stack.append(ctx)
            poison = _PoisonStream.enter()
            _ScopeAllocator.install(ctx)
            try:
                yield g
            finally:
                _ScopeAllocator.uninstall()
                _PoisonStream.exit(poison)
                _ctx_stack.pop()
                st = _scopes[id(ctx)]
                st.pop()
                if not st:
                    del _scopes[id(ctx)]
        if blk is not None:
            blk._start_piece()  # the block continues after the scope
            resumed = True
    finally:
        if blk is not None and not resumed:
            blk._abandon()


# ---------------------------------------------------------------------------
# tasks and native blocks
# ---------------------------------------------------------------------------


class _Block:
    """One :func:`warp_task` / :func:`warp_native` block. Usually one STF task;
    a scope opened inside it cuts it into pieces (see :func:`scope`), all with
    the block's dependencies and chained by the block's *support token*: a
    token nothing else knows, held read-write by every piece and by the scope
    between them, so the pieces are ordered among themselves even when the
    block declares no dependency it writes."""

    def __init__(self, ctx, deps, symbol, capture, dtypes):
        self.ctx = ctx
        self.deps = deps  # the declared dependencies (dep objects)
        self.symbol = symbol
        self.stream = None
        self.arrays = ()
        self._capture = capture
        self._dtypes = dtypes
        self._support = None
        if hasattr(
            ctx, "graph_scope"
        ):  # stackable context: scopes can be opened inside the block
            self._support = ctx.token()
            self._support.set_symbol(
                f"block:{symbol}" if symbol is not None else "block"
            )
        self._task = (
            None  # the STF task of the current piece; None while suspended by a scope
        )
        self._wp_scope = None
        self._captured = False
        self._poison = None

    def _start_piece(self):
        wp = _import_warp()
        deps = list(self.deps)
        tok = _scope_token(self.ctx)
        if tok is not None:
            deps.append(tok.read())
        if self._support is not None:
            deps.append(self._support.rw())
        t = (
            self.ctx.task(*deps, symbol=self.symbol)
            if self.symbol is not None
            else self.ctx.task(*deps)
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
            self._end_piece(type(e), e, e.__traceback__)
            raise

    def _end_piece(self, exc_type=None, exc_val=None, exc_tb=None):
        """End the current piece; returns the cleanup errors (not raised)."""
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

    def _suspend(self):
        """A scope opens inside the block: end the current piece."""
        wp = _import_warp()
        if self.arrays:
            raise RuntimeError(
                "a scope cannot be opened inside a block that holds arrays of logical data "
                f"({self.symbol!r}): the views would not be valid after the scope"
            )
        if wp.get_stream() is not self.stream:
            raise RuntimeError(
                f"a scope cannot be opened inside block {self.symbol!r} while Warp's current "
                "stream is not the block's stream"
            )
        errors = self._end_piece()
        if errors:
            raise errors[0]

    def _abandon(self):
        """The scope opened inside the block failed: the block has no piece to end."""
        self._task = None

    def __enter__(self):
        self._poison = _PoisonStream.suspended()
        self._poison.__enter__()
        try:
            self._start_piece()
        except BaseException:
            self._poison.__exit__(None, None, None)
            raise
        _open_tasks.append(self)
        return self.stream, self.arrays

    def __exit__(self, exc_type, exc_val, exc_tb):
        # same precedence as pytorch_task: body error, then cleanup errors
        errors = self._end_piece(exc_type, exc_val, exc_tb)
        if _open_tasks and _open_tasks[-1] is self:
            _open_tasks.pop()
        self._poison.__exit__(None, None, None)
        if exc_type is not None:
            return False
        if errors:
            raise errors[0]
        return False


def warp_task(ctx, *args, symbol=None, capture="auto", dtypes=None):
    """Context manager: ``ctx.task(*args)`` with Warp's current stream set to
    the task stream; yields ``(stream, arrays)``. Inside a scope the task also
    reads the scope token. ``None`` deps (from an inert :func:`token`) are
    dropped; with ``ctx=None`` the body runs as is on Warp's current stream and
    ``arrays`` is empty.

    Tasks of a scope that declare nothing overlap; plain code after them (a
    native block, or the rest of the block the scope was opened in) joins
    them. Dependencies are only needed for an order between sibling tasks.
    A task cannot be opened directly inside another block: open a
    :func:`scope` around the tasks that may overlap.

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
    outer = _open_block(ctx)
    if outer is not None:
        raise RuntimeError(
            f"block {symbol!r} opened inside block {outer.symbol!r}: it would not be ordered "
            "against that block's own work. Open a scope() around the tasks that may overlap"
        )
    return _Block(ctx, deps, symbol, capture, dtypes)


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
    """Drop the Warp ``Graph`` handles kept alive by :func:`warp_task`. Call
    only after every STF context that captured through them is finalized."""
    _graph_handles.clear()


__all__ = [
    "array_arg",
    "current_context",
    "current_task_symbol",
    "release_graph_handles",
    "scope",
    "token",
    "warp_arguments",
    "warp_native",
    "warp_task",
    "wrap_stream",
]
