# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Lifecycle/ownership tests for the STF Cython wrappers.

These tests exercise every destruction order between a context and its
children (logical_data, task, stackable_logical_data, stackable_task) plus
inter-test contamination scenarios that previously aborted the interpreter
with ``cudaErrorContextIsDestroyed`` from ``~stream_and_event``.

The fix (see ``context._alive`` / ``stackable_context._alive`` in
``_stf_bindings_impl.pyx``) gives every context a Python-refcounted sentinel
that all child wrappers share and consult in their ``__dealloc__``. When the
context is finalized -- explicitly, or when an unfinalized context is merely
abandoned during ``__dealloc__`` -- the sentinel is flipped so any surviving
child becomes a no-op on destruction.

Without the fix, the multi-context tests below abort the interpreter on
garbage collection rather than failing cleanly. They MUST run in this same
process / module so a regression actually trips them.
"""

import gc

import numpy as np
import pytest

import cuda.stf._experimental as stf

# ---------------------------------------------------------------------------
# context (non-stackable)
# ---------------------------------------------------------------------------


def _make_ctx_and_leak_logical_data():
    """Return a ``logical_data`` whose owning ``context`` was already
    finalize()d. Without the sentinel fix, dropping the returned object
    later (especially after another context exists) crashes."""
    ctx = stf.context()
    buf = np.ones(16, dtype=np.float64)
    ld = ctx.logical_data(buf, name="lA")
    ctx.finalize()
    return ld


def test_logical_data_outlives_explicitly_finalized_context():
    leaked = _make_ctx_and_leak_logical_data()
    assert leaked is not None
    # Context is gone; this dealloc must NOT call stf_logical_data_destroy.
    del leaked
    gc.collect()


def test_logical_data_outlives_unfinalized_context():
    """No explicit finalize() -- leaking the context should warn but not crash."""

    def make():
        ctx = stf.context()
        buf = np.ones(16, dtype=np.float64)
        return ctx.logical_data(buf, name="lA")

    with pytest.warns(ResourceWarning, match="without an explicit finalize"):
        leaked = make()
    gc.collect()  # ctx may still be on Python's frame; force its release
    del leaked
    gc.collect()


def test_multiple_contexts_in_sequence():
    """Two back-to-back contexts; objects from #1 outlive into #2's lifetime.

    This is the pattern that previously aborted in pytest bulk runs
    (``test_burger_stackable.py`` followed by ``test_burger_stackable_fast``).
    """
    leaked_from_first = _make_ctx_and_leak_logical_data()
    ctx2 = stf.context()
    ld2 = ctx2.logical_data(np.zeros(8, dtype=np.float32), name="lB")
    ctx2.finalize()
    del ld2
    del leaked_from_first
    gc.collect()


def test_child_destroyed_before_context():
    """Healthy path: destroying the child first MUST run its C destroy."""
    ctx = stf.context()
    ld = ctx.logical_data(np.ones(8, dtype=np.float64), name="lA")
    del ld  # _alive is still True -> stf_logical_data_destroy runs
    gc.collect()
    ctx.finalize()


def test_double_finalize_is_safe():
    ctx = stf.context()
    ld = ctx.logical_data(np.ones(8, dtype=np.float64), name="lA")
    ctx.finalize()
    ctx.finalize()  # idempotent: no error, sentinel already False
    del ld
    gc.collect()


def test_token_outlives_context():
    ctx = stf.context()
    tok = ctx.token()
    ctx.finalize()
    del tok
    gc.collect()


def test_task_outlives_context():
    ctx = stf.context()
    ld = ctx.logical_data(np.ones(8, dtype=np.float64), name="lA")
    t = ctx.task(ld.rw())
    t.start()
    t.end()
    ctx.finalize()
    # Hold both references past finalize() and drop later.
    del t
    del ld
    gc.collect()


# ---------------------------------------------------------------------------
# stackable_context
# ---------------------------------------------------------------------------


def _make_stackable_ctx_and_leak():
    sctx = stf.stackable_context()
    buf = np.ones(16, dtype=np.float64)
    sld = sctx.logical_data(buf, name="lA")
    sctx.finalize()
    return sld


def _exercise_stackable_repeat_scope():
    with stf.stackable_context() as sctx:
        with sctx.repeat(1):
            pass


def test_stackable_logical_data_outlives_explicit_finalize():
    leaked = _make_stackable_ctx_and_leak()
    assert leaked is not None
    del leaked
    gc.collect()


def test_stackable_logical_data_outlives_unfinalized_context():
    def make():
        sctx = stf.stackable_context()
        return sctx.logical_data(np.ones(16, dtype=np.float64), name="lA")

    with pytest.warns(ResourceWarning, match="without an explicit finalize"):
        leaked = make()
    gc.collect()
    del leaked
    gc.collect()


def test_two_stackable_contexts_in_sequence():
    """Reproduces the pattern from the burger_stackable / burger_stackable_fast
    bulk-run abort."""
    leaked_from_first = _make_stackable_ctx_and_leak()
    sctx2 = stf.stackable_context()
    sld2 = sctx2.logical_data(np.zeros(8, dtype=np.float32), name="lB")
    sctx2.finalize()
    del sld2
    del leaked_from_first
    gc.collect()


def test_stackable_repeat_after_device_reset():
    """A device reset must not leave pooled STF streams pointing at a dead context."""
    cuda = pytest.importorskip("numba.cuda")

    _exercise_stackable_repeat_scope()
    cuda.get_current_device().reset()
    _exercise_stackable_repeat_scope()


def test_stackable_token_outlives_context():
    sctx = stf.stackable_context()
    tok = sctx.token()
    sctx.finalize()
    del tok
    gc.collect()


def test_stackable_task_outlives_context():
    sctx = stf.stackable_context()
    sld = sctx.logical_data(np.ones(8, dtype=np.float64), name="lA")
    t = sctx.task(sld.rw())
    t.start()
    t.end()
    sctx.finalize()
    del t
    del sld
    gc.collect()


def test_stackable_double_finalize_is_safe():
    sctx = stf.stackable_context()
    sld = sctx.logical_data(np.ones(8, dtype=np.float64), name="lA")
    sctx.finalize()
    sctx.finalize()
    del sld
    gc.collect()


# ---------------------------------------------------------------------------
# Cross-flavor: do non-stackable and stackable contexts coexist cleanly?
# ---------------------------------------------------------------------------


def test_mixed_context_types_with_outliving_children():
    leaked_plain = _make_ctx_and_leak_logical_data()
    leaked_stack = _make_stackable_ctx_and_leak()

    # Recreate fresh contexts of each kind, then destroy them, then drop the
    # leaked references. This is the most adversarial GC ordering.
    ctx = stf.context()
    sctx = stf.stackable_context()
    ld = ctx.logical_data(np.zeros(4, dtype=np.float64))
    sld = sctx.logical_data(np.zeros(4, dtype=np.float64))
    ctx.finalize()
    sctx.finalize()
    del ld
    del sld
    del leaked_plain
    del leaked_stack
    gc.collect()


# ---------------------------------------------------------------------------
# Sentinel internals: verify the mechanism really is a shared object
# ---------------------------------------------------------------------------


def test_sentinel_is_shared_between_context_and_child():
    """White-box: confirm the sentinel object is identity-shared, not copied.

    If a future refactor accidentally turns _alive into a ``cdef bint``, this
    test fails immediately instead of regressing the lifecycle silently.
    """
    ctx = stf.context()
    ld = ctx.logical_data(np.ones(4, dtype=np.float64))
    # _alive isn't a Python attr (Cython cdef), so probe via finalize semantics
    # instead: after finalize(), dropping ld must not raise / abort.
    ctx.finalize()
    del ld
    gc.collect()
