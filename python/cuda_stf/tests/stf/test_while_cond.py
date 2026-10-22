# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for compound while-loop conditions.

Covers the ``stf.cond`` leaf constructor, the comparison-operator sugar on
stackable logical data, the ``&`` / ``|`` / ``~`` combinators, and the
device-side evaluation of multi-term conditions through
``stf_stackable_while_cond_multi``.

Requires CUDA 12.4+ (conditional graph nodes).
"""

import numpy as np
import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.pytorch import pytorch_task  # noqa: E402

pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Expression-building tests (no while loop launched)
# ---------------------------------------------------------------------------


def _make_scalars(ctx, n=2):
    return [ctx.logical_data_empty((1,), np.float64, name=f"s{i}") for i in range(n)]


def test_cond_construction_and_operator_sugar():
    ctx = stf.stackable_context()
    la, lb = _make_scalars(ctx)

    # Canonical constructor and sugar produce equivalent leaves.
    leaf = stf.cond(la, ">", 0.5)
    sugar = la > 0.5
    for expr in (leaf, sugar):
        assert isinstance(expr, stf.cond)
        assert expr._op == ">"
        assert expr._threshold == 0.5
        assert expr._ld is la

    # Reflected comparison: scalar on the left flips the operator.
    reflected = 0.5 < la
    assert isinstance(reflected, stf.cond)
    assert reflected._op == ">"

    # Combinators build flat compounds; ~ negates.
    both = (la > 0.5) & (lb < 3.0)
    either = (la >= 1.0) | (lb <= 2.0)
    assert both._combiner == "all" and len(both._terms) == 2
    assert either._combiner == "any" and len(either._terms) == 2
    inverted = ~(la > 0.5)
    assert inverted._negate

    # De Morgan keeps compounds flat.
    neg = ~both
    assert neg._combiner == "any"
    assert all(t._negate for t in neg._terms)

    ctx.finalize()


def test_cond_validation_errors():
    ctx = stf.stackable_context()
    la, lb = _make_scalars(ctx)

    with pytest.raises(ValueError, match="comparison operator"):
        stf.cond(la, "!=", 0.5)
    with pytest.raises(TypeError, match="stackable logical_data"):
        stf.cond(1.0, ">", 0.5)
    with pytest.raises(TypeError, match="real scalar"):
        stf.cond(la, ">", np.zeros(4))
    with pytest.raises(TypeError, match="comparing two logical data"):
        la > lb
    with pytest.raises(TypeError, match="real scalar"):
        la > "0.5"

    # Conditions have no Python truth value: and/or/not must fail loudly.
    with pytest.raises(TypeError, match="truth value"):
        bool(la > 0.5)
    with pytest.raises(TypeError, match="truth value"):
        (la > 0.5) and (lb < 1.0)

    # Mixed &/| nesting has no flat representation.
    with pytest.raises(NotImplementedError, match="mixed"):
        ((la > 0.5) & (lb < 1.0)) | (la < 0.1)

    # Equality and hashing keep their identity semantics.
    assert la == la
    assert la != lb
    assert len({la, lb}) == 2

    ctx.finalize()


def test_cond_term_limit():
    ctx = stf.stackable_context()
    scalars = _make_scalars(ctx, 9)
    expr = scalars[0] > 0.0
    for ld in scalars[1:8]:
        expr = expr & (ld > 0.0)
    with pytest.raises(ValueError, match="at most 8"):
        expr & (scalars[8] > 0.0)
    ctx.finalize()


# ---------------------------------------------------------------------------
# Device-side evaluation tests
# ---------------------------------------------------------------------------


def _run_capped_loop(make_condition, flag_value=1.0):
    """Run a while loop whose body increments a counter once per iteration.

    ``make_condition(lcounter, lflag)`` returns the condition expression; the
    flag logical data stays at ``flag_value``. Returns the final counter.
    """
    counter_host = np.zeros(1, dtype=np.float64)
    flag_host = np.full(1, flag_value, dtype=np.float64)

    ctx = stf.stackable_context()
    lcounter = ctx.logical_data(counter_host, name="counter")
    lflag = ctx.logical_data(flag_host, name="flag")

    with ctx.while_loop() as loop:
        with pytorch_task(ctx, lcounter.rw()) as (tCounter,):
            tCounter += 1.0
        loop.continue_while(make_condition(lcounter, lflag))

    ctx.finalize()
    return counter_host[0]


def test_while_all_combiner_stops_at_cap():
    # flag > 0.5 always holds; counter < 5 caps the loop at 5 iterations.
    iters = _run_capped_loop(lambda lc, lf: (lf > 0.5) & (lc < 5.0))
    assert iters == 5.0


def test_while_any_combiner_with_negated_term():
    # ~(flag > 0.5) is always false, so only counter < 3 keeps the loop going.
    iters = _run_capped_loop(lambda lc, lf: (lc < 3.0) | ~(lf > 0.5))
    assert iters == 3.0


def test_while_duplicate_ld_terms_share_dependency():
    iters = _run_capped_loop(lambda lc, lf: (lc < 4.0) & (lc > -1.0))
    assert iters == 4.0


def test_while_single_expression_and_legacy_form():
    # Single-leaf expression form.
    iters = _run_capped_loop(lambda lc, lf: lc < 2.0)
    assert iters == 2.0

    # Legacy (ld, op, threshold) form is unchanged.
    counter_host = np.zeros(1, dtype=np.float64)
    ctx = stf.stackable_context()
    lcounter = ctx.logical_data(counter_host, name="counter")
    with ctx.while_loop() as loop:
        with pytorch_task(ctx, lcounter.rw()) as (tCounter,):
            tCounter += 1.0
        loop.continue_while(lcounter, "<", 2.0)
    ctx.finalize()
    assert counter_host[0] == 2.0
