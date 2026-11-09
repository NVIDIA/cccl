# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Every way the Python API names things ends up as a label in the DOT output.

Covered: logical data names (``name=`` and ``set_symbol``), task symbols
(``symbol=`` and ``set_symbol``) and host launches on a plain ``context``;
data, token, task and scope symbols (``set_scope_symbol``) on a
``stackable_context``.

The traced program runs in a subprocess: the DOT writer is a process-wide
singleton that reads CUDASTF_DOT_FILE once and writes the file at exit.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")

_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import cuda.stf._experimental as stf

    # plain context: data and task symbols, both spellings
    ctx = stf.context()
    a = ctx.logical_data(np.zeros(16, dtype=np.float32), name="plain-data-named")
    b = ctx.logical_data(np.zeros(16, dtype=np.float32))
    b.set_symbol("plain-data-set")
    with ctx.task(a.write(), b.write(), symbol="plain-task-kw"):
        pass
    t = ctx.task(a.rw(), b.read())
    t.set_symbol("plain-task-set")
    with t:
        pass
    ctx.host_launch(a.read(), fn=lambda *args: None, symbol="plain-host-launch")
    ctx.finalize()

    # stackable context: data, token, task and scope symbols
    sctx = stf.stackable_context()
    sctx.set_scope_symbol("root-scope")
    d = sctx.logical_data(np.zeros(16, dtype=np.float32), name="stack-data-named")
    tok = sctx.token()
    tok.set_symbol("stack-token-set")
    with sctx.graph_scope():
        sctx.set_scope_symbol("outer-scope")
        with sctx.task(d.rw(), tok.write(), symbol="stack-task-kw"):
            pass
        with sctx.graph_scope():
            sctx.set_scope_symbol("inner-scope")
            st = sctx.task(d.rw(), tok.rw())
            st.set_symbol("stack-task-set")
            with st:
                pass
    sctx.finalize()
    """
)

_EXPECTED = (
    "plain-data-named",
    "plain-data-set",
    "plain-task-kw",
    "plain-task-set",
    "plain-host-launch",
    "stack-data-named",
    "stack-token-set",
    "stack-task-kw",
    "stack-task-set",
    "root-scope",
    "outer-scope",
    "inner-scope",
)


def test_symbols_reach_dot_output(tmp_path):
    dot_file = tmp_path / "symbols.dot"
    env = dict(os.environ, CUDASTF_DOT_FILE=str(dot_file))
    subprocess.run([sys.executable, "-c", _SCRIPT], env=env, check=True, timeout=300)
    text = dot_file.read_text()
    missing = [s for s in _EXPECTED if s not in text]
    assert not missing, f"symbols missing from DOT output: {missing}\n{text}"
    # scope symbols label nested clusters, in nesting order
    assert text.index('label="outer-scope"') < text.index('label="inner-scope"')
