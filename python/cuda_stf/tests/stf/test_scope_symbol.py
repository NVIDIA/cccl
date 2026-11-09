# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""stackable_context.set_level_symbol names the current level of a stackable
context for DOT output and diagnostics.

The in-process tests check that the call is accepted at every level. The DOT
check runs in a subprocess: the DOT writer is a process-wide singleton that
reads CUDASTF_DOT_FILE once and writes the file at exit.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402


def test_set_level_symbol_root_and_nested():
    ctx = stf.stackable_context()
    ctx.set_level_symbol("root")
    tok = ctx.token()
    with ctx.task(tok.write()):
        pass
    with ctx.graph_scope():
        ctx.set_level_symbol("outer")
        with ctx.task(tok.rw()):
            pass
        with ctx.graph_scope():
            ctx.set_level_symbol("inner")
            with ctx.task(tok.rw()):
                pass
        # back on the outer level
        ctx.set_level_symbol("outer-again")
    ctx.finalize()


_DOT_SCRIPT = textwrap.dedent(
    """
    import cuda.stf._experimental as stf
    ctx = stf.stackable_context()
    ctx.set_level_symbol("root-level")
    tok = ctx.token()
    with ctx.graph_scope():
        ctx.set_level_symbol("outer-level")
        with ctx.task(tok.write()):
            pass
        with ctx.graph_scope():
            ctx.set_level_symbol("inner-level")
            with ctx.task(tok.rw()):
                pass
    ctx.finalize()
    """
)


def test_set_level_symbol_labels_dot_clusters(tmp_path):
    dot_file = tmp_path / "levels.dot"
    env = dict(os.environ, CUDASTF_DOT_FILE=str(dot_file))
    subprocess.run(
        [sys.executable, "-c", _DOT_SCRIPT], env=env, check=True, timeout=300
    )
    text = dot_file.read_text()
    for name in ("root-level", "outer-level", "inner-level"):
        assert f'label="{name}"' in text, f"{name} missing from DOT output:\n{text}"
    # the inner level is nested in the outer one
    assert text.index('label="outer-level"') < text.index('label="inner-level"')
