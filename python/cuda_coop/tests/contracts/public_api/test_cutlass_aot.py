# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import textwrap

import pytest

from ._imports import run_python_with_source


@pytest.mark.evidence_for("aot.pack", backend="cutlass", evidence="api")
def test_cutlass_aot_public_surface_is_dependency_light():
    script = textwrap.dedent(
        """
        import inspect
        import sys
        import types

        cutlass = types.ModuleType("cutlass")
        cutlass.__path__ = []
        cutlass_dsl = types.ModuleType("cutlass.cutlass_dsl")

        class CuTeDSL:
            _instance = None

            @classmethod
            def _get_dsl(cls):
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

            def register_trace_context_factory(self, factory):
                self.trace_context_factory = factory

            def register_trace_finalize_hook(self, hook):
                self.trace_finalize_hook = hook

        cutlass_dsl.CuTeDSL = CuTeDSL
        cute = types.ModuleType("cutlass.cute")
        cute._get_launch_facts = lambda: {}
        base_dsl = types.ModuleType("cutlass.base_dsl")
        base_dsl.__path__ = []
        compiler = types.ModuleType("cutlass.base_dsl.compiler")
        compiler.LinkLibraries = type(
            "LinkLibraries",
            (),
            {"_option_name": "link-libraries"},
        )
        compiler.GPUArch = type("GPUArch", (), {})
        cutlass.cutlass_dsl = cutlass_dsl
        cutlass.cute = cute
        cutlass.base_dsl = base_dsl
        base_dsl.compiler = compiler
        sys.modules["cutlass"] = cutlass
        sys.modules["cutlass.cutlass_dsl"] = cutlass_dsl
        sys.modules["cutlass.cute"] = cute
        sys.modules["cutlass.base_dsl"] = base_dsl
        sys.modules["cutlass.base_dsl.compiler"] = compiler

        from cuda.coop.cutlass import aot

        assert aot.__all__ == [
            "Capture",
            "CaptureError",
            "CaptureResult",
            "EntryInfo",
            "PackError",
            "PackInfo",
            "PackIntegrityError",
            "PackMissError",
            "capture",
            "inspect",
            "use",
        ]
        assert str(inspect.signature(aot.capture)) == (
            "(output: 'str | os.PathLike[str]', *, "
            "name: 'str | None' = None) -> 'Capture'"
        )
        assert str(inspect.signature(aot.use)) == (
            "(pack: 'str | os.PathLike[str]', *, "
            "mode: \\"Literal['auto', 'required', 'off']\\" = 'auto') "
            "-> 'Iterator[PackInfo | None]'"
        )
        assert str(inspect.signature(aot.inspect)) == (
            "(pack: 'str | os.PathLike[str]') -> 'PackInfo'"
        )
        assert "cuda.bindings.nvrtc" not in sys.modules
        assert "cuda.bindings.nvjitlink" not in sys.modules
        assert {
            name for name in sys.modules if name.startswith("cutlass")
        } == {
            "cutlass",
            "cutlass.base_dsl",
            "cutlass.base_dsl.compiler",
            "cutlass.cute",
            "cutlass.cutlass_dsl",
        }
        assert "torch" not in sys.modules
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr
