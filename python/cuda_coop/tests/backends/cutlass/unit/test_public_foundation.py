# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import textwrap

from ..support import _subprocess


def test_cutlass_foundation_exports_current_public_surface():
    script = textwrap.dedent(
        """
        import sys

        import cuda.coop.cutlass as coop

        expected = [
            "Hierarchy",
            "TempStorage",
            "ThreadData",
            "ThreadDataLoadSource",
            "ThreadDataSource",
            "ThreadDataTensorMetadata",
            "ThreadGroup",
            "ThreadHierarchy",
            "adjacent_difference",
            "discontinuity",
            "exchange",
            "exclusive_scan",
            "exclusive_sum",
            "histogram",
            "inclusive_scan",
            "inclusive_sum",
            "load",
            "merge_sort_keys",
            "merge_sort_pairs",
            "radix_rank",
            "radix_sort_keys",
            "radix_sort_pairs",
            "reduce",
            "run_length_decode",
            "scan",
            "shuffle",
            "store",
            "sum",
            "this_block",
            "this_cluster",
            "this_grid",
            "this_thread",
            "this_warp",
            "topk_max_keys",
            "topk_max_pairs",
            "topk_min_keys",
            "topk_min_pairs",
        ]
        assert coop.__all__ == expected
        assert sorted(name for name in dir(coop) if not name.startswith("_")) == expected
        assert not hasattr(coop, "Payload")
        assert not hasattr(coop, "aot")
        assert not hasattr(coop, "_block")
        assert not hasattr(coop, "_warp")
        assert "numpy" not in sys.modules

        assert not any(
            name.startswith("cuda.coop.cutlass._lowering.")
            for name in sys.modules
        )
        assert "cuda.coop.cutlass._compiler._bundle" not in sys.modules
        assert "cuda.coop.cutlass._compiler._finalize" not in sys.modules
        dsl = _cuda_coop_test_dsl.CuTeDSL._get_dsl()
        assert len(dsl.trace_context_factories) == 1

        temp_storage_type = coop.TempStorage
        assert temp_storage_type.__module__ == "cuda.coop.cutlass._temp_storage"
        assert len(dsl.trace_context_factories) == 1
        """
    )

    result = _subprocess.run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_qualified_cutlass_import_reports_missing_runtime():
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"

        class BlockCutlassRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise ImportError("blocked CUTLASS runtime", name=fullname)
                return None

        sys.meta_path.insert(0, BlockCutlassRuntime())

        from cuda import coop

        assert coop.__name__ == "cuda.coop"
        assert not any(
            name == "cutlass" or name.startswith("cutlass.")
            for name in sys.modules
        )

        try:
            import cuda.coop.cutlass  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "cutlass"
            assert exc.reason_code == "backend-runtime-missing"
            assert exc.details["missing"] == "cutlass"
            assert isinstance(exc.__cause__, ImportError)
            message = str(exc)
            assert "nvidia-cutlass-dsl>=4.8" in message
            assert "cuda-coop[cutlass]" in message
        else:
            raise AssertionError("qualified CUTLASS import unexpectedly succeeded")
        """
    )

    result = _subprocess.run_python_with_source_and_site(script)

    assert result.returncode == 0, result.stderr


def test_qualified_cutlass_import_reports_missing_capabilities():
    script = textwrap.dedent(
        """
        import importlib.metadata

        importlib.metadata.version = lambda distribution: "99.7"
        _cuda_coop_test_compiler.GPUArch = None

        try:
            import cuda.coop.cutlass  # noqa: F401
        except ImportError as exc:
            assert exc.backend == "cutlass"
            assert exc.reason_code == "backend-runtime-incompatible"
            assert exc.details["missing_capabilities"] == (
                "cutlass.base_dsl.compiler.GPUArch",
            )
            message = str(exc)
            assert "cutlass.base_dsl.compiler.GPUArch" in message
            assert "detected nvidia-cutlass-dsl==99.7" in message
        else:
            raise AssertionError("incompatible CUTLASS import unexpectedly succeeded")
        """
    )

    result = _subprocess.run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_lowering_package_stays_lazy():
    script = textwrap.dedent(
        """
        import sys

        import cuda.coop.cutlass._lowering as lowering

        assert lowering.__all__ == ()
        assert not any(
            name.startswith("cuda.coop.cutlass._lowering.")
            for name in sys.modules
        )
        assert "cuda.coop.cutlass._compiler._bundle" not in sys.modules
        assert "cuda.coop.cutlass._compiler._finalize" not in sys.modules
        """
    )

    result = _subprocess.run_python_with_source(script)

    assert result.returncode == 0, result.stderr
