# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""First-launch metadata must reach the native launch configuration."""

from types import SimpleNamespace

import pytest

from ._support import requires_typed_pipeline


@requires_typed_pipeline
@pytest.mark.parametrize("configured", [0, 98304])
def test_first_launch_prepares_dynamic_bytes_before_launch(
    monkeypatch, configured
):
    from numba_cuda_mlir import descriptor

    from cuda.coop.numba_mlir import _pipeline

    _pipeline._activate()
    assert not descriptor._planner_registry.has_planners
    previous_tls = vars(descriptor._compile_arg_types).copy()
    native = object()
    geometry = {"grid": (1, 1, 1), "block": (32, 1, 1), "sharedmem": configured}
    launches, preparations = [], []
    ready = {}

    def prepare(args, values, argtypes, available, *_):
        assert descriptor._compile_arg_types.types == list(argtypes)
        assert descriptor._compile_arg_types.available_launch_config == geometry
        preparations.append(tuple(argtypes))
        required = 65536 if argtypes == ("int32",) else 81920
        result = (native, None, None, required)
        ready[tuple(argtypes)] = result
        return result

    dispatcher = SimpleNamespace(
        _literal_arg_positions=(),
        _requires_launch_config=False,
        _get_ready_launch=lambda argtypes, *_: ready.get(tuple(argtypes)),
        _prepare_for_launch=prepare,
    )

    def launcher_factory(
        native_dispatcher, grid, block, stream, sharedmem, cluster
    ):
        assert native_dispatcher is native
        return lambda *args: launches.append((sharedmem, args))

    monkeypatch.setattr(descriptor, "LaunchConfiguration", launcher_factory)
    marshaller = object.__new__(descriptor._ArgMarshaller)
    for name, value in {
        "_available_launch_config": geometry,
        "_has_extension_snapshot": False,
        "_dispatcher": dispatcher,
        "_launch_config": None,
        "_kernel_dispatcher": native,
        "_launch_config_generation": None,
        "_launch_stream": 0,
        "_adjusted_launcher_key": None,
        "_adjusted_launcher": None,
        "_launcher": lambda *args: launches.append((configured, args)),
    }.items():
        setattr(marshaller, name, value)

    for argtype in ("int32", "float64", "int32"):
        marshaller._launch([argtype], (7,))

    assert preparations == [("int32",), ("float64",)]
    assert launches == [
        (max(configured, required), (7,)) for required in (65536, 81920, 65536)
    ]
    assert vars(descriptor._compile_arg_types) == previous_tls
