# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses

import numpy as np
from numba_cuda_mlir.extending import ArgumentHandler
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

import cuda.coop.numba_mlir as coop


@dataclasses.dataclass
class _ScaleBiasTraits:
    scale: np.int32
    bias: np.int32


def test_argument_handler_marshals_registered_scalar_fields():
    traits = coop.gpu_dataclass(
        _ScaleBiasTraits(np.int32(3), np.int32(7)),
        compute_temp_storage=False,
    )
    traits_type = typeof(traits)

    handler = coop.gpu_dataclass_argument_handler
    marshalled_type, marshalled_value = handler.prepare_args(traits_type, traits)

    assert isinstance(handler, ArgumentHandler)
    assert marshalled_type is traits_type
    assert marshalled_value == (np.int32(3), np.int32(7))
    assert typeof(marshalled_value) is traits_type
    assert not hasattr(traits, "prepare_args")
    assert not hasattr(traits, "pre_launch_callback")


def test_argument_handler_preserves_unregistered_values():
    value = object()
    value_type = object()

    marshalled_type, marshalled_value = (
        coop.gpu_dataclass_argument_handler.prepare_args(value_type, value)
    )

    assert marshalled_type is value_type
    assert marshalled_value is value


def test_reregistering_an_instance_preserves_signature_interning():
    first = coop.gpu_dataclass(
        _ScaleBiasTraits(np.int32(3), np.int32(7)),
        compute_temp_storage=False,
    )
    first_type = typeof(first)

    assert coop.gpu_dataclass(first, compute_temp_storage=False) is first

    second = coop.gpu_dataclass(
        _ScaleBiasTraits(np.int32(5), np.int32(11)),
        compute_temp_storage=False,
    )

    assert typeof(first) is first_type
    assert typeof(second) is first_type
