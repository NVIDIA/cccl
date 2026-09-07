# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reference helpers shared by CUTLASS runtime families."""

from __future__ import annotations

import torch


def gather_cpu_tensor(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Gather indices from a tensor after transferring it to the host."""

    values = tensor.tolist()
    return torch.tensor([values[index] for index in indices], dtype=tensor.dtype)


def assert_pairs_still_match_input(
    keys_in: torch.Tensor,
    values_in: torch.Tensor,
    keys_out: torch.Tensor,
    values_out: torch.Tensor,
) -> None:
    """Assert that every output value remains associated with its input key."""

    key_by_value = {
        value: key
        for key, value in zip(keys_in.tolist(), values_in.tolist(), strict=True)
    }
    for key, value in zip(
        keys_out.cpu().tolist(), values_out.cpu().tolist(), strict=True
    ):
        assert key == key_by_value[value]
