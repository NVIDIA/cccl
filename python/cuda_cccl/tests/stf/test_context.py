# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cccl.experimental.stf import _stf_bindings_impl

def test_ctx():
    ctx = _stf_bindings_impl.Ctx()
    del ctx
