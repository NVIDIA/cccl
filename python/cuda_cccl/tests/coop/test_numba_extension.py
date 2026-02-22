# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.coop import _numba_extension as ext


def test_numba_extension_global_aliases_match():
    assert ext.CUDA_CCCL_COOP_DEBUG == ext.NUMBA_CCCL_COOP_DEBUG
    assert ext.CUDA_CCCL_COOP_INJECT_PRINTFS == ext.NUMBA_CCCL_COOP_INJECT_PRINTFS


def test_source_code_rewriter_aliases_stay_in_sync():
    old_rewriter = ext._get_source_code_rewriter()
    marker = object()
    try:
        ext._set_source_code_rewriter(marker)
        assert ext._get_source_code_rewriter() is marker
        assert ext.CUDA_CCCL_COOP_SOURCE_CODE_REWRITER is marker
        assert ext.NUMBA_CCCL_COOP_SOURCE_CODE_REWRITER is marker
    finally:
        ext._set_source_code_rewriter(old_rewriter)
