# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private CUTLASS DSL provider implementation.

Public cooperative APIs live in :mod:`cuda.coop.cutlass`. This package stays
non-eager beyond the provider modules selected by a traced operation. Default
root auto-registration may separately import the CUTLASS runtime to validate
and register it; the root opt-out retains the cold-import path.
"""

__all__: tuple[str, ...] = ()
