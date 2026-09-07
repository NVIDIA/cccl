# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thread-block descriptor for the qualified backend."""

from cuda.coop import ThreadGroup as ThreadGroup

def this_block() -> ThreadGroup: ...

__all__ = ["ThreadGroup", "this_block"]
