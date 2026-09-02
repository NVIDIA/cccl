# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Thread-group descriptors for the qualified backend."""

from typing import Literal

from cuda.coop._core.api.thread_group import ThreadGroup as ThreadGroup

def this_block() -> ThreadGroup[Literal["block"]]: ...
def this_warp() -> ThreadGroup[Literal["warp"]]: ...

__all__ = ["ThreadGroup", "this_block", "this_warp"]
