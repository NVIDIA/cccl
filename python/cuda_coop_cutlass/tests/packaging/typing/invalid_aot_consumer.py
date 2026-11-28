# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the accepted pack modes and capture metadata at call sites."""

from cuda.coop.cutlass import aot

aot.use("pack", mode="strict")  # expected-error: [arg-type]
aot.capture("pack", name=12)  # expected-error: [arg-type]
