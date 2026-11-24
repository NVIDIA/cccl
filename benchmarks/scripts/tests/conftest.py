# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

# The benchmark scripts are not an installed package: `analyze.py`, `search.py`
# and friends import `cccl.bench` by virtue of living next to it. Make the same
# import work when the tests are collected from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
