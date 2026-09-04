# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect private Load/Store and Reduce provider factories.

Importing this package registers each factory and its scratch ABI. Group
planning selects a factory after validating a public call. Calling a factory
specializes device code but does not run the device operation.
"""

from ._load_store import load as load
from ._load_store import store as store

__all__: tuple[str, ...] = ()
