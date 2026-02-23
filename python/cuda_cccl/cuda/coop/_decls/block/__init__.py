# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module

_SIDE_EFFECT_MODULES: tuple[str, ...] = (
    "_block_load_store",
    "_block_exchange",
    "_block_scan",
    "_block_reduce",
    "_block_histogram",
)


def import_side_effect_modules() -> None:
    for module_name in _SIDE_EFFECT_MODULES:
        import_module(f"{__name__}.{module_name}")
