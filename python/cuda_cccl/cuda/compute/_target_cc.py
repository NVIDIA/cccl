# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target compute-capability context for device-code (LTO-IR) compilation.

Operators (numba) and iterators (NVRTC C++) are compiled to LTO-IR once and
linked into every per-arch build result of an algorithm. nvJitLink requires the final
target SM to be **at least as new** as every linked LTO/PTX input's arch. So for
a multi-arch build, this shared device code must be compiled for the **lowest**
target arch; otherwise linking a (say) sm_89 operator into an sm_80 cubin fails
with ``nvJitLink error``.

This module holds a context-local "target cc" that the leaf compilers
(``_jit`` for operators, ``_cpp_compile`` for iterators) consult. It is set
around a build by the caching decorator that wraps every ``make_<algo>``.
``None`` means "use the current device" — the default single-target behavior,
unchanged.
"""

from __future__ import annotations

import contextlib
import contextvars

# (major, minor) tuple, or None to mean "current device default".
_target_cc: contextvars.ContextVar = contextvars.ContextVar(
    "cccl_target_cc", default=None
)


def get_target_cc():
    """The current build's target cc as ``(major, minor)``, or ``None``.

    ``None`` means device code should target the current device (the default).
    """
    return _target_cc.get()


@contextlib.contextmanager
def target_cc(compute_capability):
    """Set the shared-device-code target cc for the duration of a build.

    ``compute_capability`` is the ``make_<algo>`` argument (``None`` / int /
    ``(major, minor)`` / list). For a multi-arch build the shared operator /
    iterator LTO-IR is compiled for the **lowest** requested arch so it links
    into every build result. ``None`` leaves the current-device default in place.
    """
    from ._cccl_interop import normalize_compute_capabilities

    ccs = normalize_compute_capabilities(compute_capability)
    # normalized list is sorted ascending, so ccs[0] is the minimum target.
    cc = ccs[0] if ccs else None
    token = _target_cc.set(cc)
    try:
        yield
    finally:
        _target_cc.reset(token)
