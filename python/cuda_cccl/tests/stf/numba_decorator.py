# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba-based @jit decorator for STF. Not shipped in the wheel.
Requires numba-cuda. Import from tests.stf when running from source.
"""

from numba import cuda

from cuda.stf import context, dep, exec_place


class stf_kernel_decorator:
    def __init__(self, pyfunc, jit_args, jit_kwargs):
        self._pyfunc = pyfunc
        self._jit_args = jit_args
        self._jit_kwargs = jit_kwargs
        self._compiled_kernel = None
        self._launch_cfg = None

    def __getitem__(self, cfg):
        if not (isinstance(cfg, tuple) or isinstance(cfg, list)):
            raise TypeError("use kernel[grid, block ([, exec_place, ctx])]")
        n = len(cfg)
        if n not in (2, 3, 4):
            raise TypeError(
                "use kernel[grid, block], kernel[grid, block, exec_place], or kernel[grid, block, exec_place, ctx]"
            )

        grid_dim = cfg[0]
        block_dim = cfg[1]
        ctx = None
        exec_pl = None

        if n >= 3:
            exec_pl = cfg[2]

        if n == 4:
            ctx = cfg[3]

        if exec_pl is not None and not isinstance(exec_pl, exec_place):
            raise TypeError("3rd item must be an exec_place")

        if ctx is not None and not isinstance(ctx, context):
            raise TypeError("4th item must be an STF context (or None to infer)")

        self._launch_cfg = (grid_dim, block_dim, ctx, exec_pl)
        return self

    def __call__(self, *args, **kwargs):
        if self._launch_cfg is None:
            raise RuntimeError(
                "launch configuration missing – use kernel[grid, block, ctx](...)"
            )

        gridDim, blockDim, ctx, exec_pl = self._launch_cfg

        dep_items = []
        for i, a in enumerate(args):
            if isinstance(a, dep):
                if ctx is None:
                    ld = a.get_ld()
                    ctx = ld.borrow_ctx_handle()
                dep_items.append((i, a))

        task_args = [exec_pl] if exec_pl else []
        task_args.extend(a for _, a in dep_items)

        with ctx.task(*task_args) as t:
            dev_args = list(args)
            for dep_index, (pos, _) in enumerate(dep_items):
                cai = t.get_arg_cai(dep_index)
                dev_args[pos] = cuda.from_cuda_array_interface(
                    cai.__cuda_array_interface__, owner=None, sync=False
                )

            if self._compiled_kernel is None:
                self._compiled_kernel = cuda.jit(*self._jit_args, **self._jit_kwargs)(
                    self._pyfunc
                )

            nb_stream = cuda.external_stream(t.stream_ptr())
            self._compiled_kernel[gridDim, blockDim, nb_stream](*dev_args, **kwargs)

        return None


def jit(*jit_args, **jit_kwargs):
    if jit_args and callable(jit_args[0]):
        pyfunc = jit_args[0]
        return _build_kernel(pyfunc, (), jit_kwargs)

    def _decorator(fn):
        return _build_kernel(fn, jit_args, jit_kwargs)

    return _decorator


def _build_kernel(pyfunc, jit_args, jit_kwargs=None):
    if jit_kwargs is None:
        jit_kwargs = {}
    return stf_kernel_decorator(pyfunc, jit_args, jit_kwargs)


__all__ = ["jit", "stf_kernel_decorator"]
