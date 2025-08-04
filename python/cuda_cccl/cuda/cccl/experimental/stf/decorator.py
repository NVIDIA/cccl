from numba import cuda
from cuda.cccl.experimental.stf import context, dep, exec_place


class _CudaSTFKernel:
    def __init__(self, numba_kernel):
        self._nkern = numba_kernel
        self._launch_cfg = None  # (gridDim, blockDim, context, exec_place?)

    def __getitem__(self, cfg):
        if not (len(cfg) == 3 or len(cfg) == 4):
            raise TypeError("use kernel[gridDim, blockDim, ctx (, exec_place)]")

        gridDim, blockDim, ctx, *rest = cfg
        if not isinstance(ctx, context):
            raise TypeError("3rd item must be an STF context")

        exec_pl = rest[0] if rest else None
        if exec_pl and not isinstance(exec_pl, exec_place):
            raise TypeError("4th item must be an exec_place")

        self._launch_cfg = (int(gridDim), int(blockDim), ctx, exec_pl)
        return self

    def __call__(self, *args, **kwargs):
        if self._launch_cfg is None:
            raise RuntimeError("launch configuration missing – use kernel[grid, block, ctx](…)")

        gridDim, blockDim, ctx, exec_pl = self._launch_cfg

        dep_items = [(i, a) for i, a in enumerate(args) if isinstance(a, dep)]
        if not dep_items:
            raise TypeError("at least one argument must be an STF dep")

        task_args = [exec_pl] if exec_pl else []
        task_args.extend(a for _, a in dep_items)

        with ctx.task(*task_args) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            dev_args = list(args)
            for dep_index, (pos, _) in enumerate(dep_items):
                dev_args[pos] = t.get_arg_numba(dep_index)

            self._nkern[gridDim, blockDim, nb_stream](*dev_args, **kwargs)

        return None


def jit(*jit_args, **jit_kwargs):
    if jit_args and callable(jit_args[0]):
        pyfunc = jit_args[0]
        return _build_kernel(pyfunc, (), **jit_kwargs)

    def _decorator(fn):
        return _build_kernel(fn, jit_args, **jit_kwargs)

    return _decorator


def _build_kernel(pyfunc, jit_args, **jit_kwargs):
    numba_kernel = cuda.jit(*jit_args, **jit_kwargs)(pyfunc)
    return _CudaSTFKernel(numba_kernel)

