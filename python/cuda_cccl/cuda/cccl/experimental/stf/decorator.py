from numba import cuda
import numba
from cuda.cccl.experimental.stf import context, dep, exec_place
numba.config.CUDA_ENABLE_PYNVJITLINK = 1

class stf_kernel_decorator:
    def __init__(self, pyfunc, jit_args, jit_kwargs):
        self._pyfunc = pyfunc
        self._jit_args = jit_args
        self._jit_kwargs = jit_kwargs
        self._compiled_kernel = None
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
            raise RuntimeError("launch configuration missing – use kernel[grid, block, ctx](...)")

        gridDim, blockDim, ctx, exec_pl = self._launch_cfg

        dep_items = []
        for i, a in enumerate(args):
            print(f'got one arg {a} is dep ? {isinstance(a, dep)}')
            if isinstance(a, dep):
                dep_items.append((i, a))

        task_args = [exec_pl] if exec_pl else []
        task_args.extend(a for _, a in dep_items)

        with ctx.task(*task_args) as t:
            dev_args = list(args)
            print(dev_args)
            for dep_index, (pos, _) in enumerate(dep_items):
                print(f'set arg {dep_index} at position {pos}')
                dev_args[pos] = t.get_arg_numba(dep_index)

            if self._compiled_kernel is None:
                print("compile kernel")
                self._compiled_kernel = cuda.jit(*self._jit_args, **self._jit_kwargs)(self._pyfunc)

            nb_stream = cuda.external_stream(t.stream_ptr())
            self._compiled_kernel[gridDim, blockDim, nb_stream](*dev_args, **kwargs)

        return None


def jit(*jit_args, **jit_kwargs):
    if jit_args and callable(jit_args[0]):
        pyfunc = jit_args[0]
        return _build_kernel(pyfunc, (), **jit_kwargs)

    def _decorator(fn):
        return _build_kernel(fn, jit_args, **jit_kwargs)

    return _decorator


def _build_kernel(pyfunc, jit_args, **jit_kwargs):
    return stf_kernel_decorator(pyfunc, jit_args, jit_kwargs)
