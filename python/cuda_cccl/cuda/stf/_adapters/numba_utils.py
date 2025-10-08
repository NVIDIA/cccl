"""
Utilities for NUMBA-based STF operations.
"""

from numba import cuda


def init_logical_data(ctx, ld, value, data_place=None, exec_place=None):
    """
    Initialize a logical data with a constant value using CuPy's optimized fill.

    Parameters
    ----------
    ctx : context
        STF context
    ld : logical_data
        Logical data to initialize
    value : scalar
        Value to fill the array with
    data_place : data_place, optional
        Data place for the initialization task
    exec_place : exec_place, optional
        Execution place for the fill operation
    """
    # Create write dependency with optional data place
    dep_arg = ld.write(data_place) if data_place else ld.write()

    # Create task arguments - include exec_place if provided
    task_args = []
    if exec_place is not None:
        task_args.append(exec_place)
    task_args.append(dep_arg)

    with ctx.task(*task_args) as t:
        # Get the array as a numba device array
        nb_stream = cuda.external_stream(t.stream_ptr())
        array = t.numba_arguments()

        try:
            # Use CuPy's optimized operations (much faster than custom kernels)
            import cupy as cp

            with cp.cuda.Stream(nb_stream):
                cp_view = cp.asarray(array)
                if value == 0 or value == 0.0:
                    # Use CuPy's potentially optimized zero operation
                    cp_view.fill(0)  # CuPy may have special optimizations for zero
                else:
                    # Use generic fill for non-zero values
                    cp_view.fill(value)
        except ImportError:
            # Fallback to simple kernel if CuPy not available
            _fill_with_simple_kernel(array, value, nb_stream)


@cuda.jit
def _fill_kernel_fallback(array, value):
    """Fallback 1D kernel when CuPy is not available."""
    idx = cuda.grid(1)
    if idx < array.size:
        array.flat[idx] = value


@cuda.jit
def _zero_kernel_fallback(array):
    """Optimized fallback kernel for zero-filling when CuPy is not available."""
    idx = cuda.grid(1)
    if idx < array.size:
        array.flat[idx] = 0


def _fill_with_simple_kernel(array, value, stream):
    """Fallback method using simple NUMBA kernel when CuPy unavailable."""
    total_size = array.size
    threads_per_block = 256
    blocks_per_grid = (total_size + threads_per_block - 1) // threads_per_block

    if value == 0 or value == 0.0:
        # Use the specialized zero kernel for potentially better performance
        _zero_kernel_fallback[blocks_per_grid, threads_per_block, stream](array)
    else:
        # Use generic fill kernel for non-zero values
        typed_value = array.dtype.type(value)
        _fill_kernel_fallback[blocks_per_grid, threads_per_block, stream](
            array, typed_value
        )
