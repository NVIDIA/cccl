def cai_to_numba(cai: dict):
    from numba import cuda
    return cuda.from_cuda_array_interface(cai, owner=None, sync=False)
