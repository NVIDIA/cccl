def cai_to_numba(cai: dict):
    from numba import cuda as _cuda
    return _cuda.from_cuda_array_interface(cai)
