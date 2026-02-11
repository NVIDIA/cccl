import cupy as cp

import cuda.bench as bench


def as_cupy_stream(cs: bench.CudaStream) -> cp.cuda.Stream:
    """Convert nvbench CudaStream to CuPy Stream."""
    return cp.cuda.ExternalStream(cs.addressof())
