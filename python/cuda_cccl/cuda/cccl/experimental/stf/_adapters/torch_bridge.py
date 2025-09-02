from __future__ import annotations


def cai_to_torch(cai: dict):
    """
    Convert a __cuda_array_interface__ dict to a torch.Tensor
    without making PyTorch a hard dependency of the core extension.

    Strategy (in order):
      1) Try Numba -> DLPack -> torch (fast & common).
      2) Try CuPy  -> DLPack -> torch (common on CUDA setups).
      3) Otherwise, error with a clear message.
    """
    import torch

    # 1) Numba bridge
    try:
        from numba import cuda as _cuda

        dev_array = _cuda.from_cuda_array_interface(cai, owner=None, sync=False)
        return torch.utils.dlpack.from_dlpack(dev_array.to_dlpack())
    except Exception:
        pass

    # 2) CuPy bridge
    try:
        import cupy as cp

        class _cai_wrapper:
            def __init__(self, d):
                self.__cuda_array_interface__ = d

        cp_arr = cp.asarray(_cai_wrapper(cai))
        return torch.utils.dlpack.from_dlpack(cp_arr.toDlpack())
    except Exception as e:
        raise RuntimeError(
            "Could not convert __cuda_array_interface__ to torch.Tensor. "
            "Install numba or cupy (or expose a DLPack capsule natively)."
        ) from e
