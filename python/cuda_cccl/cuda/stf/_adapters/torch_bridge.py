from __future__ import annotations


def cai_to_torch(cai: dict):
    """
    Convert a __cuda_array_interface__ dict to a torch.Tensor
    without making PyTorch a hard dependency of the core extension.

    Uses Numba (a required dependency) to create a DeviceNDArray,
    which torch.as_tensor can consume directly via __cuda_array_interface__.
    """
    import torch
    from numba import cuda as _cuda

    dev_array = _cuda.from_cuda_array_interface(cai, owner=None, sync=False)
    return torch.as_tensor(dev_array)
