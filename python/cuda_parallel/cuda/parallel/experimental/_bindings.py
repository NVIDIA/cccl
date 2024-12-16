import os
import shutil
import importlib
import ctypes

from . import _cccl as cccl


def _get_cuda_path():
    cuda_path = os.environ.get("CUDA_PATH", "")
    if os.path.exists(cuda_path):
        return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    default_path = "/usr/local/cuda"
    if os.path.exists(default_path):
        return default_path

    return None


_bindings = None
_paths = None


def get_bindings():
    global _bindings
    if _bindings is None:
        include_path = importlib.resources.files("cuda.parallel.experimental").joinpath(
            "cccl"
        )
        cccl_c_path = os.path.join(include_path, "libcccl.c.parallel.so")
        _bindings = ctypes.CDLL(cccl_c_path)
        _bindings.cccl_device_reduce.restype = ctypes.c_int
        _bindings.cccl_device_reduce.argtypes = [
            cccl.DeviceReduceBuildResult,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulonglong),
            cccl.Iterator,
            cccl.Iterator,
            ctypes.c_ulonglong,
            cccl.Op,
            cccl.Value,
            ctypes.c_void_p,
        ]
        _bindings.cccl_device_reduce_cleanup.restype = ctypes.c_int
    return _bindings


def get_paths():
    global _paths
    if _paths is None:
        # Using `.parent` for compatibility with pip install --editable:
        include_path = importlib.resources.files("cuda.parallel").parent.joinpath(
            "_include"
        )
        include_path_str = str(include_path)
        include_option = "-I" + include_path_str
        cub_path = include_option.encode("utf-8")
        thrust_path = cub_path
        libcudacxx_path_str = str(os.path.join(include_path, "libcudacxx"))
        libcudacxx_option = "-I" + libcudacxx_path_str
        libcudacxx_path = libcudacxx_option.encode("utf-8")
        cuda_include_str = os.path.join(_get_cuda_path(), "include")
        cuda_include_option = "-I" + cuda_include_str
        cuda_include_path = cuda_include_option.encode("utf-8")
        _paths = cub_path, thrust_path, libcudacxx_path, cuda_include_path
    return _paths
