from dataclasses import dataclass
from functools import lru_cache
import os
import shutil
from typing import Optional


def _get_cuda_path() -> Optional[str]:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))

    default_path = "/usr/local/cuda"
    if os.path.exists(default_path):
        return default_path

    return None


@dataclass
class IncludePaths:
    cuda: Optional[str]
    libcudacxx: Optional[str]
    cub: Optional[str]
    thrust: Optional[str]

    def as_tuple(self):
        # Note: higher-level ... lower-level order:
        return (self.thrust, self.cub, self.libcudacxx, self.cuda)


@lru_cache()
def get_include_paths() -> IncludePaths:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    cuda_incl = None
    cuda_path = _get_cuda_path()
    if cuda_path is not None:
        cuda_incl = os.path.join(cuda_path, "include")

    with as_file(files("cuda.cccl.include")) as f:
        cccl_incl = str(f)
    assert os.path.exists(cccl_incl)

    return IncludePaths(
        cuda=cuda_incl,
        libcudacxx=os.path.join(cccl_incl, "libcudacxx"),
        cub=cccl_incl,
        thrust=cccl_incl,
    )
