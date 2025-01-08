from typing import Any

from typing_extensions import (
    Protocol,
)  # TODO: typing_extensions required for Python 3.7 docs env


class DeviceArrayLike(Protocol):
    """
    Objects representing a device array, having a `.__cuda_array_interface__`
    attribute.
    """

    __cuda_array_interface__: dict


GpuStruct = Any
GpuStruct.__doc__ = """\
    Type of instances of classes decorated with @gpu_struct.
"""
