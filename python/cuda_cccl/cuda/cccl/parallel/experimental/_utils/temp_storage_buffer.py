from types import SimpleNamespace

import numpy as np

from cuda.core.experimental import Device, DeviceMemoryResource


class TempStorageBuffer:
    def __init__(self, size: int):
        mr = DeviceMemoryResource(Device().device_id)
        self._buf = mr.allocate(size)
        self._ptr = int(self._buf.handle)
        # other cuda_array_interface attributes
        self._shape = (size,)
        self._strides = (1,)
        self._dtype = np.uint8
        self.nbytes = size
        self.data = SimpleNamespace(
            ptr=self._ptr,
        )

    def __cuda_array_interface__(self):
        return {
            "data": (self._ptr, False),
            "shape": self._shape,
            "strides": self._strides,
            "typ": self._dtype,
            "version": 3,
        }
