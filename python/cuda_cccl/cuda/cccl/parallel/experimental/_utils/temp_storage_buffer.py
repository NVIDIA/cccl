import functools
from types import SimpleNamespace
from typing import Optional

import numpy as np

from cuda.bindings import driver
from cuda.core.experimental import Device, DeviceMemoryResource, Stream
from cuda.core.experimental._utils.cuda_utils import handle_return


class TempStorageBuffer:
    """
    Simple wrapper type around the memory allocation used for temporary storage,
    exposing __cuda_array_interface__ and some other attributes for fast access.
    """

    def __init__(self, size: int, stream: Optional[Stream] = None):
        # TODO: just use DeviceMemoryResource once cuda.core is updated
        # to increase the default mempool threshold by default
        dev = Device()

        # TODO: shouldn't this be the current device already?
        # the create_stream() call fails without this:
        dev.set_current()

        mr = _DeviceMemoryResourceWithIncreasedThreshold(dev.device_id)
        self._buf = mr.allocate(size, dev.create_stream(stream))
        self._ptr = int(self._buf.handle)

        # other cuda_array_interface attributes
        self._shape = (size,)
        self._strides = (1,)
        self._dtype = np.uint8

        # attributes for fast path access in protocols.py
        self.nbytes = size
        self.data = SimpleNamespace(
            ptr=self._ptr,
        )

    def __cuda_array_interface__(self):
        return {
            "data": (self._ptr, False),
            "shape": self._shape,
            "strides": self._strides,
            "typestr": "|u1",
            "version": 3,
        }


class _DeviceMemoryResourceWithIncreasedThreshold(DeviceMemoryResource):
    # cuda.core.experimental.DeviceMemoryResource currently uses
    # cuMallocFromPoolAsync with the default memory pool, with a default
    # release threshold of 0. This can be slow.
    #
    # This type sets the release threshold to UINT64_MAX, which prevents the
    # driver from attempting to shrink the pool after every sync.
    def __init__(self, device_id: int):
        # set the release threshold for the default memory pool
        # on this device, if we haven't already:
        _set_default_mempool_threshold(device_id)

        super().__init__(device_id)


@functools.cache
def _set_default_mempool_threshold(device_id: int):
    default_pool = handle_return(driver.cuDeviceGetDefaultMemPool(device_id))
    threshold = handle_return(
        driver.cuMemPoolGetAttribute(
            default_pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
        )
    )
    if int(threshold) == 0:
        handle_return(
            driver.cuMemPoolSetAttribute(
                default_pool,
                driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                driver.cuuint64_t(0xFFFFFFFFFFFFFFFF),
            )
        )
