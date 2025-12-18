import functools
import weakref
from types import SimpleNamespace
from typing import Optional

from cuda.bindings import driver, runtime

try:
    from cuda.core import Device
    from cuda.core._utils.cuda_utils import handle_return
except ImportError:
    from cuda.core.experimental import Device
    from cuda.core.experimental._utils.cuda_utils import handle_return

from ..typing import StreamLike


@functools.cache
def _set_default_mempool_threshold(device_id: int):
    """
    Set the release threshold for the default memory pool on this device,
    if we haven't already done so. This prevents the driver from attempting
    to shrink the pool after every sync, which can be slow.
    """
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


def _finalize_buffer(ptr: int, stream_handle: Optional[int] = None):
    """Cleanup function for weakref finalizer."""
    if ptr != 0:
        try:
            handle_return(runtime.cudaFreeAsync(ptr, stream_handle))
        except Exception as e:
            # Don't raise in finalizer, just print warning
            print(f"Warning: Failed to free CUDA memory: {e}")


class TempStorageBuffer:
    """
    Simple wrapper type around the memory allocation used for temporary storage,
    exposing __cuda_array_interface__ and some other attributes for fast access.

    This implementation uses cuda.bindings.runtime.cudaMallocAsync and
    cudaFreeAsync for allocation and deallocation.
    """

    def __init__(self, size: int, stream: Optional[StreamLike] = None):
        # Get the current device
        dev = Device()

        stream_handle = stream.__cuda_stream__()[1] if stream is not None else None

        # Set the release threshold for the default memory pool on this device
        _set_default_mempool_threshold(dev.device_id)

        # Allocate memory using cudaMallocAsync
        device_ptr_int = handle_return(runtime.cudaMallocAsync(size, stream_handle))
        self._ptr = int(device_ptr_int)
        self._stream_handle = stream_handle
        self._size = size

        # attributes for fast path access in protocols.py
        self.nbytes = size
        self.data = SimpleNamespace(ptr=self._ptr)

        # Set up weakref finalizer for cleanup
        self._finalizer = weakref.finalize(
            self, _finalize_buffer, self._ptr, self._stream_handle
        )

    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self._ptr, False),
            "shape": (self._size,),
            "strides": (1,),
            "typestr": "|u1",
            "version": 3,
        }
