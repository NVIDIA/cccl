import ctypes
from enum import Enum
from typing import Tuple

from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import call_build, get_bindings
from .._caching import cache_with_key
from .._utils import protocols
from ..typing import DeviceArrayLike


class SortOrder(Enum):
    ASCENDING = 0
    DESCENDING = 1


class DoubleBuffer:
    def __init__(self, d_current: DeviceArrayLike, d_alternate: DeviceArrayLike):
        self.d_buffers = [d_current, d_alternate]
        self.selector = 0

    def current(self):
        return self.d_buffers[self.selector]

    def alternate(self):
        return self.d_buffers[self.selector ^ 1]


def make_cache_key(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_keys: DeviceArrayLike | None,
    d_out_values: DeviceArrayLike | None,
    order: SortOrder,
):
    d_in_keys_array, d_in_values_array, d_out_keys_array, d_out_values_array = (
        _get_arrays(d_in_keys, d_in_values, d_out_keys, d_out_values)
    )

    d_in_keys_key = protocols.get_dtype(d_in_keys_array)
    d_in_values_key = (
        None if d_in_values_array is None else protocols.get_dtype(d_in_values_array)
    )
    d_out_keys_key = protocols.get_dtype(d_out_keys_array)
    d_out_values_key = (
        None if d_out_values_array is None else protocols.get_dtype(d_out_values_array)
    )

    return (
        d_in_keys_key,
        d_in_values_key,
        d_out_keys_key,
        d_out_values_key,
        order,
    )


def _get_arrays(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_keys: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
) -> Tuple:
    if isinstance(d_in_keys, DoubleBuffer):
        d_in_keys_array = d_in_keys.current()
        d_out_keys_array = d_in_keys.alternate()

        if d_in_values is not None:
            assert isinstance(d_in_values, DoubleBuffer)
            d_in_values_array = d_in_values.current()
            d_out_values_array = d_in_values.alternate()
        else:
            d_in_values_array = None
            d_out_values_array = None
    else:
        d_in_keys_array = d_in_keys
        d_in_values_array = d_in_values
        d_out_keys_array = d_out_keys
        d_out_values_array = d_out_values

    return d_in_keys_array, d_in_values_array, d_out_keys_array, d_out_values_array


class _RadixSort:
    def __init__(
        self,
        d_in_keys: DeviceArrayLike | DoubleBuffer,
        d_in_values: DeviceArrayLike | DoubleBuffer | None,
        d_out_keys: DeviceArrayLike | DoubleBuffer | None,
        d_out_values: DeviceArrayLike | None,
        order: SortOrder,
    ):
        # Referenced from __del__:
        self.build_result = None

        d_in_keys_array, d_in_values_array, d_out_keys_array, d_out_values_array = (
            _get_arrays(d_in_keys, d_in_values, d_out_keys, d_out_values)
        )

        self.d_in_keys_cccl = cccl.to_cccl_iter(d_in_keys_array)
        self.d_in_values_cccl = cccl.to_cccl_iter(d_in_values_array)
        self.d_out_keys_cccl = cccl.to_cccl_iter(d_out_keys_array)
        self.d_out_values_cccl = cccl.to_cccl_iter(d_out_values_array)

        # decomposer op is not supported for now
        self.decomposer_op = cccl.to_cccl_op(None, None)

        self.bindings = get_bindings()
        self.build_result = cccl.DeviceRadixSortBuildResult()

        error = call_build(
            self.bindings.cccl_device_radix_sort_build,
            ctypes.byref(self.build_result),
            cccl.SortOrder.ASCENDING
            if order is SortOrder.ASCENDING
            else cccl.SortOrder.DESCENDING,
            self.d_in_keys_cccl,
            self.d_in_values_cccl,
            self.decomposer_op,
            None,
        )

        self.device_radix_sort = (
            self.bindings.cccl_device_ascending_radix_sort
            if order is SortOrder.ASCENDING
            else self.bindings.cccl_device_descending_radix_sort
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building radix_sort")

    def __call__(
        self,
        temp_storage,
        d_in_keys: DeviceArrayLike | DoubleBuffer,
        d_in_values: DeviceArrayLike | DoubleBuffer | None,
        d_out_keys: DeviceArrayLike | None,
        d_out_values: DeviceArrayLike | None,
        num_items: int,
        begin_bit: int | None = None,
        end_bit: int | None = None,
        stream=None,
    ):
        set_state_fn = cccl.set_cccl_iterator_state

        d_in_keys_array, d_in_values_array, d_out_keys_array, d_out_values_array = (
            _get_arrays(d_in_keys, d_in_values, d_out_keys, d_out_values)
        )

        set_state_fn(self.d_in_keys_cccl, d_in_keys_array)
        if d_in_values_array is not None:
            set_state_fn(self.d_in_values_cccl, d_in_values_array)
        set_state_fn(self.d_out_keys_cccl, d_out_keys_array)
        if d_out_values_array is not None:
            set_state_fn(self.d_out_values_cccl, d_out_values_array)

        is_overwrite_okay = isinstance(d_in_keys, DoubleBuffer)

        stream_handle = protocols.validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]

        print("before")
        print(f"{begin_bit=}")
        print(f"{end_bit=}")

        if begin_bit is None:
            begin_bit = 0
        if end_bit is None:
            key_type = protocols.get_dtype(d_in_keys_array)
            end_bit = key_type.itemsize * 8

        print("after")
        print(f"{begin_bit=}")
        print(f"{end_bit=}")

        selector = ctypes.c_int(-1)

        error = self.device_radix_sort(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            self.d_in_keys_cccl,
            self.d_out_keys_cccl,
            self.d_in_values_cccl,
            self.d_out_values_cccl,
            self.decomposer_op,
            ctypes.c_ulonglong(num_items),
            ctypes.c_int(begin_bit),
            ctypes.c_int(end_bit),
            ctypes.c_bool(is_overwrite_okay),
            ctypes.byref(selector),
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error in radix sort")

        assert selector.value in {0, 1}

        if is_overwrite_okay and temp_storage is not None:
            assert isinstance(d_in_keys, DoubleBuffer)
            d_in_keys.selector = selector.value
            if d_in_values is not None:
                assert isinstance(d_in_values, DoubleBuffer)
                d_in_values.selector = selector.value

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        self.bindings.cccl_device_radix_sort_cleanup(ctypes.byref(self.build_result))


@cache_with_key(make_cache_key)
def radix_sort(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_keys: DeviceArrayLike | None,
    d_out_values: DeviceArrayLike | None,
    order: SortOrder,
):
    """Implements a device-wide radix sort using ``d_in_keys`` in the requested order.

    Example:
        Below, ``radix_sort`` is used to sort a sequence of keys. It also rearranges the values according to the keys' order.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_radix_sort_api.py
          :language: python
          :dedent:
          :start-after: example-begin radix-sort
          :end-before: example-end radix-sort

        Instead of passing in arrays directly, we can use a ``DoubleBuffer``, which requires less temporary storage but could overwrite the input arrays

        .. literalinclude:: ../../python/cuda_parallel/tests/test_radix_sort_api.py
          :language: python
          :dedent:
          :start-after: example-begin radix-sort-buffer
          :end-before: example-end radix-sort-buffer

    Args:
        d_in_keys: Device array or DoubleBuffer containing the input keys to be sorted
        d_in_values: Optional Device array or DoubleBuffer containing the input keys to be sorted
        d_in_keys: Device array to store the sorted keys
        d_in_values: Device array to store the sorted values
        op: Callable representing the comparison operator

    Returns:
        A callable object that can be used to perform the merge sort
    """
    return _RadixSort(d_in_keys, d_in_values, d_out_keys, d_out_values, order)
