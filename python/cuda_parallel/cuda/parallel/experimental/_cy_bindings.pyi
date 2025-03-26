import ctypes
from typing import Any

from typing_extensions import Buffer

class IntEnum:
    def __init__(self, parent_class, name: str, value: int):
        pass

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def parent_class(self): ...
    @property
    def value(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other) -> bool: ...

class IntEnum_CCCLType:
    @property
    def INT8(self) -> IntEnum: ...
    @property
    def INT16(self) -> IntEnum: ...
    @property
    def INT32(self) -> IntEnum: ...
    @property
    def INT64(self) -> IntEnum: ...
    @property
    def UINT8(self) -> IntEnum: ...
    @property
    def UINT16(self) -> IntEnum: ...
    @property
    def UINT32(self) -> IntEnum: ...
    @property
    def UINT64(self) -> IntEnum: ...
    @property
    def FLOAT32(self) -> IntEnum: ...
    @property
    def FLOAT64(self) -> IntEnum: ...
    @property
    def STORAGE(self) -> IntEnum: ...

class IntEnum_OpKind:
    @property
    def STATELESS(self) -> IntEnum: ...
    @property
    def STATEFUL(self) -> IntEnum: ...

class IntEnum_IteratorKind:
    @property
    def POINTER(self) -> IntEnum: ...
    @property
    def ITERATOR(self) -> IntEnum: ...

TypeEnum: IntEnum_CCCLType
OpKind: IntEnum_OpKind
IteratorKind: IntEnum_IteratorKind

def pointer_as_bytes(int) -> bytes: ...
def is_TypeEnum(obj) -> bool: ...
def is_OpKind(obj) -> bool: ...
def is_IteratorKind(obj) -> bool: ...

class Op:
    def __init__(
        self,
        name: str = ...,
        operator_type: IntEnum = ...,
        ltoir=None,
        state=None,
        state_alignment: int = 1,
    ): ...
    @property
    def state(self) -> bytes: ...
    @state.setter
    def state(self, new_value: bytes) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def ltoir(self) -> bytes: ...
    @property
    def state_alignment(self) -> int: ...
    @property
    def state_typenum(self) -> int: ...
    def as_bytes(self) -> bytes: ...

class TypeInfo:
    def __init__(self, size: int, alignment: int, type_enum: IntEnum): ...
    @property
    def size(self) -> int: ...
    @property
    def alignment(self) -> int: ...
    @property
    def typenum(self) -> int: ...
    def as_bytes(self) -> bytes: ...

class Value:
    def __init__(self, type: TypeInfo, state: memoryview): ...
    @property
    def type(self) -> TypeInfo: ...
    @property
    def state(self) -> memoryview: ...
    @state.setter
    def state(self, new_value: memoryview) -> None: ...
    def as_bytes(self) -> bytes: ...

class IteratorStateView:
    def __init__(self, ptr, size: int, owner): ...
    @property
    def pointer(self) -> int: ...
    @property
    def reference(self): ...
    @property
    def size(self) -> int: ...

class Pointer:
    def __init__(self, arg): ...

def make_pointer_object(ptr: int | ctypes.c_void_p, owner: Any) -> Pointer: ...

class IteratorState(Buffer):
    def __init__(self, arg): ...
    @property
    def size(self) -> int: ...

class Iterator:
    def __init__(
        self,
        alignment: int,
        iterator_type: IntEnum,
        advance_fn: Op,
        dereference_fn: Op,
        value_type: TypeInfo,
        state=None,
    ):
        pass

    @property
    def advance_op(self) -> Op: ...
    @property
    def dereference_op(self) -> Op: ...
    @property
    def state(self): ...
    @state.setter
    def state(self, value) -> None: ...
    @property
    def type(self) -> IntEnum: ...
    def as_bytes(self) -> bytes: ...

class CommonData:
    def __init__(
        self,
        cc_major: int,
        cc_minor: int,
        cub_path: str,
        thrust_path: str,
        libcudacxx_path: str,
        ctk_path: str,
    ): ...
    @property
    def compute_capability(self) -> tuple[int, int]: ...
    @property
    def cub_path(self) -> str: ...
    @property
    def thrust_path(self) -> str: ...
    @property
    def libcudacxx_path(self) -> str: ...
    @property
    def ctk_path(self) -> str: ...

# ------------
# DeviceReduce
# ------------

class DeviceReduceBuildResult:
    def __init__(self): ...

def device_reduce_build(
    build_data: DeviceReduceBuildResult,
    d_in: Iterator,
    d_out: Iterator,
    binary_op: Op,
    h_init: Value,
    info: CommonData,
) -> int: ...
def device_reduce_cleanup(build_data: DeviceReduceBuildResult) -> int: ...
def device_reduce(
    build_data: DeviceReduceBuildResult,
    temp_storage_ptr: int | None,
    temp_storage_nbytes: int,
    d_in: Iterator,
    d_out: Iterator,
    num_items: int,
    binary_op: Op,
    h_init: Value,
    stream,
) -> tuple[int, int]: ...

# ----------
# DeviceScan
# ----------

class DeviceScanBuildResult:
    def __init__(self): ...

def device_scan_build(
    build_data: DeviceScanBuildResult,
    d_in: Iterator,
    d_out: Iterator,
    binary_op: Op,
    h_init: Value,
    force_inclusive: bool,
    info: CommonData,
) -> int: ...
def device_scan_cleanup(build_data: DeviceScanBuildResult) -> int: ...
def device_inclusive_scan(
    build_data: DeviceScanBuildResult,
    temp_storage_ptr: int | None,
    temp_storage_nbytes: int,
    d_in: Iterator,
    d_out: Iterator,
    num_items: int,
    binary_op: Op,
    h_init: Value,
    stream,
) -> tuple[int, int]: ...
def device_exclusive_scan(
    build_data: DeviceScanBuildResult,
    temp_storage_ptr: int | None,
    temp_storage_nbytes: int,
    d_in: Iterator,
    d_out: Iterator,
    num_items: int,
    binary_op: Op,
    h_init: Value,
    stream,
) -> tuple[int, int]: ...

# ---------------------
# DeviceSegmentedReduce
# ---------------------

class DeviceSegmentedReduceBuildResult:
    def __init__(self): ...

def device_segmented_reduce_build(
    build_data: DeviceSegmentedReduceBuildResult,
    d_in: Iterator,
    d_out: Iterator,
    start_offsets: Iterator,
    end_offsets: Iterator,
    binary_op: Op,
    h_init: Value,
    info: CommonData,
) -> int: ...
def device_segmented_reduce_cleanup(
    build_data: DeviceSegmentedReduceBuildResult,
) -> int: ...
def device_segmented_reduce(
    build_data: DeviceSegmentedReduceBuildResult,
    temp_storage_ptr: int | None,
    temp_storage_nbytes: int,
    d_in: Iterator,
    d_out: Iterator,
    num_items: int,
    start_offsets: Iterator,
    end_offsets: Iterator,
    binary_op: Op,
    h_init: Value,
    stream,
) -> tuple[int, int]: ...

# ---------------
# DeviceMergeSort
# ---------------

class DeviceMergeSortBuildResult:
    def __init__(self): ...

def device_merge_sort_build(
    build_data: DeviceMergeSortBuildResult,
    d_in_keys: Iterator,
    d_in_items: Iterator,
    d_out_keys: Iterator,
    d_out_itemss: Iterator,
    binary_op: Op,
    info: CommonData,
) -> int: ...
def device_merge_sort_cleanup(
    build_data: DeviceMergeSortBuildResult,
) -> int: ...
def device_merge_sort(
    build_data: DeviceMergeSortBuildResult,
    temp_storage_ptr: int,
    temp_storage_nbytes: int,
    d_in_keys: Iterator,
    d_in_items: Iterator,
    d_out_keys: Iterator,
    d_out_items: Iterator,
    num_items: int,
    binary_op: Op,
    stream,
) -> tuple[int, int]: ...

# -----------------
# DeviceUniqueByKey
# -----------------

class DeviceUniqueByKeyBuildResult:
    def __init__(self): ...

def device_unique_by_key_build(
    build_data: DeviceUniqueByKeyBuildResult,
    d_keys_in: Iterator,
    d_values_in: Iterator,
    d_keys_out: Iterator,
    d_values_out: Iterator,
    d_num_selected_out: Iterator,
    binary_op: Op,
    info: CommonData,
) -> int: ...
def device_unique_by_key_cleanup(
    build_data: DeviceUniqueByKeyBuildResult,
) -> int: ...
def device_unique_by_key(
    build_data: DeviceUniqueByKeyBuildResult,
    temp_storage_ptr: int,
    temp_storage_nbytes: int,
    d_keys_in: Iterator,
    d_values_in: Iterator,
    d_keys_out: Iterator,
    d_values_out: Iterator,
    d_num_selected_out: Iterator,
    binary_op: Op,
    num_items: int,
    stream,
) -> tuple[int, int]: ...
