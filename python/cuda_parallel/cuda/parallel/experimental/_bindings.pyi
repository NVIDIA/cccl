import ctypes
from typing import Any, Optional

from typing_extensions import Buffer

class IntEnumerationMember:
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

class Enumeration_CCCLType:
    @property
    def INT8(self) -> IntEnumerationMember: ...
    @property
    def INT16(self) -> IntEnumerationMember: ...
    @property
    def INT32(self) -> IntEnumerationMember: ...
    @property
    def INT64(self) -> IntEnumerationMember: ...
    @property
    def UINT8(self) -> IntEnumerationMember: ...
    @property
    def UINT16(self) -> IntEnumerationMember: ...
    @property
    def UINT32(self) -> IntEnumerationMember: ...
    @property
    def UINT64(self) -> IntEnumerationMember: ...
    @property
    def FLOAT32(self) -> IntEnumerationMember: ...
    @property
    def FLOAT64(self) -> IntEnumerationMember: ...
    @property
    def STORAGE(self) -> IntEnumerationMember: ...

class Enumeration_OpKind:
    @property
    def STATELESS(self) -> IntEnumerationMember: ...
    @property
    def STATEFUL(self) -> IntEnumerationMember: ...

class Enumeration_IteratorKind:
    @property
    def POINTER(self) -> IntEnumerationMember: ...
    @property
    def ITERATOR(self) -> IntEnumerationMember: ...

class Enumeration_SortOrder:
    @property
    def ASCENDING(self) -> IntEnumerationMember: ...
    @property
    def DESCENDING(self) -> IntEnumerationMember: ...

TypeEnum: Enumeration_CCCLType
OpKind: Enumeration_OpKind
IteratorKind: Enumeration_IteratorKind
SortOrder: Enumeration_SortOrder

def is_TypeEnum(obj) -> bool: ...
def is_OpKind(obj) -> bool: ...
def is_IteratorKind(obj) -> bool: ...
def is_SortOrder(obj) -> bool: ...

class Op:
    def __init__(
        self,
        name: Optional[str] = ...,
        operator_type: IntEnumerationMember = ...,
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
    def __init__(self, size: int, alignment: int, type_enum: IntEnumerationMember): ...
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
        iterator_type: IntEnumerationMember,
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
    def type(self) -> IntEnumerationMember: ...
    def as_bytes(self) -> bytes: ...
    def is_kind_pointer(self) -> bool: ...
    def is_kind_iterator(self) -> bool: ...

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
    def __init__(
        self,
        d_in: Iterator,
        d_out: Iterator,
        binary_op: Op,
        h_init: Value,
        info: CommonData,
    ): ...
    def compute(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        binary_op: Op,
        h_init: Value,
        stream,
    ) -> int: ...

# ----------
# DeviceScan
# ----------

class DeviceScanBuildResult:
    def __init__(
        self,
        d_in: Iterator,
        d_out: Iterator,
        binary_op: Op,
        h_init: Value,
        force_inclusive: bool,
        info: CommonData,
    ): ...
    def compute_inclusive(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        binary_op: Op,
        h_init: Value,
        stream,
    ) -> int: ...
    def compute_exclusive(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        binary_op: Op,
        h_init: Value,
        stream,
    ) -> int: ...

# ---------------------
# DeviceSegmentedReduce
# ---------------------

class DeviceSegmentedReduceBuildResult:
    def __init__(
        self,
        d_in: Iterator,
        d_out: Iterator,
        start_offsets: Iterator,
        end_offsets: Iterator,
        binary_op: Op,
        h_init: Value,
        info: CommonData,
    ): ...
    def compute(
        self,
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
    ) -> int: ...

# ---------------
# DeviceMergeSort
# ---------------

class DeviceMergeSortBuildResult:
    def __init__(self): ...
    def build(
        self,
        d_in_keys: Iterator,
        d_in_items: Iterator,
        d_out_keys: Iterator,
        d_out_items: Iterator,
        binary_op: Op,
        info: CommonData,
    ) -> int: ...
    def compute(
        self,
        temp_storage_ptr: int | None,
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
    def build(
        self,
        d_keys_in: Iterator,
        d_values_in: Iterator,
        d_keys_out: Iterator,
        d_values_out: Iterator,
        d_num_selected_out: Iterator,
        binary_op: Op,
        info: CommonData,
    ) -> int: ...
    def compute(
        self,
        temp_storage_ptr: int | None,
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

# -----------------
# DeviceRadixSort
# -----------------

class DeviceRadixSortBuildResult:
    def __init__(self): ...
    def compute(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_keys_in: Iterator,
        d_keys_out: Iterator,
        d_values_in: Iterator,
        d_values_out: Iterator,
        decomposer_op: Op,
        num_items: int,
        begin_bit: int,
        end_bit: int,
        is_overwrite_okay: bool,
        selector: int,
        stream,
    ) -> tuple[int, int]: ...

# --------------------
# DeviceUnaryTransform
# --------------------

class DeviceUnaryTransform:
    def __init__(
        self,
        d_in: Iterator,
        d_out: Iterator,
        op: Op,
        info: CommonData,
    ): ...
    def compute(
        self,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        stream,
    ) -> None: ...

# ---------------------
# DeviceBinaryTransform
# ---------------------

class DeviceBinaryTransform:
    def __init__(
        self,
        d_in1: Iterator,
        d_in2: Iterator,
        d_out: Iterator,
        op: Op,
        info: CommonData,
    ): ...
    def compute(
        self,
        d_in1: Iterator,
        d_in2: Iterator,
        d_out: Iterator,
        num_items: int,
        stream,
    ) -> None: ...
