import ctypes
from enum import IntEnum
from typing import Any, Optional

from typing_extensions import Buffer

class OpKind(IntEnum):
    _value_: int
    STATELESS = ...
    STATEFUL = ...
    PLUS = ...
    MINUS = ...
    MULTIPLIES = ...
    DIVIDES = ...
    MODULUS = ...
    EQUAL_TO = ...
    NOT_EQUAL_TO = ...
    GREATER = ...
    LESS = ...
    GREATER_EQUAL = ...
    LESS_EQUAL = ...
    LOGICAL_AND = ...
    LOGICAL_OR = ...
    LOGICAL_NOT = ...
    BIT_AND = ...
    BIT_OR = ...
    BIT_XOR = ...
    BIT_NOT = ...
    IDENTITY = ...
    NEGATE = ...
    MINIMUM = ...
    MAXIMUM = ...

class TypeEnum(IntEnum):
    _value_: int
    INT8 = ...
    INT16 = ...
    INT32 = ...
    INT64 = ...
    UINT8 = ...
    UINT16 = ...
    UINT32 = ...
    UINT64 = ...
    FLOAT16 = ...
    FLOAT32 = ...
    FLOAT64 = ...
    STORAGE = ...
    BOOLEAN = ...

class IteratorKind(IntEnum):
    _value_: int
    POINTER = ...
    ITERATOR = ...

class SortOrder(IntEnum):
    _value_: int
    ASCENDING = ...
    DESCENDING = ...

class InitKind(IntEnum):
    _value_: int
    NO_INIT = ...
    FUTURE_VALUE_INIT = ...
    VALUE_INIT = ...

class Determinism(IntEnum):
    _value_: int
    NOT_GUARANTEED = ...
    RUN_TO_RUN = ...
    GPU_TO_GPU = ...

class Op:
    def __init__(
        self,
        name: Optional[str] = ...,
        operator_type: OpKind = ...,
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
    def __init__(self, size: int, alignment: int, type_enum: TypeEnum): ...
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
        iterator_type: IteratorKind,
        advance_fn: Op,
        dereference_fn: Op,
        value_type: TypeInfo,
        state=None,
        host_advance_fn=None,
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
    def type(self) -> IteratorKind: ...
    @property
    def value_type(self) -> TypeInfo: ...
    def as_bytes(self) -> bytes: ...
    def is_kind_pointer(self) -> bool: ...
    def is_kind_iterator(self) -> bool: ...
    @property
    def host_advance_fn(self): ...
    @host_advance_fn.setter
    def host_advance_fn(self, value) -> None: ...

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
        determinism: Determinism,
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
    def compute_nondeterministic(
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
        init_type: TypeInfo,
        force_inclusive: bool,
        init_kind: InitKind,
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
    def compute_inclusive_future_value(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        binary_op: Op,
        h_init: Iterator,
        stream,
    ) -> int: ...
    def compute_exclusive_future_value(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        binary_op: Op,
        h_init: Iterator,
        stream,
    ) -> int: ...
    def compute_inclusive_no_init(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in: Iterator,
        d_out: Iterator,
        num_items: int,
        binary_op: Op,
        h_init: None,
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
    def __init__(
        self,
        d_in_keys: Iterator,
        d_in_items: Iterator,
        d_out_keys: Iterator,
        d_out_items: Iterator,
        binary_op: Op,
        info: CommonData,
    ): ...
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
    ) -> int: ...

# -----------------
# DeviceUniqueByKey
# -----------------

class DeviceUniqueByKeyBuildResult:
    def __init__(
        self,
        d_keys_in: Iterator,
        d_values_in: Iterator,
        d_keys_out: Iterator,
        d_values_out: Iterator,
        d_num_selected_out: Iterator,
        binary_op: Op,
        info: CommonData,
    ): ...
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
    ) -> int: ...

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

# ---------------
# DeviceHistogram
# ---------------

class DeviceHistogramBuildResult:
    def __init__(
        self,
        num_channels: int,
        num_active_channels: int,
        d_samples: Iterator,
        num_levels: int,
        d_histogram: Iterator,
        h_levels: Value,
        num_rows: int,
        row_stride_samples: int,
        is_evenly_segmented: bool,
        info: CommonData,
    ): ...
    def compute_even(
        self,
        d_samples: Iterator,
        d_histogram: Iterator,
        h_num_output_levels: Value,
        h_lower_level: Value,
        h_upper_level: Value,
        num_row_pixels: int,
        num_rows: int,
        row_stride_samples: int,
        stream,
    ) -> None: ...

# -----------------
# DeviceSegmentedSort
# -----------------

class DeviceSegmentedSortBuildResult:
    def __init__(self): ...
    def compute(
        self,
        temp_storage_ptr: int | None,
        temp_storage_nbytes: int,
        d_in_keys: Iterator,
        d_out_keys: Iterator,
        d_in_values: Iterator,
        d_out_values: Iterator,
        num_items: int,
        num_segments: int,
        d_begin_offsets: Iterator,
        d_end_offsets: Iterator,
        is_overwrite_okay: bool,
        selector: int,
        stream,
    ) -> tuple[int, int]: ...

# ---------------------
# DeviceThreeWayPartition
# ---------------------

class DeviceThreeWayPartitionBuildResult:
    def __init__(
        self,
        d_in: Iterator,
        d_first_part_out: Iterator,
        d_second_part_out: Iterator,
        d_unselected_out: Iterator,
        d_num_selected_out: Iterator,
        select_first_part_op: Op,
        select_second_part_op: Op,
        info: CommonData,
    ): ...
    def compute(
        self,
        d_in: Iterator,
        d_first_part_out: Iterator,
        d_second_part_out: Iterator,
        d_unselected_out: Iterator,
        d_num_selected_out: Iterator,
        num_items: int,
        stream,
    ) -> int: ...
