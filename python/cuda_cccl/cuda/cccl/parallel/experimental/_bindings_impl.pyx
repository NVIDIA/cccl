# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

# Python signatures are declared in the companion Python stub file _bindings.pyi
# Make sure to update PYI with change to Python API to ensure that Python
# static type checker tools like mypy green-lights cuda.cccl.parallel

from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int64_t, uintptr_t
from cpython.bytes cimport PyBytes_FromStringAndSize

from cpython.buffer cimport (
    Py_buffer, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyBuffer_Release, PyObject_CheckBuffer, PyObject_GetBuffer
)
from cpython.pycapsule cimport (
    PyCapsule_CheckExact, PyCapsule_IsValid, PyCapsule_GetPointer
)

import ctypes

cdef extern from "<cuda.h>":
    cdef struct OpaqueCUstream_st
    cdef struct OpaqueCUkernel_st
    cdef struct OpaqueCUlibrary_st

    ctypedef int CUresult
    ctypedef OpaqueCUstream_st *CUstream
    ctypedef OpaqueCUkernel_st *CUkernel
    ctypedef OpaqueCUlibrary_st *CUlibrary


cdef extern from "cccl/c/types.h":
    ctypedef enum cccl_type_enum:
        CCCL_INT8
        CCCL_INT16
        CCCL_INT32
        CCCL_INT64
        CCCL_UINT8
        CCCL_UINT16
        CCCL_UINT32
        CCCL_UINT64
        CCCL_FLOAT32
        CCCL_FLOAT64
        CCCL_STORAGE

    ctypedef enum cccl_op_kind_t:
       CCCL_STATELESS
       CCCL_STATEFUL

    ctypedef enum cccl_iterator_kind_t:
       CCCL_POINTER
       CCCL_ITERATOR

    cdef struct cccl_type_info:
        size_t size
        size_t alignment
        cccl_type_enum type

    cdef enum cccl_op_code_type:
        CCCL_OP_LTOIR = 0
        CCCL_OP_CPP_SOURCE = 1

    cdef struct cccl_op_t:
        cccl_op_kind_t type
        const char* name
        const char* code
        size_t code_size
        cccl_op_code_type code_type
        size_t size
        size_t alignment
        void *state

    cdef struct cccl_value_t:
        cccl_type_info type
        void *state

    cdef union cccl_increment_t:
        int64_t signed_offset
        uint64_t unsigned_offset

    ctypedef void (*cccl_host_op_fn_ptr_t)(void *, cccl_increment_t) nogil

    cdef struct cccl_iterator_t:
        size_t size
        size_t alignment
        cccl_iterator_kind_t type
        cccl_op_t advance
        cccl_op_t dereference
        cccl_type_info value_type
        void *state
        cccl_host_op_fn_ptr_t host_advance

    ctypedef enum cccl_sort_order_t:
        CCCL_ASCENDING
        CCCL_DESCENDING

cdef void arg_type_check(
    str arg_name,
    object expected_type,
    object arg
) except *:
    if not isinstance(arg, expected_type):
        raise TypeError(
            f"Expected {arg_name} to have type '{expected_type}', "
            f"got '{type(arg)}'"
        )


cdef class IntEnumerationMember:
    """
    Represents enumeration member which records the enumeration it is a part of
    for type-checking.
    """
    cdef object parent_class
    cdef str enum_name
    cdef str attr_name
    cdef int attr_value

    def __cinit__(self, object parent_class, str enum_name, str attr_name, int attr_value):
        self.parent_class = parent_class
        self.enum_name = enum_name
        self.attr_name = attr_name
        self.attr_value = attr_value

    cdef str get_repr_str(self):
        return f"<{self.enum_name}.{self.attr_name}: {self.attr_value}>"

    def __repr__(self):
        return self.get_repr_str()

    def __str__(self):
        return self.get_repr_str()

    @property
    def parent_class(self):
        "Type of parental enumeration"
        return self.parent_class

    @property
    def name(self):
        "Name of the enumeration member"
        return self.attr_name

    @property
    def value(self):
        return self.attr_value

    def __int__(self):
        return int(self.attr_value)

    def __hash__(self):
        cdef object _cmp_key = (type(self), self.parent_class, <object>self.attr_value)
        return hash(_cmp_key)

    def __eq__(self, other):
        cdef IntEnumerationMember rhs
        if type(other) == type(self):
            rhs = <IntEnumerationMember>other
            return (self.attr_value == rhs.attr_value) and (self.parent_class == rhs.parent_class)
        else:
            return False


cdef class IntEnumerationBase:
    cdef str enum_name

    def __cinit__(self):
        self.enum_name = "Undefined"

    @property
    def __name__(self):
        return self.enum_name

    def __repr__(self):
        return f"<enum '{self.enum_name}'>"

    def __str__(self):
        return f"<enum '{self.enum_name}'>"


cdef class Enumeration_CCCLType(IntEnumerationBase):
    "Enumeration of CCCL types"
    cdef IntEnumerationMember _int8
    cdef IntEnumerationMember _int16
    cdef IntEnumerationMember _int32
    cdef IntEnumerationMember _int64
    cdef IntEnumerationMember _uint8
    cdef IntEnumerationMember _uint16
    cdef IntEnumerationMember _uint32
    cdef IntEnumerationMember _uint64
    cdef IntEnumerationMember _float32
    cdef IntEnumerationMember _float64
    cdef IntEnumerationMember _storage

    def __cinit__(self):
        self.enum_name = "TypeEnum"
        self._int8 = self.make_INT8()
        self._int16 = self.make_INT16()
        self._int32 = self.make_INT32()
        self._int64 = self.make_INT64()
        self._uint8 = self.make_UINT8()
        self._uint16 = self.make_UINT16()
        self._uint32 = self.make_UINT32()
        self._uint64 = self.make_UINT64()
        self._float32 = self.make_FLOAT32()
        self._float64 = self.make_FLOAT64()
        self._storage = self.make_STORAGE()

    @property
    def INT8(self):
        return self._int8

    @property
    def INT16(self):
        return self._int16

    @property
    def INT32(self):
        return self._int32

    @property
    def INT64(self):
        return self._int64

    @property
    def UINT8(self):
        return self._uint8

    @property
    def UINT16(self):
        return self._uint16

    @property
    def UINT32(self):
        return self._uint32

    @property
    def UINT64(self):
        return self._uint64

    @property
    def FLOAT32(self):
        return self._float32

    @property
    def FLOAT64(self):
        return self._float64

    @property
    def STORAGE(self):
        return self._storage

    cdef IntEnumerationMember make_INT8(self):
        cdef str prop_name = "INT8"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_INT8
        )

    cdef IntEnumerationMember make_INT16(self):
        cdef str prop_name = "INT16"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_INT16
        )

    cdef IntEnumerationMember make_INT32(self):
        cdef str prop_name = "INT32"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_INT32
        )

    cdef IntEnumerationMember make_INT64(self):
        cdef str prop_name = "INT64"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_INT64
        )

    cdef IntEnumerationMember make_UINT8(self):
        cdef str prop_name = "UINT8"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_UINT8
        )

    cdef IntEnumerationMember make_UINT16(self):
        cdef str prop_name = "UINT16"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_UINT16
        )

    cdef IntEnumerationMember make_UINT32(self):
        cdef str prop_name = "UINT32"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_UINT32
        )

    cdef IntEnumerationMember make_UINT64(self):
        cdef str prop_name = "UINT64"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_UINT64
        )


    cdef IntEnumerationMember make_FLOAT32(self):
        cdef str prop_name = "FLOAT32"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_FLOAT32
        )

    cdef IntEnumerationMember make_FLOAT64(self):
        cdef str prop_name = "FLOAT64"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_FLOAT64
        )


    cdef IntEnumerationMember make_STORAGE(self):
        cdef str prop_name = "STORAGE"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_type_enum.CCCL_STORAGE
        )


cdef class Enumeration_OpKind(IntEnumerationBase):
    "Enumeration of operator kinds"
    cdef IntEnumerationMember _stateless
    cdef IntEnumerationMember _stateful

    def __cinit__(self):
        self.enum_name = "OpKindEnum"
        self._stateless = self.make_STATELESS()
        self._stateful = self.make_STATEFUL()

    cdef IntEnumerationMember make_STATELESS(self):
        cdef str prop_name = "STATELESS"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_op_kind_t.CCCL_STATELESS
        )

    cdef IntEnumerationMember make_STATEFUL(self):
        cdef str prop_name = "STATEFUL"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_op_kind_t.CCCL_STATEFUL
        )


    @property
    def STATELESS(self):
        return self._stateless

    @property
    def STATEFUL(self):
        return self._stateful


cdef class Enumeration_IteratorKind(IntEnumerationBase):
    "Enumeration of iterator kinds"
    cdef IntEnumerationMember _pointer
    cdef IntEnumerationMember _iterator

    def __cinit__(self):
        self.enum_name = "IteratorKindEnum"
        self._pointer = self.make_POINTER()
        self._iterator = self.make_ITERATOR()

    cdef IntEnumerationMember make_POINTER(self):
        cdef str prop_name = "POINTER"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_iterator_kind_t.CCCL_POINTER
        )

    cdef IntEnumerationMember make_ITERATOR(self):
        cdef str prop_name = "ITERATOR"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_iterator_kind_t.CCCL_ITERATOR
        )

    @property
    def POINTER(self):
        return self._pointer

    @property
    def ITERATOR(self):
        return self._iterator

cdef class Enumeration_SortOrder(IntEnumerationBase):
    "Enumeration of sort orders (ascending or descending)"
    cdef IntEnumerationMember _ascending
    cdef IntEnumerationMember _descending

    def __cinit__(self):
        self.enum_name = "SortOrder"
        self._ascending = self.make_ASCENDING()
        self._descending = self.make_DESCENDING()

    cdef IntEnumerationMember make_ASCENDING(self):
        cdef str prop_name = "ASCENDING"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_sort_order_t.CCCL_ASCENDING
        )

    cdef IntEnumerationMember make_DESCENDING(self):
        cdef str prop_name = "DESCENDING"
        return IntEnumerationMember(
            type(self),
            self.enum_name,
            prop_name,
            cccl_sort_order_t.CCCL_DESCENDING
        )

    @property
    def ASCENDING(self):
        return self._ascending

    @property
    def DESCENDING(self):
        return self._descending


TypeEnum = Enumeration_CCCLType()
OpKind = Enumeration_OpKind()
IteratorKind = Enumeration_IteratorKind()
SortOrder = Enumeration_SortOrder()

cpdef bint is_TypeEnum(IntEnumerationMember attr):
    "Return True if attribute is a member of TypeEnum enumeration"
    return attr.parent_class is Enumeration_CCCLType


cpdef bint is_OpKind(IntEnumerationMember attr):
    "Return True if attribute is a member of OpKind enumeration"
    return attr.parent_class is Enumeration_OpKind


cpdef bint is_IteratorKind(IntEnumerationMember attr):
    "Return True if attribute is a member of IteratorKind enumeration"
    return attr.parent_class is Enumeration_IteratorKind

cpdef bint is_SortOrder(IntEnumerationMember attr):
    "Return True if attribute is a member of SortOrder enumeration"
    return attr.parent_class is Enumeration_SortOrder


cdef void _validate_alignment(int alignment) except *:
    """
    Alignment must be positive integer and a power of two
    that can be represented by uint32_t type.
    """
    cdef uint32_t val
    if alignment < 1:
        raise ValueError(
            "Alignment must be non-negative, "
            f"got {alignment}."
        )
    val = <uint32_t>alignment
    if (val & (val - 1)) != 0:
        raise ValueError(
            "Alignment must be a power of two, "
            f"got {alignment}"
        )


cdef class Op:
    """
    Represents CCCL Operation

    Args:
        name (str):
            Name of the operation
        operator_type (OpKind):
            Whether operator is stateless or stateful
        ltoir (bytes):
            The LTOIR for the operation compiled for device
        state (bytes, optional):
            State for the stateful operation.
        state_alignment (int, optional):
            Alignment of the state struct. Default: `1`.
    """
    # need Python owner of memory used for operator name
    cdef bytes op_encoded_name
    cdef bytes code_bytes
    cdef bytes state_bytes
    cdef cccl_op_t op_data


    cdef void _set_members(self, cccl_op_kind_t op_type, str name, bytes lto_ir, bytes state, int state_alignment):
        memset(&self.op_data, 0, sizeof(cccl_op_t))
        # Reference Python objects in the class to ensure lifetime
        self.op_encoded_name = name.encode("utf-8")
        self.code_bytes = lto_ir
        self.state_bytes = state
        # set fields of op_data struct
        self.op_data.type = op_type
        self.op_data.name = <const char *>self.op_encoded_name
        self.op_data.code = <const char *>lto_ir
        self.op_data.code_size = len(lto_ir)
        self.op_data.code_type = cccl_op_code_type.CCCL_OP_LTOIR
        self.op_data.size = len(state)
        self.op_data.alignment = state_alignment
        self.op_data.state = <void *><const char *>state


    def __cinit__(self, /, *, name = None, operator_type = OpKind.STATELESS, ltoir = None, state = None, state_alignment = 1):
        if name is None and ltoir is None:
            name = ""
            ltoir = b""
        if state is None:
            state = b""
        arg_type_check(arg_name="name", expected_type=str, arg=name)
        arg_type_check(arg_name="ltoir", expected_type=bytes, arg=ltoir)
        arg_type_check(arg_name="state", expected_type=bytes, arg=state)
        arg_type_check(arg_name="state_alignment", expected_type=int, arg=state_alignment)
        arg_type_check(arg_name="operator_type", expected_type=IntEnumerationMember, arg=operator_type)
        if not is_OpKind(operator_type):
            raise TypeError(
                f"The operator_type argument should be an enumerator of operator kinds"
            )
        _validate_alignment(state_alignment)
        self._set_members(
            <cccl_op_kind_t> operator_type.value,
            <str> name,
            <bytes> ltoir,
            <bytes> state,
            <int> state_alignment
        )


    cdef void set_state(self, bytes state):
        self.state_bytes = state
        self.op_data.state = <void *><const char *>state

    @property
    def state(self):
       return self.state_bytes

    @state.setter
    def state(self, bytes new_value):
        self.set_state(<bytes>new_value)

    @property
    def name(self):
        return self.op_encoded_name.decode("utf-8")

    @property
    def ltoir(self):
        # Backward compatibility property
        return self.code_bytes

    @property
    def code(self):
        return self.code_bytes

    @property
    def state_alignment(self):
        return self.op_data.alignment

    @property
    def state_typenum(self):
        return self.op_data.type

    def as_bytes(self):
        "Debugging utility to view memory content of library struct"
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.op_data))
        memcpy(&mem_view[0], &self.op_data, sizeof(self.op_data))
        return bytes(mem_view)


cdef class TypeInfo:
    """
    Represents CCCL type info structure

    Args:
        size (int):
            Size of the type in bytes.
        alignment (int):
            Alignment of the type in bytes.
        type_enum (IntEnumerationMember):
            Enumeration member identifying the type.
    """
    cdef cccl_type_info type_info

    def __cinit__(self, int size, int alignment, IntEnumerationMember type_enum):
        if size < 1:
            raise ValueError("Size argument must be positive")
        _validate_alignment(alignment)
        if not is_TypeEnum(type_enum):
            raise TypeError(
                f"The type argument should be enum of CCCL types"
            )
        self.type_info.size = size
        self.type_info.alignment = alignment
        self.type_info.type = <cccl_type_enum> type_enum.value

    @property
    def size(self):
        return self.type_info.size

    @property
    def alignment(self):
        return self.type_info.alignment

    @property
    def typenum(self):
        return self.type_info.type

    def as_bytes(self):
        "Debugging utility to view memory content of library struct"
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.type_info))
        memcpy(&mem_view[0], &self.type_info, sizeof(self.type_info))
        return bytes(mem_view)


cdef class Value:
    """
    Represents CCCL value structure

    Args:
        value_type (TypeInfo):
            type descriptor
        state (object):
            state of the value type. Object is expected to
            implement Python buffer protocol and be able to provide
            simple contiguous array of type `uint8_t`.
    """
    cdef uint8_t[::1] state_obj
    cdef TypeInfo value_type
    cdef cccl_value_t value_data;

    def __cinit__(self, TypeInfo value_type, uint8_t[::1] state):
        self.state_obj = state
        self.value_type = value_type
        self.value_data.type = value_type.type_info
        self.value_data.state = <void *>&state[0]

    @property
    def type(self):
        return self.value_type

    @property
    def state(self):
        return self.state_obj

    @state.setter
    def state(self, uint8_t[::1] new_value):
        if (len(self.state_obj) == len(new_value)):
            self.state_obj = new_value
            self.value_data.state = <void *>&self.state_obj[0]
        else:
            raise ValueError("Size mismatch")

    def as_bytes(self):
        "Debugging utility to view memory of native struct"
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.value_data))
        memcpy(&mem_view[0], &self.value_data, sizeof(self.value_data))
        return bytes(mem_view)


cdef void ensure_buffer(object o) except *:
    if not PyObject_CheckBuffer(o):
        raise TypeError(
            "Object with buffer protocol expected, "
            f"got {type(o)}"
        )


cdef void * get_buffer_pointer(object o, size_t *size):
    cdef int status = 0
    cdef void *ptr = NULL
    cdef Py_buffer view

    status = PyObject_GetBuffer(o, &view, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
    if status != 0:  # pragma: no cover
        size[0] = 0
        raise RuntimeError(
            "Can not access simple contiguous buffer"
        )

    ptr = view.buf
    if size is not NULL:
        size[0] = <size_t>view.len
    PyBuffer_Release(&view)

    return ptr


cdef void * ctypes_typed_pointer_payload_ptr(object ctypes_typed_ptr):
    "Get pointer to the value buffer represented by ctypes.pointer(ctypes_val)"
    cdef size_t size = 0
    cdef size_t *ptr_ref = NULL
    ensure_buffer(ctypes_typed_ptr)
    ptr_ref = <size_t *>get_buffer_pointer(ctypes_typed_ptr, &size)
    return <void *>(ptr_ref[0])


cdef void * ctypes_value_ptr(object ctypes_cdata):
    "Get pointer to the value buffer behind ctypes_val"
    cdef size_t size = 0
    ensure_buffer(ctypes_cdata)
    return get_buffer_pointer(ctypes_cdata, &size)


cdef inline void * int_as_ptr(size_t ptr_val):
    return <void *>(ptr_val)


cdef class StateBase:
    cdef void *ptr
    cdef object ref

    def __cinit__(self):
        self.ptr = NULL
        self.ref = None

    cdef inline void set_state(self, void *ptr, object ref):
        self.ptr = ptr
        self.ref = ref

    @property
    def pointer(self):
        return <size_t>self.ptr

    @property
    def reference(self):
        return self.ref


cdef class Pointer(StateBase):
    "Represents the pointer value"

    def __cinit__(self, arg):
        cdef void *ptr
        cdef object ref

        if isinstance(arg, int):
            ptr = int_as_ptr(arg)
            ref = None
        elif isinstance(arg, ctypes._Pointer):
            ptr = ctypes_typed_pointer_payload_ptr(arg)
            ref = arg
        elif isinstance(arg, ctypes.c_void_p):
            ptr = int_as_ptr(arg.value)
            ref = arg
        else:
            raise TypeError(
                "Expect ctypes pointer, integers, or PointerProxy, "
                f"got type {type(arg)}"
            )
        self.set_state(ptr, ref)


def make_pointer_object(ptr, owner):
    cdef Pointer res = Pointer(0)

    if isinstance(ptr, int):
        res.ptr = int_as_ptr(ptr)
    elif isinstance(ptr, ctypes.c_void_p):
        res.ptr = int_as_ptr(ptr.value)
    else:
        raise TypeError(
            "First argument must be an integer, or ctypes.c_void_p, "
            f"got {type(ptr)}"
        )
    res.ref = owner
    return res


cdef class IteratorState(StateBase):
    "Represents blob referenced by pointer"
    cdef size_t state_nbytes

    def __cinit__(self, arg):
        cdef size_t buffer_size = 0
        cdef void *ptr = NULL
        cdef object ref = None

        super().__init__()
        if isinstance(arg, ctypes._Pointer):
            ptr = ctypes_typed_pointer_payload_ptr(arg)
            ref = arg.contents
            self.state_nbytes = ctypes.sizeof(ref)
        elif PyObject_CheckBuffer(arg):
            ptr = get_buffer_pointer(arg, &buffer_size)
            ref = arg
            self.state_nbytes = buffer_size
        else:
            raise TypeError(
                "Expected a ctypes pointer with content, or object of type bytes or bytearray, "
                f"got type {type(arg)}"
            )
        self.set_state(ptr, ref)

    cdef inline size_t get_size(self):
        return self.state_nbytes

    @property
    def size(self):
        return self.state_nbytes

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t cast_size = <Py_ssize_t>self.state_nbytes
        buffer.buf = <char *>self.ptr
        buffer.obj = self
        buffer.len = cast_size
        buffer.readonly = 0
        buffer.itemsize = 1
        buffer.format = "B"  # unsigned char
        buffer.ndim = 1
        buffer.shape = <Py_ssize_t *>&self.state_nbytes
        buffer.strides = &buffer.itemsize
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cdef const char *function_ptr_capsule_name = "void (void *, cccl_increment_t)";

cdef bint is_function_pointer_capsule(object o) noexcept:
    """
    Returns non-zero if input is a valid capsule with
    name 'void (void *, cccl_increment_t)'.
    """
    return (
        PyCapsule_CheckExact(o) and
        PyCapsule_IsValid(o, function_ptr_capsule_name)
    )


cdef inline void* get_function_pointer_from_capsule(object cap) except *:
    return PyCapsule_GetPointer(cap, function_ptr_capsule_name)


cdef cccl_host_op_fn_ptr_t unbox_host_advance_fn(object host_fn_obj) except *:
    cdef void *fn_ptr = NULL
    if isinstance(host_fn_obj, ctypes._CFuncPtr):
        # the _CFuncPtr object encapsulates a pointer to the function pointer
        fn_ptr = ctypes_typed_pointer_payload_ptr(host_fn_obj)
        return <cccl_host_op_fn_ptr_t>fn_ptr

    if isinstance(host_fn_obj, int):
        fn_ptr = <void *><uintptr_t>host_fn_obj
        return <cccl_host_op_fn_ptr_t>fn_ptr

    if isinstance(host_fn_obj, ctypes.c_void_p):
        fn_ptr = <void *><uintptr_t>host_fn_obj.value
        return <cccl_host_op_fn_ptr_t>fn_ptr

    if is_function_pointer_capsule(host_fn_obj):
        fn_ptr = get_function_pointer_from_capsule(host_fn_obj)
        return <cccl_host_op_fn_ptr_t>fn_ptr

    raise TypeError(
        "Expected ctypes function pointer, ctypes.c_void_p, integer or a named capsule, "
        f"got {type(host_fn_obj)}"
    )


cdef class Iterator:
    """
    Represents CCCL iterator.

    Args:
        alignment (int):
            Alignment of the iterator state
        iterator_type (IntEnumerationMember):
            The type of iterator, `IteratorKind.POINTER` or
            `IteratorKind.ITERATOR`
        advance_fn (Op):
            Descriptor for user-defined `advance` function
            compiled for device
        dereference_fn (Op):
            Descriptor for user-defined `dereference` or `assign`
            function compiled for device
        value_type (TypeInfo):
            Descriptor of the type addressed by the iterator
        state (object, optional):
            Python object for the state of the iterator. For iterators of
            type `ITERATOR` the state object is expected to implement Python
            buffer protocol for SIMPLE 1d buffer of type unsigned byte.
            For iterators of type `POINTER` the state may be an integer convertible
            to `uintptr_t`, or a `ctypes` pointer (typed or untyped).
            Value `None` represents absence of iterator state.
        host_advance_fn (object, optional):
            Python object for host callable function to advance state by a given
            increment. The argument may only be set for iterators of type
            `IteratorKind.ITERATOR` and raise an exception otherwise. Supported
            types are `int` or `ctypes.c_void_p` (raw pointer), ctypes function
            pointer, or a Python capsule with name `"void *(void *, cccl_increment_t)"`.
    """
    cdef Op advance
    cdef Op dereference
    cdef object state_obj
    cdef object host_advance_obj
    cdef cccl_iterator_t iter_data

    def __cinit__(self,
        int alignment,
        IntEnumerationMember iterator_type,
        Op advance_fn,
        Op dereference_fn,
        TypeInfo value_type,
        state=None,
        host_advance_fn=None
    ):
        cdef cccl_iterator_kind_t it_kind
        _validate_alignment(alignment)
        if not is_IteratorKind(iterator_type):
            raise TypeError("iterator_type must describe iterator kind")
        it_kind = iterator_type.value
        if it_kind == cccl_iterator_kind_t.CCCL_POINTER:
            if state is None:
                self.state_obj = None
                self.iter_data.size = 0
                self.iter_data.state = NULL
            elif isinstance(state, int):
                self.state_obj = None
                self.iter_data.size = 0
                self.iter_data.state = int_as_ptr(state)
            elif isinstance(state, Pointer):
                self.state_obj = state.reference
                self.iter_data.size = 0
                self.iter_data.state = (<Pointer>state).ptr
            else:
                raise TypeError(
                    "Expect for Iterator of kind POINTER, state must have type Pointer or int, "
                    f"got {type(state)}"
                )
            if host_advance_fn is not None:
                raise ValueError(
                    "host_advance_fn must be set to None for iterators of kind POINTER"
                )
            self.iter_data.host_advance = NULL
            self.host_advance_obj = None
        elif it_kind == cccl_iterator_kind_t.CCCL_ITERATOR:
            if state is None:
                self.state_obj = None
                self.iter_data.size = 0
                self.iter_data.state = NULL
            elif isinstance(state, IteratorState):
                self.state_obj = state.reference
                self.iter_data.size = (<IteratorState>state).size
                self.iter_data.state = (<IteratorState>state).ptr
            else:
                raise TypeError(
                    "For Iterator of kind ITERATOR, state must have type IteratorState, "
                    f"got type {type(state)}"
                )
            if host_advance_fn is not None:
                self.iter_data.host_advance = unbox_host_advance_fn(host_advance_fn)
                self.host_advance_obj = host_advance_fn
            else:
                self.iter_data.host_advance = NULL
                self.host_advance_obj = None
        else:  # pragma: no cover
            raise ValueError("Unrecognized iterator kind")
        self.advance = advance_fn
        self.dereference = dereference_fn
        self.iter_data.alignment = alignment
        self.iter_data.type = <cccl_iterator_kind_t> it_kind
        self.iter_data.advance = self.advance.op_data
        self.iter_data.dereference = self.dereference.op_data
        self.iter_data.value_type = value_type.type_info

    @property
    def advance_op(self):
        return self.advance

    @property
    def dereference_or_assign_op(self):
        return self.dereference

    @property
    def state(self):
        if self.iter_data.type == cccl_iterator_kind_t.CCCL_POINTER:
            return <size_t>self.iter_data.state
        else:
            return self.state_obj

    @state.setter
    def state(self, new_value):
        cdef ssize_t state_sz = 0
        cdef size_t ptr = 0
        cdef cccl_iterator_kind_t it_kind = self.iter_data.type
        if it_kind == cccl_iterator_kind_t.CCCL_POINTER:
            if isinstance(new_value, Pointer):
                self.state_obj = (<Pointer>new_value).ref
                self.iter_data.size = state_sz
                self.iter_data.state = (<Pointer>new_value).ptr
            elif isinstance(new_value, int):
                self.state_obj = None
                self.iter_data.size = state_sz
                self.iter_data.state = int_as_ptr(new_value)
            elif new_value is None:
                self.state_obj = None
                self.iter_data.size = 0
                self.iter_data.state = NULL
            else:
                raise TypeError(
                    "For iterator with type POINTER, state value must have type int or type Pointer, "
                    f"got type {type(new_value)}"
                )
        elif it_kind == cccl_iterator_kind_t.CCCL_ITERATOR:
            if isinstance(new_value, IteratorState):
                self.state_obj = new_value.reference
                self.iter_data.size = (<IteratorState>new_value).size
                self.iter_data.state = (<IteratorState>new_value).ptr
            elif isinstance(new_value, Pointer):
                self.state_obj = new_value.reference
                if self.iter_data.size == 0:
                    raise ValueError("Assigning incomplete state value to iterator without state size information")
                self.iter_data.state = (<Pointer>new_value).ptr
            elif PyObject_CheckBuffer(new_value):
                self.iter_data.state = get_buffer_pointer(new_value, &self.iter_data.size)
                self.state_obj = new_value
            elif new_value is None:
                self.state_obj = None
                self.iter_data.size = 0
                self.iter_data.state = NULL
            else:
                raise TypeError(
                    "For iterator with type ITERATOR, state value must have type IteratorState or type bytes, "
                    f"got type {type(new_value)}"
                )
        else:
            raise TypeError("The new value should be an integer for iterators of POINTER kind, and bytes for ITERATOR kind")

    @property
    def type(self):
        cdef cccl_iterator_kind_t it_kind = self.iter_data.type
        if it_kind == cccl_iterator_kind_t.CCCL_POINTER:
            return IteratorKind.POINTER
        else:
            return IteratorKind.ITERATOR

    def is_kind_pointer(self):
        cdef cccl_iterator_kind_t it_kind = self.iter_data.type
        return (it_kind == cccl_iterator_kind_t.CCCL_POINTER)

    def is_kind_iterator(self):
        cdef cccl_iterator_kind_t it_kind = self.iter_data.type
        return (it_kind == cccl_iterator_kind_t.CCCL_ITERATOR)

    def as_bytes(self):
        "Debugging ulitity to get memory view into library struct"
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.iter_data))
        memcpy(&mem_view[0], &self.iter_data, sizeof(self.iter_data))
        return bytes(mem_view)

    @property
    def host_advance_fn(self):
        return self.host_advance_obj

    @host_advance_fn.setter
    def host_advance_fn(self, func):
        if (self.iter_data.type == cccl_iterator_kind_t.CCCL_ITERATOR):
            if func is not None:
                self.iter_data.host_advance = unbox_host_advance_fn(func)
                self.host_advance_obj = func
            else:
                self.iter_data.host_advance = NULL
                self.host_advance_obj = None
        else:
            raise ValueError


cdef class CommonData:
    cdef int cc_major
    cdef int cc_minor
    cdef bytes encoded_cub_path
    cdef bytes encoded_thrust_path
    cdef bytes encoded_libcudacxx_path
    cdef bytes encoded_ctk_path

    def __cinit__(self, int cc_major, int cc_minor, str cub_path, str thrust_path, str libcudacxx_path, str ctk_path):
        self.cc_major = cc_major
        self.cc_minor = cc_minor
        self.encoded_cub_path = cub_path.encode("utf-8")
        self.encoded_thrust_path = thrust_path.encode("utf-8")
        self.encoded_libcudacxx_path = libcudacxx_path.encode("utf-8")
        self.encoded_ctk_path = ctk_path.encode("utf-8")

    cdef inline int get_cc_major(self):
        return self.cc_major

    cdef inline int get_cc_minor(self):
        return self.cc_minor

    cdef inline const char * cub_path_get_c_str(self):
        return <const char *>self.encoded_cub_path if self.encoded_cub_path else NULL

    cdef inline const char * thrust_path_get_c_str(self):
        return <const char *>self.encoded_thrust_path if self.encoded_thrust_path else NULL

    cdef inline const char * libcudacxx_path_get_c_str(self):
        return <const char *>self.encoded_libcudacxx_path if self.encoded_libcudacxx_path else NULL

    cdef inline const char * ctk_path_get_c_str(self):
        return <const char *>self.encoded_ctk_path if self.encoded_ctk_path else NULL

    @property
    def compute_capability(self):
        return (self.cc_major, self.cc_minor)

    @property
    def cub_path(self):
        return self.encoded_cub_path.decode("utf-8")

    @property
    def ctk_path(self):
        return self.encoded_ctk_path.decode("utf-8")

    @property
    def thrust_path(self):
        return self.encoded_thrust_path.decode("utf-8")

    @property
    def libcudacxx_path(self):
        return self.encoded_libcudacxx_path.decode("utf-8")

# --------------
#   DeviceReduce
# --------------

cdef extern from "cccl/c/reduce.h":
    cdef struct cccl_device_reduce_build_result_t 'cccl_device_reduce_build_result_t':
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_reduce_build(
        cccl_device_reduce_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_reduce(
        cccl_device_reduce_build_result_t,
        void *,
        size_t *,
        cccl_iterator_t,
        cccl_iterator_t,
        uint64_t,
        cccl_op_t,
        cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_reduce_cleanup(
        cccl_device_reduce_build_result_t*
    ) nogil


cdef class DeviceReduceBuildResult:
    cdef cccl_device_reduce_build_result_t build_data

    def __cinit__(
        DeviceReduceBuildResult self,
        Iterator d_in,
        Iterator d_out,
        Op op,
        Value h_init,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()
        memset(&self.build_data, 0, sizeof(cccl_device_reduce_build_result_t))

        with nogil:
            status = cccl_device_reduce_build(
                &self.build_data,
                d_in.iter_data,
                d_out.iter_data,
                op.op_data,
                h_init.value_data,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(
                f"Failed building reduce, error code: {status}"
            )

    def __dealloc__(DeviceReduceBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_reduce_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during reduce result cleanup")

    cpdef int compute(
        DeviceReduceBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_in,
        Iterator d_out,
        size_t num_items,
        Op op,
        Value h_init,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><uintptr_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_reduce(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_in.iter_data,
                d_out.iter_data,
                <uint64_t>num_items,
                op.op_data,
                h_init.value_data,
                c_stream
            )
        if status != 0:
            raise RuntimeError(
                f"Failed executing reduce, error code: {status}"
            )
        return storage_sz

    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]

# ------------
#   DeviceScan
# ------------


cdef extern from "cccl/c/scan.h":
    ctypedef bint _Bool

    cdef struct cccl_device_scan_build_result_t 'cccl_device_scan_build_result_t':
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_scan_build(
        cccl_device_scan_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        _Bool,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_exclusive_scan(
        cccl_device_scan_build_result_t,
        void *,
        size_t *,
        cccl_iterator_t,
        cccl_iterator_t,
        uint64_t,
        cccl_op_t,
        cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_inclusive_scan(
        cccl_device_scan_build_result_t,
        void *,
        size_t *,
        cccl_iterator_t,
        cccl_iterator_t,
        uint64_t,
        cccl_op_t,
        cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_scan_cleanup(
        cccl_device_scan_build_result_t*
    ) nogil


cdef class DeviceScanBuildResult:
    cdef cccl_device_scan_build_result_t build_data

    def __cinit__(
        DeviceScanBuildResult self,
        Iterator d_in,
        Iterator d_out,
        Op op,
        Value h_init,
        bint force_inclusive,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()
        memset(&self.build_data, 0, sizeof(cccl_device_scan_build_result_t))

        with nogil:
            status = cccl_device_scan_build(
                &self.build_data,
                d_in.iter_data,
                d_out.iter_data,
                op.op_data,
                h_init.value_data,
                force_inclusive,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(f"Error {status} building scan")

    def __dealloc__(DeviceScanBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_scan_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during scan result cleanup")

    cpdef int compute_inclusive(
        DeviceScanBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_in,
        Iterator d_out,
        size_t num_items,
        Op op,
        Value h_init,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><uintptr_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_inclusive_scan(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_in.iter_data,
                d_out.iter_data,
                <uint64_t>num_items,
                op.op_data,
                h_init.value_data,
                c_stream
            )
        if status != 0:
            raise RuntimeError(
                f"Failed executing inclusive scan, error code: {status}"
            )
        return storage_sz

    cpdef int compute_exclusive(
        DeviceScanBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_in,
        Iterator d_out,
        size_t num_items,
        Op op,
        Value h_init,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><uintptr_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_exclusive_scan(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_in.iter_data,
                d_out.iter_data,
                <uint64_t>num_items,
                op.op_data,
                h_init.value_data,
                c_stream
            )
        if status != 0:
            raise RuntimeError(
                f"Failed executing exclusive scan, error code: {status}"
            )
        return storage_sz

    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]

# -----------------------
#   DeviceSegmentedReduce
# -----------------------


cdef extern from "cccl/c/segmented_reduce.h":
    cdef struct cccl_device_segmented_reduce_build_result_t 'cccl_device_segmented_reduce_build_result_t':
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_segmented_reduce_build(
        cccl_device_segmented_reduce_build_result_t*,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_segmented_reduce(
        cccl_device_segmented_reduce_build_result_t,
        void *,
        size_t *,
        cccl_iterator_t,
        cccl_iterator_t,
        uint64_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_op_t,
        cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_segmented_reduce_cleanup(
        cccl_device_segmented_reduce_build_result_t* bld_ptr
    ) nogil


cdef class DeviceSegmentedReduceBuildResult:
    cdef cccl_device_segmented_reduce_build_result_t build_data

    def __cinit__(
        DeviceSegmentedReduceBuildResult self,
        Iterator d_in,
        Iterator d_out,
        Iterator start_offsets,
        Iterator end_offsets,
        Op op,
        Value h_init,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        memset(&self.build_data, 0, sizeof(cccl_device_segmented_reduce_build_result_t))
        with nogil:
            status = cccl_device_segmented_reduce_build(
                &self.build_data,
                d_in.iter_data,
                d_out.iter_data,
                start_offsets.iter_data,
                end_offsets.iter_data,
                op.op_data,
                h_init.value_data,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(
                f"Failed building segmented_reduce, error code: {status}"
            )

    def __dealloc__(DeviceSegmentedReduceBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_segmented_reduce_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during segmented_reduce result cleanup")

    cpdef int compute(
        DeviceSegmentedReduceBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_in,
        Iterator d_out,
        size_t num_items,
        Iterator start_offsets,
        Iterator end_offsets,
        Op op,
        Value h_init,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><uintptr_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_segmented_reduce(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_in.iter_data,
                d_out.iter_data,
                <uint64_t>num_items,
                start_offsets.iter_data,
                end_offsets.iter_data,
                op.op_data,
                h_init.value_data,
                c_stream
            )
        if status != 0:
            raise RuntimeError(
                f"Failed executing segmented_reduce, error code: {status}"
            )
        return storage_sz

    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]
# -----------------
#   DeviceMergeSort
# -----------------


cdef extern from "cccl/c/merge_sort.h":
    cdef struct cccl_device_merge_sort_build_result_t 'cccl_device_merge_sort_build_result_t':
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_merge_sort_build(
        cccl_device_merge_sort_build_result_t *bld_ptr,
        cccl_iterator_t d_in_keys,
        cccl_iterator_t d_in_items,
        cccl_iterator_t d_out_keys,
        cccl_iterator_t d_out_items,
        cccl_op_t,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_merge_sort(
        cccl_device_merge_sort_build_result_t,
        void *,
        size_t *,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        cccl_iterator_t,
        uint64_t,
        cccl_op_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_merge_sort_cleanup(
        cccl_device_merge_sort_build_result_t* bld_ptr
    ) nogil


cdef class DeviceMergeSortBuildResult:
    cdef cccl_device_merge_sort_build_result_t build_data

    def __cinit__(
        DeviceMergeSortBuildResult self,
        Iterator d_in_keys,
        Iterator d_in_items,
        Iterator d_out_keys,
        Iterator d_out_items,
        Op op,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        memset(&self.build_data, 0, sizeof(cccl_device_merge_sort_build_result_t))
        with nogil:
            status = cccl_device_merge_sort_build(
                &self.build_data,
                d_in_keys.iter_data,
                d_in_items.iter_data,
                d_out_keys.iter_data,
                d_out_items.iter_data,
                op.op_data,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(
                f"Failed building merge_sort, error code: {status}"
            )

    def __dealloc__(DeviceMergeSortBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_merge_sort_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during merge_sort result cleanup")

    cpdef int compute(
        DeviceMergeSortBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_in_keys,
        Iterator d_in_items,
        Iterator d_out_keys,
        Iterator d_out_items,
        size_t num_items,
        Op op,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><uintptr_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL
        with nogil:
            status = cccl_device_merge_sort(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_in_keys.iter_data,
                d_in_items.iter_data,
                d_out_keys.iter_data,
                d_out_items.iter_data,
                <uint64_t>num_items,
                op.op_data,
                c_stream
            )
        if status != 0:
            raise RuntimeError(
                f"Failed executing merge_sort, error code: {status}"
            )
        return storage_sz


    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]


# -------------------
#   DeviceUniqueByKey
# -------------------

cdef extern from "cccl/c/unique_by_key.h":
    cdef struct cccl_device_unique_by_key_build_result_t 'cccl_device_unique_by_key_build_result_t':
        const char* cubin
        size_t cubin_size


    cdef CUresult cccl_device_unique_by_key_build(
        cccl_device_unique_by_key_build_result_t *build_ptr,
        cccl_iterator_t d_keys_in,
        cccl_iterator_t d_values_in,
        cccl_iterator_t d_keys_out,
        cccl_iterator_t d_values_out,
        cccl_iterator_t d_num_selected_out,
        cccl_op_t comparison_op,
        int, int, const char *, const char *, const char *, const char *
    ) nogil

    cdef CUresult cccl_device_unique_by_key(
        cccl_device_unique_by_key_build_result_t build,
        void *d_storage_ptr,
        size_t *d_storage_nbytes,
        cccl_iterator_t d_keys_in,
        cccl_iterator_t d_values_in,
        cccl_iterator_t d_keys_out,
        cccl_iterator_t d_values_out,
        cccl_iterator_t d_num_selected_out,
        cccl_op_t comparison_op,
        size_t num_items,
        CUstream stream
    ) nogil

    cdef CUresult cccl_device_unique_by_key_cleanup(
        cccl_device_unique_by_key_build_result_t *build_ptr,
    ) nogil


cdef class DeviceUniqueByKeyBuildResult:
    cdef cccl_device_unique_by_key_build_result_t build_data

    def __cinit__(
        DeviceUniqueByKeyBuildResult self,
        Iterator d_keys_in,
        Iterator d_values_in,
        Iterator d_keys_out,
        Iterator d_values_out,
        Iterator d_num_selected_out,
        Op comparison_op,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        memset(&self.build_data, 0, sizeof(cccl_device_unique_by_key_build_result_t))
        with nogil:
            status = cccl_device_unique_by_key_build(
                &self.build_data,
                d_keys_in.iter_data,
                d_values_in.iter_data,
                d_keys_out.iter_data,
                d_values_out.iter_data,
                d_num_selected_out.iter_data,
                comparison_op.op_data,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(
                f"Failed building unique_by_key, error code: {status}"
            )

    def __dealloc__(DeviceUniqueByKeyBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_unique_by_key_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during unique_by_key result cleanup")

    cpdef int compute(
        DeviceUniqueByKeyBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_keys_in,
        Iterator d_values_in,
        Iterator d_keys_out,
        Iterator d_values_out,
        Iterator d_num_selected_out,
        Op comparison_op,
        size_t num_items,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><uintptr_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_unique_by_key(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_keys_in.iter_data,
                d_values_in.iter_data,
                d_keys_out.iter_data,
                d_values_out.iter_data,
                d_num_selected_out.iter_data,
                comparison_op.op_data,
                <uint64_t>num_items,
                c_stream
            )

        if status != 0:
            raise RuntimeError(
                f"Failed executing unique_by_key, error code: {status}"
            )
        return storage_sz

    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]

# -----------------
# DeviceRadixSort
# -----------------

cdef extern from "cccl/c/radix_sort.h":
    cdef struct cccl_device_radix_sort_build_result_t 'cccl_device_radix_sort_build_result_t':
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_radix_sort_build(
        cccl_device_radix_sort_build_result_t *build_ptr,
        cccl_sort_order_t sort_order,
        cccl_iterator_t d_keys_in,
        cccl_iterator_t d_values_in,
        cccl_op_t decomposer,
        const char* decomposer_return_type,
        int, int, const char *, const char *, const char *, const char *
    ) nogil

    cdef CUresult cccl_device_radix_sort(
        cccl_device_radix_sort_build_result_t build,
        void *d_storage_ptr,
        size_t *d_storage_nbytes,
        cccl_iterator_t d_keys_in,
        cccl_iterator_t d_keys_out,
        cccl_iterator_t d_values_in,
        cccl_iterator_t d_values_out,
        cccl_op_t decomposer,
        size_t num_items,
        int begin_bit,
        int end_bit,
        bint is_overwrite_okay,
        int* selector,
        CUstream stream
    ) nogil

    cdef CUresult cccl_device_radix_sort_cleanup(
        cccl_device_radix_sort_build_result_t *build_ptr,
    ) nogil


cdef class DeviceRadixSortBuildResult:
    cdef cccl_device_radix_sort_build_result_t build_data

    def __dealloc__(DeviceRadixSortBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_radix_sort_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during radix_sort result cleanup")

    def __cinit__(
        DeviceRadixSortBuildResult self,
        cccl_sort_order_t order,
        Iterator d_keys_in,
        Iterator d_values_in,
        Op decomposer_op,
        const char* decomposer_return_type,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        memset(&self.build_data, 0, sizeof(cccl_device_radix_sort_build_result_t))
        with nogil:
            status = cccl_device_radix_sort_build(
                &self.build_data,
                order,
                d_keys_in.iter_data,
                d_values_in.iter_data,
                decomposer_op.op_data,
                decomposer_return_type,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(
                f"Failed building radix_sort, error code: {status}"
            )

    cpdef tuple compute(
        DeviceRadixSortBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_keys_in,
        Iterator d_keys_out,
        Iterator d_values_in,
        Iterator d_values_out,
        Op decomposer_op,
        size_t num_items,
        int begin_bit,
        int end_bit,
        bint is_overwrite_okay,
        selector,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef int selector_int = <int>selector
        cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_radix_sort(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_keys_in.iter_data,
                d_keys_out.iter_data,
                d_values_in.iter_data,
                d_values_out.iter_data,
                decomposer_op.op_data,
                <uint64_t>num_items,
                begin_bit,
                end_bit,
                is_overwrite_okay,
                &selector_int,
                c_stream
            )

        if status != 0:
            raise RuntimeError(
                f"Failed executing ascending radix_sort, error code: {status}"
            )
        return <object>storage_sz, <object>selector_int


    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]

# --------------------------------------------
#   DeviceUnaryTransform/DeviceBinaryTransform
# --------------------------------------------
cdef extern from "cccl/c/transform.h":
    cdef struct cccl_device_transform_build_result_t:
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_unary_transform_build(
        cccl_device_transform_build_result_t *build_ptr,
        cccl_iterator_t d_in,
        cccl_iterator_t d_out,
        cccl_op_t op,
        int, int, const char *, const char *, const char *, const char *
    ) nogil

    cdef CUresult cccl_device_unary_transform(
      cccl_device_transform_build_result_t build,
      cccl_iterator_t d_in,
      cccl_iterator_t d_out,
      uint64_t num_items,
      cccl_op_t op,
      CUstream stream) nogil

    cdef CUresult cccl_device_binary_transform_build(
      cccl_device_transform_build_result_t* build_ptr,
      cccl_iterator_t d_in1,
      cccl_iterator_t d_in2,
      cccl_iterator_t d_out,
      cccl_op_t op,
      int, int, const char *, const char *, const char *, const char *
    ) nogil

    cdef CUresult cccl_device_binary_transform(
      cccl_device_transform_build_result_t build,
      cccl_iterator_t d_in1,
      cccl_iterator_t d_in2,
      cccl_iterator_t d_out,
      uint64_t num_items,
      cccl_op_t op,
      CUstream stream) nogil

    cdef CUresult cccl_device_transform_cleanup(
        cccl_device_transform_build_result_t *build_ptr,
    ) nogil


cdef class DeviceUnaryTransform:
    cdef cccl_device_transform_build_result_t build_data

    def __cinit__(
        self,
        Iterator d_in,
        Iterator d_out,
        Op op,
        CommonData common_data
    ):
        memset(&self.build_data, 0, sizeof(cccl_device_transform_build_result_t))

        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        with nogil:
            status = cccl_device_unary_transform_build(
                &self.build_data,
                d_in.iter_data,
                d_out.iter_data,
                op.op_data,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError("Failed to build unary transform")

    def __dealloc__(DeviceUnaryTransform self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_transform_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during unary transform result cleanup")

    cpdef void compute(
        DeviceUnaryTransform self,
        Iterator d_in,
        Iterator d_out,
        size_t num_items,
        Op op,
        stream
    ):
        cdef CUresult status = -1
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL
        with nogil:
            status = cccl_device_unary_transform(
                self.build_data,
                d_in.iter_data,
                d_out.iter_data,
                <uint64_t>num_items,
                op.op_data,
                c_stream
            )
        if (status != 0):
            raise RuntimeError("Failed to compute unary transform")


    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]


cdef class DeviceBinaryTransform:
    cdef cccl_device_transform_build_result_t build_data

    def __cinit__(
        self,
        Iterator d_in1,
        Iterator d_in2,
        Iterator d_out,
        Op op,
        CommonData common_data
    ):
        memset(&self.build_data, 0, sizeof(cccl_device_transform_build_result_t))

        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        with nogil:
            status = cccl_device_binary_transform_build(
                &self.build_data,
                d_in1.iter_data,
                d_in2.iter_data,
                d_out.iter_data,
                op.op_data,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError("Failed to build binary transform")

    def __dealloc__(DeviceBinaryTransform self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_transform_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during binary transform result cleanup")

    cpdef void compute(
        DeviceBinaryTransform self,
        Iterator d_in1,
        Iterator d_in2,
        Iterator d_out,
        size_t num_items,
        Op op,
        stream
    ):
        cdef CUresult status = -1
        cdef CUstream c_stream = <CUstream><uintptr_t>(stream) if stream else NULL
        with nogil:
            status = cccl_device_binary_transform(
                self.build_data,
                d_in1.iter_data,
                d_in2.iter_data,
                d_out.iter_data,
                <uint64_t>num_items,
                op.op_data,
                c_stream
            )
        if (status != 0):
            raise RuntimeError("Failed to compute binary transform")

    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]


# -----------------
#   DeviceHistogram
# -----------------
cdef extern from "cccl/c/histogram.h":
    cdef struct cccl_device_histogram_build_result_t 'cccl_device_histogram_build_result_t':
        const char* cubin
        size_t cubin_size

    cdef CUresult cccl_device_histogram_build(
        cccl_device_histogram_build_result_t *build_ptr,
        int num_channels,
        int num_active_channels,
        cccl_iterator_t d_samples,
        int num_output_levels_val,
        cccl_iterator_t d_output_histograms,
        cccl_value_t h_levels,
        int64_t num_rows,
        int64_t row_stride_samples,
        bint is_evenly_segmented,
        int, int, const char *, const char *, const char *, const char *
    ) nogil

    cdef CUresult cccl_device_histogram_even(
        cccl_device_histogram_build_result_t build,
        void *d_storage_ptr,
        size_t *d_storage_nbytes,
        cccl_iterator_t d_samples,
        cccl_iterator_t d_output_histograms,
        cccl_value_t num_output_levels,
        cccl_value_t lower_level,
        cccl_value_t upper_level,
        int64_t num_row_pixels,
        int64_t num_rows,
        int64_t row_stride_samples,
        CUstream stream
    ) nogil

    cdef CUresult cccl_device_histogram_cleanup(
        cccl_device_histogram_build_result_t *build_ptr,
    ) nogil


cdef class DeviceHistogramBuildResult:
    cdef cccl_device_histogram_build_result_t build_data

    def __dealloc__(DeviceHistogramBuildResult self):
        cdef CUresult status = -1
        with nogil:
            status = cccl_device_histogram_cleanup(&self.build_data)
        if (status != 0):
            print(f"Return code {status} encountered during histogram result cleanup")


    def __cinit__(
        DeviceHistogramBuildResult self,
        int num_channels,
        int num_active_channels,
        Iterator d_samples,
        int num_levels,
        Iterator d_histogram,
        Value h_levels,
        int num_rows,
        int row_stride_samples,
        bint is_evenly_segmented,
        CommonData common_data
    ):
        cdef CUresult status = -1
        cdef int cc_major = common_data.get_cc_major()
        cdef int cc_minor = common_data.get_cc_minor()
        cdef const char *cub_path = common_data.cub_path_get_c_str()
        cdef const char *thrust_path = common_data.thrust_path_get_c_str()
        cdef const char *libcudacxx_path = common_data.libcudacxx_path_get_c_str()
        cdef const char *ctk_path = common_data.ctk_path_get_c_str()

        memset(&self.build_data, 0, sizeof(cccl_device_histogram_build_result_t))
        with nogil:
            status = cccl_device_histogram_build(
                &self.build_data,
                num_channels,
                num_active_channels,
                d_samples.iter_data,
                num_levels,
                d_histogram.iter_data,
                h_levels.value_data,
                num_rows,
                row_stride_samples,
                is_evenly_segmented,
                cc_major,
                cc_minor,
                cub_path,
                thrust_path,
                libcudacxx_path,
                ctk_path,
            )
        if status != 0:
            raise RuntimeError(
                f"Failed building histogram, error code: {status}"
            )

    cpdef int compute_even(
        DeviceHistogramBuildResult self,
        temp_storage_ptr,
        temp_storage_bytes,
        Iterator d_samples,
        Iterator d_histogram,
        Value h_num_output_levels,
        Value h_lower_level,
        Value h_upper_level,
        int num_row_pixels,
        int num_rows,
        int row_stride_samples,
        stream
    ):
        cdef CUresult status = -1
        cdef void *storage_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
        cdef size_t storage_sz = <size_t>temp_storage_bytes
        cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL

        with nogil:
            status = cccl_device_histogram_even(
                self.build_data,
                storage_ptr,
                &storage_sz,
                d_samples.iter_data,
                d_histogram.iter_data,
                h_num_output_levels.value_data,
                h_lower_level.value_data,
                h_upper_level.value_data,
                num_row_pixels,
                num_rows,
                row_stride_samples,
                c_stream
            )
        if status != 0:
            raise RuntimeError(
                f"Failed executing histogram, error code: {status}"
            )
        return storage_sz


    def _get_cubin(self):
        return self.build_data.cubin[:self.build_data.cubin_size]
