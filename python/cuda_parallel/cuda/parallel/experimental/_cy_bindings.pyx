# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.bytes cimport PyBytes_FromStringAndSize

from cpython.buffer cimport (
    Py_buffer, PyBUF_SIMPLE, PyBUF_ANY_CONTIGUOUS,
    PyBuffer_Release, PyObject_CheckBuffer, PyObject_GetBuffer
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
    ctypedef enum cy_cccl_type_enum 'cccl_type_enum':
        cy_CCCL_INT8   'CCCL_INT8'
        cy_CCCL_INT16  'CCCL_INT16'
        cy_CCCL_INT32  'CCCL_INT32'
        cy_CCCL_INT64  'CCCL_INT64'
        cy_CCCL_UINT8  'CCCL_UINT8'
        cy_CCCL_UINT16 'CCCL_UINT16'
        cy_CCCL_UINT32 'CCCL_UINT32'
        cy_CCCL_UINT64 'CCCL_UINT64'
        cy_CCCL_FLOAT32 'CCCL_FLOAT32'
        cy_CCCL_FLOAT64 'CCCL_FLOAT64'
        cy_CCCL_STORAGE 'CCCL_STORAGE'

    ctypedef enum cy_cccl_op_kind_t 'cccl_op_kind_t':
       cy_CCCL_STATELESS  'CCCL_STATELESS'
       cy_CCCL_STATEFUL   'CCCL_STATEFUL'

    ctypedef enum cy_cccl_iterator_kind_t 'cccl_iterator_kind_t':
       cy_CCCL_POINTER  'CCCL_POINTER'
       cy_CCCL_ITERATOR 'CCCL_ITERATOR'

    cdef struct cy_cccl_type_info 'cccl_type_info':
        size_t size
        size_t alignment
        cy_cccl_type_enum type

    cdef struct cy_cccl_op_t 'cccl_op_t':
        cy_cccl_op_kind_t type
        const char* name
        const char* ltoir
        size_t ltoir_size
        size_t size
        size_t alignment
        void *state

    cdef struct cy_cccl_value_t 'cccl_value_t':
        cy_cccl_type_info type
        void *state

    cdef struct cy_cccl_iterator_t 'cccl_iterator_t':
        size_t size
        size_t alignment
        cy_cccl_iterator_kind_t type
        cy_cccl_op_t advance
        cy_cccl_op_t dereference
        cy_cccl_type_info value_type
        void *state


cdef object arg_type_check(
    str arg_name,
    object expected_type,
    object arg
):
    if not isinstance(arg, expected_type):
        raise TypeError(
            f"Expected {arg_name} to have type '{expected_type}', "
            f"got '{type(arg)}'"
        )


cdef class IntEnum:
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
        return self.parent_class

    @property
    def name(self):
        return self.attr_name

    @property
    def value(self):
        return self.attr_value

    @property
    def _cmp_key(self):
        return (type(self), self.parent_class, <object>self.attr_value)

    def __hash__(self):
        return hash(self._cmp_key)

    def __eq__(self, other):
        if type(other) == type(self):
            return self._cmp_key == other._cmp_key
        else:
            return False


cdef class IntEnumBase:
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


cdef class IntEnum_CCCLType(IntEnumBase):
    "Enumeration of CCCL types"

    def __cinit__(self):
        self.enum_name = "TypeEnum"

    @property
    def INT8(self):
        cdef str prop_name = "INT8"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_INT8
        )

    @property
    def INT16(self):
        cdef str prop_name = "INT16"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_INT16
        )

    @property
    def INT32(self):
        cdef str prop_name = "INT32"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_INT32
        )

    @property
    def INT64(self):
        cdef str prop_name = "INT64"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_INT64
        )

    @property
    def UINT8(self):
        cdef str prop_name = "UINT8"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_UINT8
        )

    @property
    def UINT16(self):
        cdef str prop_name = "UINT16"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_UINT16
        )

    @property
    def UINT32(self):
        cdef str prop_name = "UINT32"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_UINT32
        )

    @property
    def UINT64(self):
        cdef str prop_name = "UINT64"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_UINT64
        )


    @property
    def FLOAT32(self):
        cdef str prop_name = "FLOAT32"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_FLOAT32
        )

    @property
    def FLOAT64(self):
        cdef str prop_name = "FLOAT64"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_FLOAT64
        )


    @property
    def STORAGE(self):
        cdef str prop_name = "STORAGE"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_type_enum.cy_CCCL_STORAGE
        )


cdef class IntEnum_OpKind(IntEnumBase):
    "Enumeration of operator kinds"

    def __cinit__(self):
        self.enum_name = "OpKindEnum"

    @property
    def STATELESS(self):
        cdef str prop_name = "STATELESS"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_op_kind_t.cy_CCCL_STATELESS
        )

    @property
    def STATEFUL(self):
        cdef str prop_name = "STATEFUL"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_op_kind_t.cy_CCCL_STATEFUL
        )


cdef class IntEnum_IteratorKind(IntEnumBase):
    "Enumeration of iterator kinds"

    def __cinit__(self):
        self.enum_name = "IteratorKindEnum"

    @property
    def POINTER(self):
        cdef str prop_name = "POINTER"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_iterator_kind_t.cy_CCCL_POINTER
        )

    @property
    def ITERATOR(self):
        cdef str prop_name = "ITERATOR"
        return IntEnum(
            type(self),
            self.enum_name,
            prop_name,
            cy_cccl_iterator_kind_t.cy_CCCL_ITERATOR
        )


TypeEnum = IntEnum_CCCLType()
OpKind = IntEnum_OpKind()
IteratorKind = IntEnum_IteratorKind()


def pointer_as_bytes(size_t ptr):
    """
    Return bytes object whose content stores a pointer value

    Returns bytes(ctypes.c_void_p(ptr)) but faster.

    .. Note:

       # In [6]: %timeit cb.pointer_as_bytes(ptr)
       # 55.1 ns ± 0.904 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
       #
       # In [7]: %timeit bytes(ctypes.c_void_p(ptr))
       # 103 ns ± 0.0436 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
       #

    """
    return PyBytes_FromStringAndSize(<char *>&ptr, sizeof(size_t))


cpdef bint is_TypeEnum(IntEnum attr):
    "Return True is attribute represents a type enumerator"
    return attr.parent_class is IntEnum_CCCLType


cpdef bint is_OpKind(IntEnum attr):
    "Return True is attribute represents an enumerator of operator kinds"
    return attr.parent_class is IntEnum_OpKind


cpdef bint is_IteratorKind(IntEnum attr):
    "Return True is attribute represents an enumerator of iterator kinds"
    return attr.parent_class is IntEnum_IteratorKind


cdef object _validate_alignment(int alignment):
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
    # need Python owner of memory used for operator name
    cdef bytes op_encoded_name
    cdef bytes ltoir_bytes
    cdef bytes state_bytes
    cdef cy_cccl_op_t op_data


    cdef void _set_members(self, cy_cccl_op_kind_t type, str name, bytes lto_ir, bytes state, int state_alignment):
        memset(&self.op_data, 0, sizeof(cy_cccl_op_t))
        # Reference Python objects in the class to ensure lifetime
        self.op_encoded_name = name.encode("utf-8")
        self.ltoir_bytes = lto_ir
        self.state_bytes = state
        # set fields of op_data struct
        self.op_data.type = type
        self.op_data.name = <const char *>self.op_encoded_name
        self.op_data.ltoir = <const char *>lto_ir
        self.op_data.ltoir_size = len(lto_ir)
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
        arg_type_check(arg_name="operator_type", expected_type=IntEnum, arg=operator_type)
        if not is_OpKind(operator_type):
            raise TypeError(
                f"The operator_type argument should be an enumerator of operator kinds"
            )
        _validate_alignment(state_alignment)
        self._set_members(
            <cy_cccl_op_kind_t> operator_type.value,
            <str> name,
            <bytes> ltoir,
            <bytes> state,
            <int> state_alignment
        )


    cdef void set_state(self, bytes state):
        self.state_bytes = state
        self.op_data.state = <void *><const char *>state

    cdef cy_cccl_op_t* get_ptr(self):
        return &self.op_data

    cdef cy_cccl_op_t get(self):
        return self.op_data

    property state:
        def __get__(self):
            return self.state_bytes

        def __set__(self, bytes new_value):
            self.set_state(<bytes>new_value)

    @property
    def name(self):
        return self.op_encoded_name.decode("utf-8")

    @property
    def ltoir(self):
        return self.ltoir_bytes

    @property
    def state_alignment(self):
        return self.op_data.alignment

    @property
    def state_typenum(self):
        return self.op_data.type

    def as_bytes(self):
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.op_data))
        memcpy(&mem_view[0], &self.op_data, sizeof(self.op_data))
        return bytes(mem_view)


cdef class TypeInfo:
    cdef cy_cccl_type_info type_info

    def __cinit__(self, int size, int alignment, IntEnum type_enum):
        if size < 1:
            raise ValueError("Size argument must be positive")
        _validate_alignment(alignment)
        if not is_TypeEnum(type_enum):
            raise TypeError(
                f"The type argument should be enum of CCCL types"
            )
        self.type_info.size = size
        self.type_info.alignment = alignment
        self.type_info.type = <cy_cccl_type_enum> type_enum.value

    @property
    def size(self):
        return self.type_info.size

    @property
    def alignment(self):
        return self.type_info.alignment

    @property
    def typenum(self):
        return self.type_info.type

    cdef cy_cccl_type_info *get_ptr(self):
        return &self.type_info

    cdef cy_cccl_type_info get(self):
        return self.type_info

    def as_bytes(self):
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.type_info))
        memcpy(&mem_view[0], &self.type_info, sizeof(self.type_info))
        return bytes(mem_view)


cdef class Value:
    cdef uint8_t[::1] state_obj
    cdef TypeInfo value_type
    cdef cy_cccl_value_t value_data;

    def __cinit__(self, TypeInfo value_type, uint8_t[::1] state):
        self.state_obj = state
        self.value_type = value_type
        self.value_data.type = value_type.get()
        self.value_data.state = <void *>&state[0]

    cdef cy_cccl_value_t * get_ptr(self):
        return &self.value_data

    cdef cy_cccl_value_t get(self):
        return self.value_data

    @property
    def type(self):
        return self.value_type

    property state:
        def __get__(self):
            return self.state_obj

        def __set__(self, uint8_t[::1] new_value):
            if (len(self.state_obj) == len(new_value)):
                self.state_obj = new_value
                self.value_data.state = <void *>&self.state_obj[0]
            else:
                raise ValueError("Size mismatch")

    def as_bytes(self):
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


cdef class PointerProxy:
    cdef void * ptr
    cdef object owner

    def __cinit__(self, ptr, owner):
        if isinstance(ptr, ctypes.c_void_p):
            self.ptr = int_as_ptr(ptr.value)
        elif isinstance(ptr, int):
            self.ptr = int_as_ptr(ptr)
        else:
            raise TypeError
        self.owner = owner

    @property
    def pointer(self):
        return <size_t>self.ptr

    @property
    def reference(self):
        return self.owner


cdef class IteratorStateView:
    cdef void * ptr
    cdef size_t size
    cdef object owner

    def __cinit__(self, ptr, size_t size, owner):
        if isinstance(ptr, int):
            self.ptr = int_as_ptr(ptr)
            self.size = size
        elif isinstance(ptr, ctypes._Pointer):
            self.ptr = ctypes_typed_pointer_payload_ptr(ptr)
            self.size = ctypes.sizeof(ptr.contents)
        elif isinstance(ptr, ctypes.c_void_p):
            self.ptr = int_as_ptr(ptr.value)
            self.size = size
        elif PyObject_CheckBuffer(ptr):
            self.ptr = get_buffer_pointer(ptr, &self.size)
        else:
            raise TypeError(
                "First argument must be int, type ctypes pointer, or ctypes.c_void_p, "
                f"got type {type(ptr)}"
            )
        self.owner = owner

    @property
    def pointer(self):
        return <size_t>self.ptr

    @property
    def size(self):
        return self.size

    @property
    def reference(self):
        return self.owner


cdef class StateBase:
    cdef void *ptr
    cdef object ref

    def __cinit__(self):
        self.ptr = NULL
        self.ref = None

    cdef void set_state(self, void *ptr, object ref):
        self.ptr = ptr
        self.ref = ref

    cdef void * get(self):
        return self.ptr

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
        cdef PointerProxy handle

        super().__init__()
        if isinstance(arg, PointerProxy):
            handle = <PointerProxy>arg
            ptr = handle.ptr
            ref = handle.owner
        elif isinstance(arg, int):
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


cdef class IteratorState(StateBase):
    "Represents blob referenced by pointer"
    cdef size_t size

    def __cinit__(self, arg):
        cdef size_t buffer_size = 0
        cdef void *ptr = NULL
        cdef object ref = None

        super().__init__()
        if isinstance(arg, ctypes._Pointer):
            ptr = ctypes_typed_pointer_payload_ptr(arg)
            ref = arg.contents
            self.size = ctypes.sizeof(ref)
        elif PyObject_CheckBuffer(arg):
            ptr = get_buffer_pointer(arg, &buffer_size)
            ref = arg
            self.size = buffer_size
        else:
            raise TypeError(
                "Expected a ctypes pointer with content, or object of type bytes or bytearray, "
                f"got type {type(arg)}"
            )
        self.set_state(ptr, ref)

    cdef size_t get_size(self):
        return self.size

    @property
    def size(self):
        return self.size

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t cast_size = <Py_ssize_t>self.size
        buffer.buf = <char *>self.ptr
        buffer.obj = self
        buffer.len = cast_size
        buffer.readonly = 0
        buffer.itemsize = 1
        buffer.format = "B"  # unsigned char
        buffer.ndim = 1
        buffer.shape = <Py_ssize_t *>&self.size
        buffer.strides = &buffer.itemsize
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cdef class Iterator:
    cdef Op advance
    cdef Op dereference
    cdef object state_obj
    cdef cy_cccl_iterator_t iter_data

    def __cinit__(self,
        int alignment,
        IntEnum iterator_type,
        Op advance_fn,
        Op dereference_fn,
        TypeInfo value_type,
        state = None
    ):
        _validate_alignment(alignment)
        if not is_IteratorKind(iterator_type):
            raise TypeError("iterator_type must describe iterator kind")
        it_kind = iterator_type.value
        if it_kind == cy_cccl_iterator_kind_t.cy_CCCL_POINTER:
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
                self.iter_data.state = (<Pointer>state).get()
            else:
                raise TypeError(
                    "Expect for Iterator of kind POINTER, state must have type Pointer or int, "
                    f"got {type(state)}"
                )
        elif it_kind == cy_cccl_iterator_kind_t.cy_CCCL_ITERATOR:
            if state is None:
                self.state_obj = None
                self.iter_data.size = 0
                self.iter_data.state = NULL
            elif isinstance(state, IteratorState):
                self.state_obj = state.reference
                self.iter_data.size = (<IteratorState>state).get_size()
                self.iter_data.state = (<IteratorState>state).get()
            elif isinstance(state, IteratorStateView):
                self.state_obj = state.reference
                self.iter_data.size = <size_t>state.size
                self.iter_data.state = <void *>((<IteratorStateView>state).ptr)
            else:
                raise TypeError(
                    "For Iterator of kind ITERATOR, state must have type IteratorState or IteratorStateView, "
                    f"got type {type(state)}"
                )
        else:  # pragma: no cover
            raise ValueError("Unrecognized iterator kind")
        self.advance = advance_fn
        self.dereference = dereference_fn
        self.iter_data.alignment = alignment
        self.iter_data.type = <cy_cccl_iterator_kind_t> it_kind
        self.iter_data.advance = self.advance.get()
        self.iter_data.dereference = self.dereference.get()
        self.iter_data.value_type = value_type.get()

    cdef cy_cccl_iterator_t *get_ptr(self):
        return &self.iter_data

    cdef cy_cccl_iterator_t get(self):
        return self.iter_data

    @property
    def advance_op(self):
        return self.advance

    @property
    def dereference_or_assign_op(self):
        return self.dereference

    property state:
        def __get__(self):
            if self.iter_data.type == cy_cccl_iterator_kind_t.cy_CCCL_POINTER:
                return <size_t>self.iter_data.state
            else:
                return self.state_obj

        def __set__(self, new_value):
            cdef ssize_t state_sz = 0
            cdef size_t ptr = 0
            if self.iter_data.type == cy_cccl_iterator_kind_t.cy_CCCL_POINTER:
                if isinstance(new_value, Pointer):
                    self.state_obj = new_value.reference
                    self.iter_data.size = state_sz
                    self.iter_data.state = (<Pointer>new_value).get()
                elif isinstance(new_value, int):
                    self.state_obj = None
                    self.iter_data.size = state_sz
                    self.iter_data.state = int_as_ptr(new_value)
                else:
                    raise TypeError(
                        "For iterator with type POINTER, state value must have type int or type Pointer, "
                        f"got type {type(new_value)}"
                    )
            elif self.iter_data.type == cy_cccl_iterator_kind_t.cy_CCCL_ITERATOR:
                if isinstance(new_value, IteratorState):
                    self.state_obj = new_value.reference
                    self.iter_data.size = (<IteratorState>new_value).get_size()
                    self.iter_data.state = (<IteratorState>new_value).get()
                elif isinstance(new_value, IteratorStateView):
                    self.state_obj = new_value.reference
                    self.iter_data.size = (<IteratorStateView>new_value).size
                    self.iter_data.state = (<IteratorStateView>new_value).ptr
                elif isinstance(new_value, Pointer):
                    self.state_obj = new_value.reference
                    if self.iter_data.size == 0:
                        raise ValueError("Assigning incomplete state value to iterator without state size information")
                    self.iter_data.state = (<Pointer>new_value).get()
                elif PyObject_CheckBuffer(new_value):
                    self.iter_data.state = get_buffer_pointer(new_value, &self.iter_data.size)
                    self.state_obj = new_value
                else:
                    raise TypeError(
                        "For iterator with type ITERATOR, state value must have type IteratorState or type bytes, "
                        f"got type {type(new_value)}"
                    )
            else:
                raise TypeError("The new value should be an integer for iterators of POINTER kind, and bytes for ITERATOR kind")

    @property
    def type(self):
        cdef cy_cccl_iterator_kind_t it_kind = self.iter_data.type
        if it_kind == cy_cccl_iterator_kind_t.cy_CCCL_POINTER:
            return IteratorKind.POINTER
        else:
            return IteratorKind.ITERATOR

    def as_bytes(self):
        cdef uint8_t[:] mem_view = bytearray(sizeof(self.iter_data))
        memcpy(&mem_view[0], &self.iter_data, sizeof(self.iter_data))
        return bytes(mem_view)


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

    cdef int get_cc_major(self):
        return self.cc_major

    cdef int get_cc_minor(self):
        return self.cc_minor

    cdef const char * cub_path_get_c_str(self):
        return <const char *>self.encoded_cub_path

    cdef const char * thrust_path_get_c_str(self):
        return <const char *>self.encoded_thrust_path

    cdef const char * libcudacxx_path_get_c_str(self):
        return <const char *>self.encoded_libcudacxx_path

    cdef const char * ctk_path_get_c_str(self):
        return <const char *>self.encoded_ctk_path

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
    cdef struct cy_cccl_device_reduce_build_result_t 'cccl_device_reduce_build_result_t':
        int cc
        void *cubin
        size_t cubin_size
        CUlibrary library
        uint64_t accumulator_size
        CUkernel single_tile_kernel
        CUkernel single_tile_second_kernel
        CUkernel reduction_kernel

    cdef CUresult cccl_device_reduce_build(
        cy_cccl_device_reduce_build_result_t*,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_reduce(
        cy_cccl_device_reduce_build_result_t,
        void *,
        size_t *,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        uint64_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_reduce_cleanup(
        cy_cccl_device_reduce_build_result_t*
    ) nogil


cdef class DeviceReduceBuildResult:
    cdef cy_cccl_device_reduce_build_result_t _build_data

    def __cinit__(self):
       memset(&self._build_data, 0, sizeof(cy_cccl_device_reduce_build_result_t))

    cdef cy_cccl_device_reduce_build_result_t* get_ptr(self):
       return &self._build_data

    cdef cy_cccl_device_reduce_build_result_t get(self):
       return self._build_data


cpdef CUresult device_reduce_build(
    DeviceReduceBuildResult build,
    Iterator d_in,
    Iterator d_out,
    Op op,
    Value h_init,
    CommonData common_data
):
    cdef CUresult status
    status = cccl_device_reduce_build(
        build.get_ptr(),
        d_in.get(),
        d_out.get(),
        op.get(),
        h_init.get(),
        common_data.get_cc_major(),
        common_data.get_cc_minor(),
        common_data.cub_path_get_c_str(),
        common_data.thrust_path_get_c_str(),
        common_data.libcudacxx_path_get_c_str(),
        common_data.ctk_path_get_c_str(),
    )
    return status


cpdef device_reduce(
    DeviceReduceBuildResult build,
    temp_storage_ptr,
    temp_storage_bytes,
    Iterator d_in,
    Iterator d_out,
    size_t num_items,
    Op op,
    Value h_init,
    stream
):
    cdef CUresult status
    cdef void *storage_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
    cdef size_t storage_sz = <size_t>temp_storage_bytes
    cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL
    status = cccl_device_reduce(
        build.get(),
        storage_ptr,
        &storage_sz,
        d_in.get(),
        d_out.get(),
        <uint64_t>num_items,
        op.get(),
        h_init.get(),
        c_stream
    )
    return status, <object>storage_sz


cpdef CUresult device_reduce_cleanup(DeviceReduceBuildResult build):
    cdef CUresult status

    status = cccl_device_reduce_cleanup(build.get_ptr())
    return status


# ------------
#   DeviceScan
# ------------


cdef extern from "cccl/c/scan.h":
    ctypedef bint _Bool

    cdef struct cy_cccl_device_scan_build_result_t 'cccl_device_scan_build_result_t':
        int cc
        void *cubin
        size_t cubin_size
        CUlibrary library
        cy_cccl_type_info accumulator_type
        CUkernel init_kernel
        CUkernel scan_kernel
        CUkernel reduction_kernel
        _Bool force_inclusive
        size_t description_bytes_per_tile
        size_t payload_bytes_per_tile

    cdef CUresult cccl_device_scan_build(
        cy_cccl_device_scan_build_result_t*,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        _Bool,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_exclusive_scan(
        cy_cccl_device_scan_build_result_t,
        void *,
        size_t *,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        uint64_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_inclusive_scan(
        cy_cccl_device_scan_build_result_t,
        void *,
        size_t *,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        uint64_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_scan_cleanup(
        cy_cccl_device_scan_build_result_t*
    ) nogil


cdef class DeviceScanBuildResult:
    cdef cy_cccl_device_scan_build_result_t _build_data

    def __cinit__(self):
       memset(&self._build_data, 0, sizeof(cy_cccl_device_scan_build_result_t))

    cdef cy_cccl_device_scan_build_result_t* get_ptr(self):
       return &self._build_data

    cdef cy_cccl_device_scan_build_result_t get(self):
       return self._build_data


cpdef CUresult device_scan_build(
    DeviceScanBuildResult build,
    Iterator d_in,
    Iterator d_out,
    Op op,
    Value h_init,
    bint force_inclusive,
    CommonData common_data
):
    cdef CUresult status
    status = cccl_device_scan_build(
        build.get_ptr(),
        d_in.get(),
        d_out.get(),
        op.get(),
        h_init.get(),
        force_inclusive,
        common_data.get_cc_major(),
        common_data.get_cc_minor(),
        common_data.cub_path_get_c_str(),
        common_data.thrust_path_get_c_str(),
        common_data.libcudacxx_path_get_c_str(),
        common_data.ctk_path_get_c_str(),
    )
    return status


cpdef device_inclusive_scan(
    DeviceScanBuildResult build,
    temp_storage_ptr,
    temp_storage_bytes,
    Iterator d_in,
    Iterator d_out,
    size_t num_items,
    Op op,
    Value h_init,
    stream
):
    cdef CUresult status
    cdef void *storage_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
    cdef size_t storage_sz = <size_t>temp_storage_bytes
    cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL
    status = cccl_device_inclusive_scan(
        build.get(),
        storage_ptr,
        &storage_sz,
        d_in.get(),
        d_out.get(),
        <uint64_t>num_items,
        op.get(),
        h_init.get(),
        c_stream
    )
    return status, <object>storage_sz


cpdef device_exclusive_scan(
    DeviceScanBuildResult build,
    temp_storage_ptr,
    temp_storage_bytes,
    Iterator d_in,
    Iterator d_out,
    size_t num_items,
    Op op,
    Value h_init,
    stream
):
    cdef CUresult status
    cdef void *storage_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
    cdef size_t storage_sz = <size_t>temp_storage_bytes
    cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL
    status = cccl_device_exclusive_scan(
        build.get(),
        storage_ptr,
        &storage_sz,
        d_in.get(),
        d_out.get(),
        <uint64_t>num_items,
        op.get(),
        h_init.get(),
        c_stream
    )
    return status, <object>storage_sz


cpdef CUresult device_scan_cleanup(DeviceScanBuildResult build):
    cdef CUresult status

    status = cccl_device_scan_cleanup(build.get_ptr())
    return status


# -----------------------
#   DeviceSegmentedReduce
# -----------------------


cdef extern from "cccl/c/segmented_reduce.h":
    cdef struct cy_cccl_device_segmented_reduce_build_result_t 'cccl_device_segmented_reduce_build_result_t':
        int cc
        void *cubin
        size_t cubin_size
        CUlibrary library
        uint64_t accumulator_size
        CUkernel segmented_reduce_kernel

    cdef CUresult cccl_device_segmented_reduce_build(
        cy_cccl_device_segmented_reduce_build_result_t*,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        int, int, const char*, const char*, const char*, const char*
    ) nogil

    cdef CUresult cccl_device_segmented_reduce(
        cy_cccl_device_segmented_reduce_build_result_t,
        void *,
        size_t *,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        uint64_t,
        cy_cccl_iterator_t,
        cy_cccl_iterator_t,
        cy_cccl_op_t,
        cy_cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_segmented_reduce_cleanup(
        cy_cccl_device_segmented_reduce_build_result_t* bld_ptr
    ) nogil


cdef class DeviceSegmentedReduceBuildResult:
    cdef cy_cccl_device_segmented_reduce_build_result_t _build_data

    def __cinit__(self):
       memset(&self._build_data, 0, sizeof(cy_cccl_device_segmented_reduce_build_result_t))

    cdef cy_cccl_device_segmented_reduce_build_result_t* get_ptr(self):
       return &self._build_data

    cdef cy_cccl_device_segmented_reduce_build_result_t get(self):
       return self._build_data

cpdef CUresult device_segmented_reduce_build(
    DeviceSegmentedReduceBuildResult build,
    Iterator d_in,
    Iterator d_out,
    Iterator start_offsets,
    Iterator end_offsets,
    Op op,
    Value h_init,
    CommonData common_data
):
    cdef CUresult status
    status = cccl_device_segmented_reduce_build(
        build.get_ptr(),
        d_in.get(),
        d_out.get(),
        start_offsets.get(),
        end_offsets.get(),
        op.get(),
        h_init.get(),
        common_data.get_cc_major(),
        common_data.get_cc_minor(),
        common_data.cub_path_get_c_str(),
        common_data.thrust_path_get_c_str(),
        common_data.libcudacxx_path_get_c_str(),
        common_data.ctk_path_get_c_str(),
    )
    return status


cpdef device_segmented_reduce(
    DeviceSegmentedReduceBuildResult build,
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
    cdef CUresult status
    cdef void *storage_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
    cdef size_t storage_sz = <size_t>temp_storage_bytes
    cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL
    status = cccl_device_segmented_reduce(
        build.get(),
        storage_ptr,
        &storage_sz,
        d_in.get(),
        d_out.get(),
        <uint64_t>num_items,
        start_offsets.get(),
        end_offsets.get(),
        op.get(),
        h_init.get(),
        c_stream
    )
    return status, <object>storage_sz

cpdef CUresult device_segmented_reduce_cleanup(
    DeviceSegmentedReduceBuildResult build
):
    cdef CUresult status

    status = cccl_device_segmented_reduce_cleanup(build.get_ptr())
    return status
