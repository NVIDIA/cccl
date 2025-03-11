# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

from libc.string cimport memset
from libc.stdint cimport uint8_t, uint32_t
from cpython.bytes cimport PyBytes_FromStringAndSize


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
        int size
        int alignment
        cy_cccl_type_enum type

    cdef struct cy_cccl_op_t 'cccl_op_t':
        cy_cccl_op_kind_t type
        const char* name
        const char* ltoir
        int ltoir_size
        int size
        int alignment
        void *state

    cdef struct cy_cccl_value_t 'cccl_value_t':
        cy_cccl_type_info type
        void *state

    cdef struct cy_cccl_iterator_t 'cccl_iterator_t':
        int size
        int alignment
        cy_cccl_iterator_kind_t type
        cy_cccl_op_t advance
        cy_cccl_op_t dereference
        cy_cccl_type_info value_type
        void *state

cdef extern from "cccl/c/reduce.h":
    cdef struct cy_cccl_device_reduce_build_result_t 'cccl_device_reduce_build_result_t':
        int cc
        void *cubin
        size_t cubin_size
        CUlibrary library
        unsigned long long accumulator_size
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
        unsigned long long,
        cy_cccl_op_t,
        cy_cccl_value_t,
        CUstream
    ) nogil

    cdef CUresult cccl_device_reduce_cleanup(
        cy_cccl_device_reduce_build_result_t*
    ) nogil


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


cdef class Value:
    cdef bytes state_bytes
    cdef TypeInfo value_type
    cdef cy_cccl_value_t value_data;

    def __cinit__(self, TypeInfo value_type, bytes state):
        self.state_bytes = state
        self.value_type = value_type
        self.value_data.type = value_type.get()
        self.value_data.state = <void *><const char *>state

    cdef cy_cccl_value_t * get_ptr(self):
        return &self.value_data

    cdef cy_cccl_value_t get(self):
        return self.value_data

    @property
    def type(self):
        return self.value_type

    property state:
        def __get__(self):
            return self.state_bytes

        def __set__(self, bytes new_value):
            if (len(self.state_bytes) == len(new_value)):
                self.state_bytes = new_value
                self.value_data.state = <void *><const char *>self.state_bytes
            else:
                raise ValueError("Size mismatch")


cdef class Iterator:
    cdef Op advance
    cdef Op dereference
    cdef bytes state_bytes
    cdef cy_cccl_iterator_t iter_data

    def __cinit__(self,
        int alignment,
        IntEnum iterator_type,
        Op advance_fn,
        Op dereference_fn,
        TypeInfo value_type,
        bytes state = None
    ):
        _validate_alignment(alignment)
        if not is_IteratorKind(iterator_type):
            raise TypeError("iterator_type must describe iterator kind")
        if state is None:
            state = b""
        self.advance = advance_fn
        self.dereference = dereference_fn
        self.state_bytes = state
        self.iter_data.size = len(state)
        self.iter_data.state = <void *><const char *>(state)
        self.iter_data.alignment = alignment
        self.iter_data.type = <cy_cccl_iterator_kind_t> iterator_type.value
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
            return self.state_bytes

        def __set__(self, bytes new_value):
            cdef ssize_t state_sz = len(new_value)
            self.state_bytes = new_value
            self.iter_data.size = state_sz
            self.iter_data.state = <void *><const char *>(self.state_bytes)

    @property
    def type(self):
        cdef cy_cccl_iterator_kind_t it_kind = self.iter_data.type
        if it_kind == cy_cccl_iterator_kind_t.cy_CCCL_POINTER:
            return IteratorKind.POINTER
        else:
            return IteratorKind.ITERATOR


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
    cdef void *t_ptr = (<void *><size_t>temp_storage_ptr) if temp_storage_ptr else NULL
    cdef size_t t_sz = <size_t>temp_storage_bytes
    cdef CUstream c_stream = <CUstream><size_t>(stream) if stream else NULL
    status = cccl_device_reduce(
        build.get(),
        t_ptr,
        &t_sz,
        d_in.get(),
        d_out.get(),
        num_items,
        op.get(),
        h_init.get(),
        c_stream
    )
    return status, <object>t_sz


cpdef CUresult device_reduce_cleanup(DeviceReduceBuildResult build):
    cdef CUresult status

    status = cccl_device_reduce_cleanup(build.get_ptr())
    return status
