import ctypes

import cupy as cp
import numpy as np
import pytest

import cuda.parallel.experimental._cy_bindings as cyb


@pytest.fixture(
    params=[
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FLOAT32",
        "FLOAT64",
        "STORAGE",
    ]
)
def cccl_type_enum(request):
    return getattr(cyb.TypeEnum, request.param)


@pytest.fixture(params=["STATEFUL", "STATELESS"])
def cccl_op_kind(request):
    return getattr(cyb.OpKind, request.param)


@pytest.fixture(params=["POINTER", "ITERATOR"])
def cccl_iterator_kind(request):
    return getattr(cyb.IteratorKind, request.param)


def test_TypeEnum_positive(cccl_type_enum):
    assert cyb.is_TypeEnum(cccl_type_enum)
    assert "TypeEnum" in repr(cccl_type_enum)
    assert "TypeEnum" in str(cccl_type_enum)
    assert isinstance(cccl_type_enum.value, int)


def test_TypeEnum_negative(cccl_op_kind):
    assert not cyb.is_TypeEnum(cccl_op_kind)


def test_OpKind_positive(cccl_op_kind):
    assert cyb.is_OpKind(cccl_op_kind)
    assert "OpKind" in repr(cccl_op_kind)
    assert "OpKind" in str(cccl_op_kind)
    assert isinstance(cccl_op_kind.value, int)


def test_OpKind_negatuve(cccl_iterator_kind):
    assert not cyb.is_OpKind(cccl_iterator_kind)


def test_IteratorKind_positive(cccl_iterator_kind):
    assert cyb.is_IteratorKind(cccl_iterator_kind)
    assert "IteratorKind" in repr(cccl_iterator_kind)
    assert "IteratorKind" in str(cccl_iterator_kind)
    assert isinstance(cccl_iterator_kind.value, int)


def test_IteratorKind_negative(cccl_op_kind):
    assert not cyb.is_IteratorKind(cccl_op_kind)


def test_pointer_as_bytes():
    ptr_test_value = 42
    b1 = cyb.pointer_as_bytes(ptr_test_value)
    assert isinstance(b1, bytes)
    b2 = bytes(ctypes.c_void_p(ptr_test_value))
    assert b1 == b2


def test_Op_default():
    res = cyb.Op()
    assert isinstance(res, cyb.Op)


def test_Op_state_setter():
    res = cyb.Op()
    bytes = b"\x00" * 20
    res.state = bytes


def test_Op_state_getter():
    res = cyb.Op()
    assert isinstance(res.state, bytes)


def test_Op_params_stateless():
    fake_ltoir = b"\x00" * 127
    res = cyb.Op(name="fn", operator_type=cyb.OpKind.STATELESS, ltoir=fake_ltoir)

    assert isinstance(res, cyb.Op)


def test_Op_params_stateful():
    fake_ltoir = b"\x42" * 127
    fake_state = b"\x01" * 16
    res = cyb.Op(
        name="fn",
        operator_type=cyb.OpKind.STATEFUL,
        ltoir=fake_ltoir,
        state=fake_state,
        state_alignment=16,
    )

    assert isinstance(res, cyb.Op)
    assert res.state == fake_state
    assert res.ltoir == fake_ltoir


def test_TypeInfo_ctor(cccl_type_enum):
    ti = cyb.TypeInfo(4, 1, cccl_type_enum)
    assert isinstance(ti, cyb.TypeInfo)


def test_TypeInfo_validate(cccl_type_enum):
    # size must positive
    with pytest.raises(ValueError):
        cyb.TypeInfo(0, 1, cyb.TypeEnum.FLOAT32)

    # alignment must be positive, power of two
    with pytest.raises(ValueError):
        cyb.TypeInfo(8, 3, cyb.TypeEnum.FLOAT32)


def test_Value_ctor():
    ti = cyb.TypeInfo(64, 64, cyb.TypeEnum.UINT64)
    state = bytearray(ctypes.c_uint64(2**63 + 17))
    v = cyb.Value(ti, state)
    assert isinstance(v, cyb.Value)


def test_Iterator_ctor1():
    fake_ptr = 42
    type_info = cyb.TypeInfo(32, 32, cyb.TypeEnum.INT32)
    cccl_it = cyb.Iterator(
        1,  # state alignment
        cyb.IteratorKind.POINTER,
        cyb.Op(),
        cyb.Op(),
        type_info,
        cyb.Pointer(cyb.PointerProxy(fake_ptr, None)),
    )
    assert isinstance(cccl_it, cyb.Iterator)


def test_Iterator_ctor2():
    fake_ptr = 42
    type_info = cyb.TypeInfo(32, 32, cyb.TypeEnum.INT32)
    cccl_it = cyb.Iterator(
        1,  # state alignment
        cyb.IteratorKind.ITERATOR,
        cyb.Op(),
        cyb.Op(),
        type_info,
        cyb.IteratorState(
            cyb.pointer_as_bytes(fake_ptr),
        ),
    )
    assert isinstance(cccl_it, cyb.Iterator)


def test_CommonData():
    cub_path = "/example/path/to/cub/includes"
    thrust_path = "/example/path/to/thrust/includes"
    libcudacxx_path = "/example/path/to/libcudacxx/includes"
    gtk_path = "/usr/local/cuda/include"
    common_data = cyb.CommonData(8, 6, cub_path, thrust_path, libcudacxx_path, gtk_path)
    assert isinstance(common_data, cyb.CommonData)


def test_cy_PointerProxy():
    d = cp.ones(128, dtype="u1")
    ptr = d.data.ptr
    pp1 = cyb.PointerProxy(ptr, d)
    ctp = ctypes.c_void_p(ptr)
    pp2 = cyb.PointerProxy(ctp, d)

    assert pp1.reference is d
    assert pp2.reference is d
    assert pp1.pointer == ptr
    assert pp2.pointer == ptr


def test_cy_IteratorStateView():
    # typed pointer
    p = ctypes.pointer(ctypes.c_int(42))
    sv1 = cyb.IteratorStateView(p, 4, p)
    vp = ctypes.cast(p, ctypes.c_void_p)
    sv2 = cyb.IteratorStateView(vp, 4, p)
    sv3 = cyb.IteratorStateView(vp.value, 4, p)

    assert sv1.pointer == vp.value
    assert sv2.pointer == vp.value
    assert sv3.pointer == vp.value

    assert sv1.reference is p
    assert sv2.reference is p
    assert sv3.reference is p


def test_cy_reduce_basic_pointer():
    import cuda.parallel.experimental.algorithms._cy_reduce as cyr

    d = cp.ones(10, dtype="i4")
    res = cp.empty(tuple(), dtype=d.dtype)
    h_init = np.zeros(tuple(), dtype=d.dtype)

    def my_add(a, b):
        return a + b

    alg = cyr.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, d.size, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)
    alg(scratch, d, res, d.size, h_init)

    assert res.get() == d.size


def test_cy_reduce_basic_iterator():
    import cuda.parallel.experimental.algorithms._cy_reduce as cyr
    import cuda.parallel.experimental.iterators._cy_iterators as iterators

    n = 15
    dt = cp.int32
    d = iterators.CountingIterator(np.int32(0))
    res = cp.empty(tuple(), dtype=dt)
    h_init = np.zeros(tuple(), dtype=dt)

    def my_add(a, b):
        return a + b

    alg = cyr.reduce_into(d, res, my_add, h_init)
    temp_bytes = alg(None, d, res, n, h_init)
    scratch = cp.empty(temp_bytes, dtype=cp.uint8)
    alg(scratch, d, res, n, h_init)

    assert res.get() * 2 == n * (n - 1)
