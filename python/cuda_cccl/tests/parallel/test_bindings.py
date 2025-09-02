import ctypes

import pytest

import cuda.cccl.parallel.experimental._bindings as bindings


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
    return getattr(bindings.TypeEnum, request.param)


@pytest.fixture(params=["STATEFUL", "STATELESS"])
def cccl_op_kind(request):
    return getattr(bindings.OpKind, request.param)


@pytest.fixture(params=["POINTER", "ITERATOR"])
def cccl_iterator_kind(request):
    return getattr(bindings.IteratorKind, request.param)


def test_TypeEnum_positive(cccl_type_enum):
    assert isinstance(cccl_type_enum, bindings.TypeEnum)
    assert isinstance(cccl_type_enum.value, int)


def test_TypeEnum_negative(cccl_iterator_kind):
    assert not isinstance(cccl_iterator_kind, bindings.TypeEnum)


def test_OpKind_positive(cccl_op_kind):
    assert isinstance(cccl_op_kind, bindings.OpKind)
    assert isinstance(cccl_op_kind.value, int)


def test_OpKind_negative(cccl_iterator_kind):
    assert not isinstance(cccl_iterator_kind, bindings.OpKind)


def test_IteratorKind_positive(cccl_iterator_kind):
    assert isinstance(cccl_iterator_kind, bindings.IteratorKind)
    assert isinstance(cccl_iterator_kind.value, int)


def test_IteratorKind_negative(cccl_type_enum):
    assert not isinstance(cccl_type_enum, bindings.IteratorKind)


def test_Op_default():
    res = bindings.Op()
    assert isinstance(res, bindings.Op)


def test_Op_state_setter():
    res = bindings.Op()
    bytes = b"\x00" * 20
    res.state = bytes


def test_Op_state_getter():
    res = bindings.Op()
    assert isinstance(res.state, bytes)


def test_Op_params_stateless():
    fake_ltoir = b"\x00" * 127
    res = bindings.Op(
        name="fn", operator_type=bindings.OpKind.STATELESS, ltoir=fake_ltoir
    )

    assert isinstance(res, bindings.Op)


def test_Op_params_stateful():
    fake_ltoir = b"\x42" * 127
    fake_state = b"\x01" * 16
    res = bindings.Op(
        name="fn",
        operator_type=bindings.OpKind.STATEFUL,
        ltoir=fake_ltoir,
        state=fake_state,
        state_alignment=16,
    )

    assert isinstance(res, bindings.Op)
    assert res.state == fake_state
    assert res.ltoir == fake_ltoir


def test_TypeInfo_ctor(cccl_type_enum):
    ti = bindings.TypeInfo(4, 1, cccl_type_enum)
    assert isinstance(ti, bindings.TypeInfo)


def test_TypeInfo_validate(cccl_type_enum):
    # size must positive
    with pytest.raises(ValueError):
        bindings.TypeInfo(0, 1, bindings.TypeEnum.FLOAT32)

    # alignment must be positive, power of two
    with pytest.raises(ValueError):
        bindings.TypeInfo(8, 3, bindings.TypeEnum.FLOAT32)


def test_Value_ctor():
    ti = bindings.TypeInfo(64, 64, bindings.TypeEnum.UINT64)
    state = bytearray(ctypes.c_uint64(2**63 + 17))
    v = bindings.Value(ti, state)
    assert isinstance(v, bindings.Value)


def test_Iterator_ctor1():
    fake_ptr = 42
    type_info = bindings.TypeInfo(32, 32, bindings.TypeEnum.INT32)
    cccl_it = bindings.Iterator(
        1,  # state alignment
        bindings.IteratorKind.POINTER,
        bindings.Op(),
        bindings.Op(),
        type_info,
        bindings.Pointer(fake_ptr),
    )
    assert isinstance(cccl_it, bindings.Iterator)


def test_Iterator_ctor2():
    fake_ptr = 42
    type_info = bindings.TypeInfo(32, 32, bindings.TypeEnum.INT32)
    cccl_it = bindings.Iterator(
        1,  # state alignment
        bindings.IteratorKind.ITERATOR,
        bindings.Op(),
        bindings.Op(),
        type_info,
        bindings.IteratorState(
            ctypes.c_void_p(fake_ptr),
        ),
    )
    assert isinstance(cccl_it, bindings.Iterator)


def test_CommonData():
    cub_path = "/example/path/to/cub/includes"
    thrust_path = "/example/path/to/thrust/includes"
    libcudacxx_path = "/example/path/to/libcudacxx/includes"
    gtk_path = "/usr/local/cuda/include"
    common_data = bindings.CommonData(
        8, 6, cub_path, thrust_path, libcudacxx_path, gtk_path
    )
    assert isinstance(common_data, bindings.CommonData)
