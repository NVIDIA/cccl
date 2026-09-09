# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Field conversion and access semantics for ``gpu_struct``.

Covers how values are converted as they are packed into struct fields (the
constructor, the tuple-to-struct cast and the struct-to-struct cast) and how
fields are selected by index.
"""

import numpy as np
import pytest
from _utils.device_array import DeviceArray

import cuda.compute
from cuda.compute import gpu_struct

# Signed inputs whose sign must survive being widened into a wider field.
NEGATIVE_INPUT = np.array([-1, -2147483648, -7, 0, 5], dtype=np.int32)


def test_constructor_widens_signed_field():
    """``Wide(x)`` with a narrower signed ``x`` sign-extends into the field."""
    Wide = gpu_struct({"a": np.int64, "b": np.int64})

    def widen(x):
        return Wide(x, x)

    d_in = DeviceArray.from_numpy(NEGATIVE_INPUT)
    d_out = DeviceArray.empty(NEGATIVE_INPUT.shape, Wide.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=widen, num_items=NEGATIVE_INPUT.size
    )

    result = d_out.copy_to_host()
    expected = NEGATIVE_INPUT.astype(np.int64)
    np.testing.assert_array_equal(result["a"], expected)
    np.testing.assert_array_equal(result["b"], expected)


def test_tuple_return_widens_signed_field():
    """A tuple of narrower signed values packed into a struct sign-extends."""
    Wide = gpu_struct({"a": np.int64, "b": np.int64})

    def widen(x):
        return (x, x)

    d_in = DeviceArray.from_numpy(NEGATIVE_INPUT)
    d_out = DeviceArray.empty(NEGATIVE_INPUT.shape, Wide.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=widen, num_items=NEGATIVE_INPUT.size
    )

    result = d_out.copy_to_host()
    expected = NEGATIVE_INPUT.astype(np.int64)
    np.testing.assert_array_equal(result["a"], expected)
    np.testing.assert_array_equal(result["b"], expected)


def test_nested_struct_widens_signed_field():
    """Sign is preserved through a nested struct field built from a tuple."""
    Inner = gpu_struct({"a": np.int64})
    Outer = gpu_struct({"x": np.int64, "inner": Inner})

    def widen(x):
        return Outer(x, (x,))

    d_in = DeviceArray.from_numpy(NEGATIVE_INPUT)
    d_out = DeviceArray.empty(NEGATIVE_INPUT.shape, Outer.dtype)

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=widen, num_items=NEGATIVE_INPUT.size
    )

    result = d_out.copy_to_host()
    expected = NEGATIVE_INPUT.astype(np.int64)
    np.testing.assert_array_equal(result["x"], expected)
    np.testing.assert_array_equal(result["inner"]["a"], expected)


def test_unsigned_widening_zero_extends():
    """Unsigned sources still zero-extend when widened."""
    Wide = gpu_struct({"a": np.uint64})
    h_in = np.array([0, 1, 4294967295], dtype=np.uint32)

    def widen(x):
        return Wide(x)

    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Wide.dtype)

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=widen, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host()["a"], h_in.astype(np.uint64))


def test_nested_tuple_too_short_is_rejected():
    """A tuple with fewer values than the nested struct's fields is an error."""
    Inner = gpu_struct({"a": np.int32, "b": np.int32})
    Outer = gpu_struct({"x": np.int32, "inner": Inner})

    def build(x):
        return Outer(x, (x,))

    h_in = np.arange(3, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="tuple of size 1"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_nested_tuple_too_long_is_rejected():
    """A tuple with more values than the nested struct's fields is an error."""
    Inner = gpu_struct({"a": np.int32, "b": np.int32})
    Outer = gpu_struct({"x": np.int32, "inner": Inner})

    def build(x):
        return Outer(x, (x, x, x))

    h_in = np.arange(3, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="tuple of size 3"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_tuple_for_scalar_field_is_rejected():
    """A tuple supplied for a scalar field is an error, not an AttributeError."""
    Inner = gpu_struct({"a": np.int32, "b": np.int32})
    Outer = gpu_struct({"x": np.int32, "inner": Inner})

    def build(x):
        return Outer((x, x), Inner(x, x))

    h_in = np.arange(3, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="cannot initialize field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_constant_index_selects_the_field():
    """``struct[i]`` with a constant index reads field ``i``."""
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    def second(s):
        return s[1]

    h_in = np.zeros(4, dtype=Pair.dtype)
    h_in["a"] = np.arange(4)
    h_in["b"] = np.arange(100, 104)

    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int32))

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=second, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host(), h_in["b"])


def test_index_out_of_range_is_rejected():
    """An out-of-range constant index is reported against the struct."""
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    def out_of_range(s):
        return s[5]

    h_in = np.zeros(4, dtype=Pair.dtype)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int32))

    with pytest.raises(Exception, match="out of range"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=out_of_range, num_items=h_in.size
        )


def test_runtime_index_selects_a_field():
    """A struct can be indexed with a value only known at run time.

    Every field is read through the same expression, so the result takes the
    type the fields unify to, exactly as any other value that depends on a
    runtime condition does.
    """
    Pair = gpu_struct({"a": np.int32, "b": np.int32})

    def sum_fields(pair):
        total = 0
        for i in range(2):
            total += pair[i]
        return total

    h_in = np.zeros(2, dtype=Pair.dtype)
    h_in["a"] = [1, 3]
    h_in["b"] = [10, 20]
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int64))

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=sum_fields, num_items=h_in.size
    )

    np.testing.assert_array_equal(d_out.copy_to_host(), np.array([11, 23]))


def test_runtime_index_of_a_struct_with_differing_field_types():
    """Indexing fields of different types yields the type they unify to."""
    Mixed = gpu_struct({"a": np.int32, "b": np.float64})

    def sum_fields(mixed):
        total = 0.0
        for i in range(2):
            total += mixed[i]
        return total

    h_in = np.zeros(2, dtype=Mixed.dtype)
    h_in["a"] = [1, 3]
    h_in["b"] = [10.5, 20.5]
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.float64))

    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_out, op=sum_fields, num_items=h_in.size
    )

    np.testing.assert_allclose(d_out.copy_to_host(), np.array([11.5, 23.5]))


def test_runtime_index_keeps_the_sign_of_a_narrower_field():
    """A negative field widened into the unified type keeps its sign."""
    Mixed = gpu_struct({"a": np.int32, "b": np.int64})

    def first(mixed):
        total = 0
        for i in range(1):
            total += mixed[i]
        return total

    h_in = np.zeros(3, dtype=Mixed.dtype)
    h_in["a"] = [-1, -2147483648, 7]
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, np.dtype(np.int64))

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=first, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host(), h_in["a"].astype(np.int64))


def test_constructor_rejects_incompatible_argument_type():
    """A field cannot be initialized from an unrelated type.

    The error names the field and both types, rather than failing later while
    the constructor is lowered.
    """
    Pair = gpu_struct({"a": np.int32, "b": np.int32})
    Other = gpu_struct({"c": np.int32})

    def build(s):
        # 's' is a struct; field 'a' is a scalar.
        return Pair(s, s.c)

    h_in = np.zeros(4, dtype=Other.dtype)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Pair.dtype)

    with pytest.raises(Exception, match="cannot initialize field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=build, num_items=h_in.size
        )


def test_complex_field_is_read_and_constructed():
    """A struct may hold a complex field.

    Complex values are MLIR complex scalars in SSA but are stored as a literal
    ``{real, imag}`` LLVM struct, which is the only form the LLVM dialect
    accepts as a struct member.
    """
    Sample = gpu_struct({"z": np.complex64, "n": np.int32})

    h_in = np.zeros(4, dtype=Sample.dtype)
    h_in["z"] = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    h_in["n"] = np.arange(4)
    d_in = DeviceArray.from_numpy(h_in)

    def read_complex(s):
        return s.z

    d_z = DeviceArray.empty(h_in.shape, np.dtype(np.complex64))
    cuda.compute.unary_transform(
        d_in=d_in, d_out=d_z, op=read_complex, num_items=h_in.size
    )
    np.testing.assert_array_equal(d_z.copy_to_host(), h_in["z"])

    def scale(s):
        return Sample(s.z * 2, s.n + 1)

    d_out = DeviceArray.empty(h_in.shape, Sample.dtype)
    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=scale, num_items=h_in.size)
    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result["z"], h_in["z"] * 2)
    np.testing.assert_array_equal(result["n"], h_in["n"] + 1)


def test_operator_may_return_a_narrower_struct():
    """An operator's struct may be narrower than the declared output struct.

    Each field is converted to its declared type as the result is stored.
    """
    Narrow = gpu_struct({"a": np.int32, "b": np.int32})
    Wide = gpu_struct({"a": np.int64, "b": np.int64})

    def build(x):
        return Narrow(x, -x)

    h_in = np.arange(4, dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Wide.dtype)

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=build, num_items=h_in.size)

    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result["a"], h_in.astype(np.int64))
    np.testing.assert_array_equal(result["b"], -h_in.astype(np.int64))


# (field dtype, input dtype, input values) whose conversion depends on knowing
# the signedness of the source, the target, or both.
SIGNEDNESS_CASES = [
    (np.float64, np.uint32, [3000000000, 7]),
    (np.float64, np.uint8, [200, 7]),
    (np.float64, np.int32, [-1, -5]),
    (np.uint32, np.float64, [3000000000.0, 7.0]),
    (np.int32, np.float64, [-3.0, 2.0]),
    (np.uint64, np.uint32, [3000000000, 7]),
]


@pytest.mark.parametrize("field_dtype,input_dtype,values", SIGNEDNESS_CASES)
def test_field_conversion_preserves_signedness(field_dtype, input_dtype, values):
    """Packing a value into a field of another numeric type keeps its value.

    Converting between an integer and a float has to select the signed or the
    unsigned instruction; picking the wrong one turns a large unsigned value
    negative, or saturates a float that does not fit the signed range.
    """
    Boxed = gpu_struct({"a": field_dtype})

    def pack(x):
        return Boxed(x)

    h_in = np.array(values, dtype=input_dtype)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Boxed.dtype)

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=pack, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host()["a"], h_in.astype(field_dtype))


def test_complex_argument_rejected_for_a_real_field():
    """A complex value cannot initialize a real field.

    Converting it would drop the imaginary part silently, so the call is
    rejected while typing instead.
    """
    Real = gpu_struct({"a": np.float64})

    def pack(x):
        return Real(x + 1j)

    h_in = np.array([1.0, 2.0], dtype=np.float64)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Real.dtype)

    with pytest.raises(Exception, match="cannot initialize field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=pack, num_items=h_in.size
        )


def test_struct_of_different_layout_rejected_for_a_struct_field():
    """A struct field only accepts a struct with a matching layout.

    The struct-to-struct cast rebuilds the value field by field, so a mismatch
    would otherwise type cleanly and then fail while lowering.
    """
    OneField = gpu_struct({"p": np.int32})
    TwoField = gpu_struct({"p": np.int32, "q": np.int32})
    Outer = gpu_struct({"n": TwoField})

    def pack(x):
        return Outer(OneField(x))

    h_in = np.array([1, 2], dtype=np.int32)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Outer.dtype)

    with pytest.raises(Exception, match="cannot initialize field"):
        cuda.compute.unary_transform(
            d_in=d_in, d_out=d_out, op=pack, num_items=h_in.size
        )


def test_float_into_a_bool_field_asks_whether_it_is_nonzero():
    """A float packed into a bool field converts as ``x != 0``.

    Truncating the converted integer instead reports 0.0 as true and 2.0 as
    false.
    """
    Flag = gpu_struct({"a": np.bool_})

    def pack(x):
        return Flag(x * 0.5)

    h_in = np.array([0.5, 2.5, 0.0, -1.0], dtype=np.float64)
    d_in = DeviceArray.from_numpy(h_in)
    d_out = DeviceArray.empty(h_in.shape, Flag.dtype)

    cuda.compute.unary_transform(d_in=d_in, d_out=d_out, op=pack, num_items=h_in.size)

    np.testing.assert_array_equal(d_out.copy_to_host()["a"], (h_in * 0.5) != 0)


def test_tuple_to_struct_conversion_checks_element_types():
    """A tuple converts to a struct only if its elements convert to the fields.

    A uniform tuple is checked the same way as a mixed one: matching the field
    count is not enough, or a call types cleanly and then has no lowering.
    """
    from numba_cuda_mlir.descriptor import mlir_target

    from cuda.compute._jit import _register_struct_with_numba

    types = cuda.compute._jit._mlir.types
    typingctx = mlir_target.typing_context

    Inner = gpu_struct({"p": np.int32})
    Pair = gpu_struct({"a": np.int32, "b": np.int32})
    inner_type = _register_struct_with_numba(Inner)
    pair_type = _register_struct_with_numba(Pair)

    def converts(source):
        return pair_type.can_convert_from(typingctx, source) is not None

    # Elements convert to both fields.
    assert converts(types.UniTuple(types.int32, 2))
    # Right element type, wrong number of them.
    assert not converts(types.UniTuple(types.int32, 3))
    # Right number, but a struct does not convert to an int32 field. The mixed
    # tuple below has always been rejected; the uniform one used to be accepted.
    assert not converts(types.UniTuple(inner_type, 2))
    assert not converts(types.Tuple([types.int32, inner_type]))
