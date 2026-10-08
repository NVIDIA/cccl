# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Regression tests for issue #11347: cuda.compute's struct layout for
complex64/complex128 fields (built via NumPy's ``align=True``, which
under-reports their alignment relative to libcu++'s
``cuda::std::complex<T>``) can disagree with the layout a RawOp written
against ``<cuda/std/complex>`` assumes, silently corrupting the fields it
reads/writes.

``RawOp.compile`` now rejects such (type, RawOp) combinations outright, since
cuda.compute cannot introspect precompiled LTO-IR to know whether it actually
relies on the standard layout. Python-callable operators are unaffected,
since numba-cuda-mlir represents complex values consistently end-to-end
regardless of NumPy's alignment bookkeeping.
"""

import numpy as np
import pytest
from _utils.device_array import DeviceArray, get_compute_capability
from cuda.core import Program, ProgramOptions

import cuda.compute
from cuda.compute import CountingIterator, ZipIterator, types
from cuda.compute._cpp_compile import _get_include_paths
from cuda.compute.op import RawOp

pytestmark = pytest.mark.no_numba


def get_arch():
    cc_major, cc_minor = get_compute_capability()
    return f"sm_{cc_major}{cc_minor}"


def compile_to_ltoir(source: str, arch: str, include_paths=None) -> bytes:
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
        std="c++17",
        include_path=include_paths,
    )
    return Program(source, "c++", options=opts).compile("ltoir").code


# ---------------------------------------------------------------------------
# Unit-level: the layout check itself, no LTO-IR compilation involved (a
# dummy RawOp is enough, since the check runs before `self._ltoir` is used).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "idx_dtype,cplx_dtype",
    [("int64", "complex128"), ("int32", "complex64")],
)
def test_mismatched_complex_struct_rejected_without_compiling(idx_dtype, cplx_dtype):
    fields = {
        "idx": types.from_numpy_dtype(np.dtype(idx_dtype)),
        "val": types.from_numpy_dtype(np.dtype(cplx_dtype)),
    }
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((td,), None)


def test_coincidental_complex_struct_accepted_without_compiling():
    """int64 + complex64: NumPy's (wrong) complex64 alignment of 4 still
    happens to place `val` at the same offset the true device alignment (8)
    would, so this specific field order/type combination is not actually
    broken and must not be rejected."""
    fields = {
        "idx": types.int64,
        "val": types.from_numpy_dtype(np.dtype("complex64")),
    }
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    op.compile((td,), None)  # must not raise


def test_plain_struct_without_complex_field_accepted():
    fields = {"a": types.int32, "b": types.float64}
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    op.compile((td,), None)  # must not raise


def test_nested_struct_with_mismatched_complex_rejected():
    """Inner={idx:int64, val:complex128} is broken on its own (val needs
    native offset 16, NumPy gives it 8), and here nothing follows `n` in
    Outer to absorb the resulting size difference (24 vs the native 32), so
    the complete Outer value is genuinely wrong end to end and must still be
    rejected under the top-level-only comparison (see
    test_nested_struct_with_absorbed_padding_is_accepted for the case where
    a trailing field *does* absorb the difference and must NOT be rejected).
    """
    inner = types.struct(
        {"idx": types.int64, "val": types.from_numpy_dtype(np.dtype("complex128"))}
    )
    outer = types.struct({"n": inner})
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((outer,), None)


def test_nested_struct_with_absorbed_padding_is_accepted():
    """Inner={z:complex64, n:int32} is broken on its own (NumPy: size 12,
    alignment 4; native: size 16, alignment 8 -- a size-and-alignment
    mismatch whose *member offsets* happen to agree: z@0 and n@8 either way,
    since z is the first field and n's own 4-byte alignment need is already
    satisfied at offset 8 regardless of which alignment is used for z), but
    Outer={inner:Inner, tail:int64} places `tail` right after it:
    int64 needs 8-byte alignment either way, so both NumPy's (wrong) and
    native's (correct) computation round `tail`'s offset up to the same 16,
    giving byte-for-byte identical complete Outer layouts (offsets [0, 16],
    itemsize 24, alignment 8 both ways). A RawOp casting the *complete* Outer
    value to native C++ reads every field correctly, so this must be
    accepted -- rejecting it (as validating Inner independently would) is a
    false positive.
    """
    inner = types.struct(
        {"z": types.from_numpy_dtype(np.dtype("complex64")), "n": types.int32}
    )
    outer = types.struct({"inner": inner, "tail": types.int64})
    op = RawOp(ltoir=b"", name="unused")
    op.compile((outer,), None)  # must not raise


def test_nested_struct_with_reordered_fields_is_rejected_despite_absorbed_total_size():
    """Inner={n:int32, z:complex64} -- z SECOND this time, unlike the accepted
    case above. NumPy places z at offset 4 (using its own 4-byte complex64
    alignment); native CUDA/C++ places it at offset 8 (8-byte alignment).
    Outer={inner:Inner, tail:int64} still absorbs Inner's *total* size/
    alignment difference at the outer level (both give offsets=[0, 16],
    itemsize 24) -- a check that only compares the outer struct's own
    immediate offsets/itemsize/alignment would wrongly accept this. But `z`
    itself sits at absolute byte offset 4 in cuda.compute's value and byte
    offset 8 in the native one: a RawOp reading `outer.inner.z` still gets
    the wrong bytes, so this must be rejected on `z`'s own absolute offset,
    independent of whether the outer struct's totals happen to match."""
    inner = types.struct(
        {"n": types.int32, "z": types.from_numpy_dtype(np.dtype("complex64"))}
    )
    outer = types.struct({"inner": inner, "tail": types.int64})
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match=r"'<value>\.inner\.z' sits at absolute offset"):
        op.compile((outer,), None)


def test_pointer_buried_under_nested_inline_struct_is_rejected():
    """A pointer field is not always a *direct* field of the struct being
    checked -- here it is two inline-struct levels down
    (Outer -> Holder -> pointer-to-Bad). Holder and Outer themselves contain
    only pointer-sized fields and have perfectly ordinary layouts either way,
    but the pointee `Bad` has a genuinely mismatched complex field. A RawOp
    can still dereference `outer.holder.p->z` and get the wrong bytes, so the
    validator must keep walking into inline struct fields (not just the
    top-level struct's direct fields) to find pointers to check."""
    bad = types.struct(
        {"n": types.int32, "z": types.from_numpy_dtype(np.dtype("complex64"))}
    )
    holder = types.struct({"p": bad.pointer()})
    outer = types.struct({"holder": holder})
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((outer,), None)


def test_subarray_of_complex_field_mismatch_is_rejected():
    """{int32 n; complex64 z[2];}: a fixed-size subarray field has NumPy
    kind 'V', not 'c', so a check keyed only on `dtype.kind == "c"` would
    miss it entirely. NumPy places `z` at offset 4 (subarray alignment 4,
    same as a lone complex64); native CUDA/C++ needs offset 8 (array
    elements take their element type's alignment, 8)."""
    fields = {
        "n": types.int32,
        "z": types.from_numpy_dtype(np.dtype((np.complex64, (2,)))),
    }
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((td,), None)


def test_subarray_of_complex_field_coincidental_match_is_accepted():
    """{int64 n; complex64 z[2];}: int64's own 8-byte alignment already
    forces `z` to offset 8 under both NumPy's and native's rules, so this
    specific combination is genuinely correct and must not be rejected."""
    fields = {
        "n": types.int64,
        "z": types.from_numpy_dtype(np.dtype((np.complex64, (2,)))),
    }
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    op.compile((td,), None)  # must not raise


def test_nested_subarray_of_complex_mismatch_is_rejected_at_correct_path():
    inner = types.struct(
        {"n": types.int32, "z": types.from_numpy_dtype(np.dtype((np.complex64, (2,))))}
    )
    outer = types.struct({"tail": types.int64, "inner": inner})
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match=r"'<value>\.inner\.z' sits at absolute offset"):
        op.compile((outer,), None)


def test_pointer_to_subarray_of_complex_struct_is_rejected():
    inner = types.struct(
        {"n": types.int32, "z": types.from_numpy_dtype(np.dtype((np.complex64, (2,))))}
    )
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((inner.pointer(),), None)


def test_single_complex_field_alignment_only_mismatch_is_rejected():
    """{complex64 z} alone: NumPy gives offsets=[0], itemsize=8 -- identical
    to what native CUDA/C++ would give -- but NumPy's struct alignment (4)
    differs from the native requirement (8). An offsets/itemsize-only check
    misses this entirely; the struct's own alignment must be compared too."""
    fields = {"z": types.from_numpy_dtype(np.dtype("complex64"))}
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((td,), None)


def test_trailing_fields_alignment_only_mismatch_is_rejected():
    """{complex64 z, int32 n1, int32 n2}: offsets [0, 8, 12] and itemsize 16
    are identical between NumPy and native CUDA/C++ -- only the struct's
    overall alignment differs (4 vs 8) -- another case an offsets/itemsize
    -only check would silently accept."""
    fields = {
        "z": types.from_numpy_dtype(np.dtype("complex64")),
        "n1": types.int32,
        "n2": types.int32,
    }
    td = types.struct(fields)
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((td,), None)


def test_bare_complex_argument_alignment_mismatch_is_rejected():
    """A complex128 value passed directly (not inside a struct) still has an
    under-reported alignment (8 instead of the native 16) and must be
    rejected too -- there is no struct here for an offsets-only check to even
    look at."""
    td = types.from_numpy_dtype(np.dtype("complex128"))
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="byte aligned"):
        op.compile((td,), None)


def test_pointer_to_mismatched_complex_struct_rejected():
    inner = types.struct(
        {"idx": types.int64, "val": types.from_numpy_dtype(np.dtype("complex128"))}
    )
    op = RawOp(ltoir=b"", name="unused")
    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        op.compile((inner.pointer(),), None)


# ---------------------------------------------------------------------------
# End-to-end: the actual issue #11347 repro shape, through a real RawOp
# compiled from C++ against <cuda/std/complex>.
# ---------------------------------------------------------------------------

UNPACK = r"""
#include <cuda/std/complex>
#include <cuda/std/cstddef>
struct Pair { IDX_T idx; cuda::std::complex<CPLX_T> val; };
struct Out  { long long idx; double re; double im; };
extern "C" __device__ void unpack(const void* in, void* out) {
    const Pair* p = static_cast<const Pair*>(in);
    Out* o = static_cast<Out*>(out);
    o->idx = p->idx; o->re = p->val.real(); o->im = p->val.imag();
}
"""


def _include_paths():
    return _get_include_paths()


def test_raw_op_rejects_zip_iterator_with_mismatched_complex_layout():
    """The exact shape of the issue #11347 repro: a RawOp reading a
    ZipIterator(int64, complex128[]) item as a native
    {long long; cuda::std::complex<double>} struct must be rejected before
    any kernel is built, instead of silently reading garbage."""
    OutT = types.struct({"idx": types.int64, "re": types.float64, "im": types.float64})
    src = UNPACK.replace("IDX_T", "long long").replace("CPLX_T", "double")
    ltoir = compile_to_ltoir(src, get_arch(), include_paths=_include_paths())
    op = RawOp(ltoir=ltoir, name="unpack")

    h = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    zip_it = ZipIterator(CountingIterator(np.int64(0)), DeviceArray.from_numpy(h))
    d_out = DeviceArray.empty(2, OutT.dtype)

    with pytest.raises(TypeError, match="native CUDA/C\\+\\+ code"):
        cuda.compute.unary_transform(d_in=zip_it, d_out=d_out, num_items=2, op=op)


def test_raw_op_accepts_and_correctly_reads_coincidentally_aligned_complex():
    """int64 + complex64 via CountingIterator/complex64[]: unlike int32 +
    complex64 (still broken -- NumPy's complex64 alignment of 4 places `val`
    at offset 4, native needs 8) or int64 + complex128 above, this specific
    field order's NumPy layout happens to already match the native CUDA
    layout (both place `val` at offset 8), so the RawOp must be allowed to
    run and must read back correct values (not merely "not raise")."""
    OutT = types.struct({"idx": types.int64, "re": types.float64, "im": types.float64})
    src = UNPACK.replace("IDX_T", "long long").replace("CPLX_T", "float")
    ltoir = compile_to_ltoir(src, get_arch(), include_paths=_include_paths())
    op = RawOp(ltoir=ltoir, name="unpack")

    h = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    zip_it = ZipIterator(CountingIterator(np.int64(0)), DeviceArray.from_numpy(h))
    d_out = DeviceArray.empty(3, OutT.dtype)

    cuda.compute.unary_transform(d_in=zip_it, d_out=d_out, num_items=3, op=op)

    result = d_out.copy_to_host()
    np.testing.assert_array_equal(result["idx"], [0, 1, 2])
    np.testing.assert_allclose(result["re"], h.real)
    np.testing.assert_allclose(result["im"], h.imag)
