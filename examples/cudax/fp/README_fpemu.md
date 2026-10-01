FPEMU — IEEE-754 Double Precision in Software
=============================================

`fpemu` provides IEEE-754 double-precision arithmetic built from 32-bit integer and float
operations rather than native FP64 instructions. It is for hardware where FP64 throughput is
limited or absent: the emulation runs on the units the GPU has in abundance, so a `double`
workload can be faster in software than in the hardware path it would otherwise use — and it
runs at all on parts with no usable FP64 at all.

At its default accuracy the packed form reproduces `double` bit for bit, so the trade being
made is throughput against the FP64 pipeline, not accuracy against `double`. Accuracy only
enters the picture if you deliberately step down a level, or step up to the unpacked
representation.

For the example set, see the [examples README](README.md).

Where this pays off
-------------------

Measured on a midpoint-rule π integration — 2^26 terms, five arithmetic operations per term, no
memory traffic worth speaking of — on L40S (Ada), RTX PRO 6000 Blackwell and B300, where native
FP64 runs at roughly 1/35 of FP32. "Correct digits" means correct decimal digits of the computed
integral.

| Type | Correct digits | Speed vs native `double` |
|---|---|---|
| `double`, in hardware | 14.8 | 1.0× |
| `fp64emu_high` | 14.8, bit-identical | 1.01–1.18× |
| `fp64emu_unpacked_high` | 16.3 | **1.53–1.77×** |
| `fp64emu_unpacked_mid` | 16.4 | **1.88–2.13×** |

Two results to read off it. The packed form at `high` reproduces `double` bit for bit and is
never slower than it, so exact FP64 semantics are available without using the FP64 pipe at all.
And the unpacked form beats hardware `double` on *both* axes — about 1.5 digits more accuracy at
roughly twice the throughput — which is the reserve bits and the deferred rounding described
below. Cost relative to FP32 varies more across architectures than pair arithmetic does, 15–24%
against a few percent, because the emulation also does bit manipulation whose relative
throughput differs per part; sm_120 is the best host of the three for bit-exact software FP64.

**The emulation executes on the integer and FP32 pipes and never touches an FP64 unit**, which
has an interesting consequence on these devices: the FP64 hardware sits completely idle while
the emulation runs, and it would still deliver its own `double` rate if asked. Splitting a
reduction across both — part of the range on native `double` through the FP64 pipe, the rest on
`fp64emu_unpacked` through the FP32 and integer pipes — sums to **2.9–3.1× `double`** at
double-precision quality, against 1.9× for the emulation alone. Treat that as an upper bound
rather than a result: it is the arithmetic sum of the two measured halves, which in reality
share issue bandwidth, the register file and the scheduler, and no such split kernel has been
built yet. It is why the unpacked form is the natural one for a partition of this kind, its
pack/unpack boundary being exactly where the handover belongs anyway.

None of this applies where FP64 already runs at full rate: on B200 the FP64 pipe is the fast
path, and emulation only slows the mix down.

The two representations
----------------------

| Type | Stored as | Size | Construction |
|---|---|---|---|
| `fp64emu` | The 64-bit IEEE bit pattern, exactly as a `double` is stored | 8 bytes | Implicit from the built-in arithmetic types |
| `fp64emu_unpacked` | Separate sign, exponent and mantissa fields | 16 bytes | Explicit — the cast is written out |

The packed form is the drop-in one: same size as a `double`, same bit pattern, converts
implicitly, and at `high` accuracy produces identical results. Its advantage over `double` is
purely where the work happens.

The unpacked form exists to avoid re-packing between operations, and that turns out to buy
accuracy as well as speed. Its mantissa field extends the standard 53-bit binary64 significand
by 9 extra bits, so intermediate computations on values that stay unpacked run on **62
significand bits** and round to the storage format once — at the pack — instead of after every
operation. Over a chain it is therefore both faster, since the pack/unpack tax is paid once at
the boundary rather than per operation, and more accurate. On the integration measured above it
gains about 1.4 decimal digits at `high` and 3.1 digits at `mid`.

Two consequences follow, and both matter:

- A chain of unpacked operations is **not bit-identical to `double`**, even at
  `fpemu_accuracy::high`, whose individual operations are. The behaviour is that of computing
  in x87 extended precision and storing at the end. Code that must reproduce `double` exactly
  should use the packed form.
- 16 bytes per value against 8 can cost occupancy in kernels with many live values.

Accuracy levels
---------------

Selected per type, and available for both representations (`fp64emu_high`, `fp64emu_mid`,
`fp64emu_low`, and the same names with `_unpacked_`):

| Type | Level | Mantissa | Range |
|---|---|---|---|
| `fp64emu_high` | `fpemu_accuracy::high` | Correctly rounded | Full IEEE-754: infinities, NaNs, subnormals |
| `fp64emu_mid` | `fpemu_accuracy::mid` | 1–2 ULP error | Normal range only |
| `fp64emu_low` | `fpemu_accuracy::low` | Up to half the mantissa bits lost | Normal range only |
| `fp64emu` | `fpemu_accuracy::def` | The default selector, equal to `high` | Full IEEE-754 |

Note that the default is the IEEE-correct level, so the type is safe to reach for without
knowing this table — the lower levels are opt-in. Note also that stepping down costs range as
well as precision: `mid` and `low` cover the normal range only, and do not carry the
subnormal and special-value handling that `high` does.

The four rounding modes `rn` (nearest), `rz` (toward zero), `ru` (toward +∞) and `rd`
(toward −∞) are available at every accuracy level.

Using the types
---------------

```c++
#include <cuda/fpemu>
```

The CCCL FP component lives in `cuda::experimental` (to be promoted to `cuda::` later). The
examples abbreviate it rather than using a using-directive:

```c++
namespace cudax = cuda::experimental;
```

Type names carry the `cudax::` prefix; the standard-named math functions are left
**unqualified** and found by argument-dependent lookup, so a body of `double` code keeps its
call sites when the type underneath is swapped:

```c++
cudax::fp64emu x = 2.0;      // implicit, as to double
auto r = sqrt(x);            // unqualified: ADL finds the fpemu overload
```

### Construction and conversion

| Conversion | Packed `fp64emu` | Unpacked `fp64emu_unpacked` |
|---|---|---|
| `double` → | implicit | cast required |
| `float` → | implicit | cast required |
| `int32_t` → | implicit | cast required |
| `int64_t` → | cast required | cast required |
| → `double` | implicit | implicit |
| → `float` | cast required | cast required |

The packed form is deliberately as permissive as `double` itself, which is what makes it a
drop-in. The unpacked form asks for the cast everywhere, since entering it is a change of
representation rather than just of type — the practical consequence in real code is that
`acc += 0.5` needs to become `acc += cudax::fp64emu_unpacked{0.5}`.

### Operations

| Operation | C++ | Built-in prefix |
|---|---|---|
| Add, subtract, multiply, divide | `+`, `-`, `*`, `/` and compound forms | `dadd`, `dsub`, `dmul`, `ddiv` |
| Square root | `sqrt()` | `dsqrt` |
| Fused multiply-add | `fma()` | `fma` |
| Multiply-add | `mad()` | `mad` |
| Dot product | `dot()` | `dot` |
| Complex multiply | `cmul()` | `cmul` |

All six comparisons follow IEEE-754 semantics, for both representations. Mixed-type
arithmetic works directly, so an `fpemu` value combines with a built-in scalar without a cast
on the scalar side.

There is **no transcendental math header** for `fpemu` — no `exp`, `log` or trigonometry.
Arithmetic, `fma` and `sqrt` are the surface. This is the main functional difference from the
`fpmp` types, which have `<cuda/fpmp_math>`.

`fpemu` objects may be declared `volatile`, for the legacy CUDA pattern of holding
shared-memory scalars in volatile variables. As for `fpmp2`, support is limited to storage —
loads, stores and copies, with a bit-preserving round-trip and trivial copyability retained.
Arithmetic and comparison take `const fpemu&` and will not bind a volatile lvalue, so compute
on a non-volatile copy and store the result back.

### The scalar built-ins

Underneath the class is a C-callable layer operating on `__fpbits64`, a `uint64_t` holding the
IEEE-754 bit pattern. It is what to reach for when the accuracy level needs to be chosen at
the call site rather than carried by a type. The declarations come with `<cuda/fpemu>`:

```
__fp64emu_<op>_<rm>                  default (== high)
__fp64emu_high_<op>_<rm>             correctly rounded, full IEEE-754 specials
__fp64emu_mid_<op>_<rm>              1-2 LSB error, normal range
__fp64emu_low_<op>_<rm>              up to half the mantissa, normal range
```

with `__fp64emu_unpacked_...` mirroring all four for the unpacked bit layout, `<op>` the
operation from the table above and `<rm>` the rounding mode.

```c++
__fpbits64 x = __fp64emu_from_double(1.2345);
__fpbits64 y = __fp64emu_from_double(2.3456);

__fpbits64 r = __fp64emu_high_dmul_rn(x, y);   // correctly rounded multiply
double out   = __fp64emu_to_double(r);
```

The CUDA-style spellings (`__dadd_rn`, `__dmul_rn`, `__fma_rn`, ...) also work on the class
types and deduce the accuracy level from the argument type, so existing intrinsic call sites
port across unchanged.

The example
-----------

### fpemu.cu

Both representations offer the same operations, so one example covers both, running one kernel
per representation — `fpemu_packed_operations` and `fpemu_unpacked_operations` — over the same
set of operations with the same inputs.

Demonstrated:

- construction from literals of several source types
- arithmetic: `+`, `-`, `*`, `/`
- mixed-type arithmetic — an `fpemu` value combined directly with a `double` literal, with an
  `int`, with the scalar on the left, and the `+=` scalar accumulate path
- `sqrt` and `fma`
- comparison operators
- the accuracy levels, on the two operations where they visibly disagree
- accumulating terms that fall below one ulp of the running total, where the unpacked form's
  deferred rounding separates it from `double` and from the packed form
- converting between the two representations
- the same source running on both host and device

Most of the two kernels read identically, which is the point: the differences worth watching
for are that the unpacked type needs a written-out cast wherever the packed one converts
implicitly, and that deferring the rounding pays off over a long run of operations.

That last part is what the closing section measures, and it is the one result that cannot be
seen in a single operation. It adds 1000 terms of `1e-17` onto `1.0`. One ulp of `1.0` is
about `2.22e-16`, so each individual term is some twenty times too small to move a `double` at
all: `double` and packed `fp64emu` both never leave `1.0`, while the unpacked form's extra
significand bits accumulate the terms and reach the exact answer. The accuracy levels are shown on
subtraction and multiplication specifically because division and `sqrt` happen to agree across
the levels for these operands.

Each operation prints its inputs and its result. Nothing is checked against a reference: the
point is to show what the interface looks like in use.
