FPEMU — IEEE-754 Double Precision in Software
=============================================

`fpemu` provides IEEE-754 double-precision arithmetic built from 32-bit integer and float
operations rather than native FP64 instructions. It is for hardware where FP64 throughput is
rationed: the emulation runs on the units the GPU has in abundance, so a `double` workload can
be faster in software than in the hardware path it would otherwise use.

At its default accuracy the packed form reproduces `double` bit for bit, so the trade being
made is throughput against the FP64 pipeline, not accuracy against `double`. Accuracy only
enters the picture if you deliberately step down a level, or step up to the unpacked
representation.

For the example set, see the [examples README](README.md).

Where this pays off
-------------------

The figures below come from a midpoint-rule π integration — 2^26 terms, five arithmetic
operations per term, no memory traffic worth speaking of — on three parts where native FP64
runs at 1/64 of FP32, so they price arithmetic pipelines rather than bandwidth, and every type
runs identical code. "Correct digits" means correct decimal digits of the computed integral,
with every step carried out in the type being measured, the sequential reduction of the 131072
per-thread partials included: what is reported is therefore what a program written in that
type costs end to end, not the loop in isolation. Native `double` is measured in the same run,
as the baseline each row is reported against, and accuracy is a property of the arithmetic
rather than of the GPU, so it is one column for all three parts:

| Type | Correct digits | RTX 6000 Ada | RTX PRO 6000 Blackwell | B300 |
|---|---|---|---|---|
| `double` | 14.8 | 1.00× | 1.00× | 1.00× |
| `fp64emu` | 14.8 | 1.02× | 1.12× | 0.98× |
| `fp64emu_unpacked` | 16.1 | **1.47×** | **1.65×** | **1.46×** |
| `fp64emu_mid` | 11.3 | 1.43× | 1.61× | 1.46× |
| `fp64emu_unpacked_mid` | 14.0 | **1.78×** | **1.96×** | **1.81×** |
| `fp64emu_low` | 3.1 | 1.72× | 1.95× | 1.76× |
| `fp64emu_unpacked_low` | 3.1 | 2.12× | 2.30× | 2.05× |

Two results to read off it. The packed form at the default accuracy reproduces `double` bit
for bit — the two rows are not merely equal to the digit shown, they return the same value —
while running at parity with it, 0.98× to 1.12× across the three, so exact FP64 semantics are
available without using the FP64 pipe at all. And the unpacked form runs about 1.5× hardware
`double` and comes out 1.3 digits *ahead* of it, which is the deferred rounding described below
showing up twice over: the pack/unpack tax is paid once at the boundary rather than once per
operation, and the guard bits that survive between operations absorb error that `double` has
to round away — including the error of the long final summation, which is where most of it is.

**The emulation executes on the integer and FP32 pipes and never touches an FP64 unit**, which
has an interesting consequence on parts like these: the FP64 hardware sits idle while the
emulation runs, and would still deliver its own rate if asked. Splitting a reduction across
both — part of the range on native `double`, the rest on `fp64emu_unpacked` — is therefore
possible in principle, and the unpacked form is the natural one for it, its pack boundary being
where the handover belongs. No such kernel has been built and measured here, so treat that as a
direction rather than as a result.

None of this applies where FP64 already runs at full rate: on a part whose FP64 pipe is half of
FP32, hardware `double` is the fast path and emulating it only slows the mix down.

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
gains about 1.4 decimal digits at `high` and 3.1 digits at `mid`. The `mid` gain is the more
robust of the two, since there the guard bits absorb a per-operation error that grows with the
length of the chain; the `high` gain is against a baseline carrying the summation error described
above, and disappears when that error is taken out of the comparison.

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
knowing this table — the lower levels are opt-in. Note also that stepping down costs special
values as well as precision: in the packed form `mid` and `low` handle the normal range only,
and do not carry the subnormal, NaN and infinity handling that `high` does. The unpacked form
is better behaved here, its shared pack and unpack routines being full-range at every accuracy
level, so stepping down there lowers the precision of the core without discarding the range
handling at the boundary.

What stepping down does **not** cost is exponent range, which is what makes `low` more useful
than its precision alone suggests. The format is still binary64, so a `low` value reaches to
about ±10^308 where a `float` stops at ±10^38, while the significand it delivers is in the
neighbourhood of `float`'s. It is therefore the level for computations that need the *range*
rather than the precision — a power series whose terms are well served by 24 bits of
significand, but whose intermediate products leave `float`'s exponent behind long before the
sum converges — and it is the cheapest arithmetic the component offers.

The four rounding modes `rn` (nearest), `rz` (toward zero), `ru` (toward +∞) and `rd`
(toward −∞) are available at every accuracy level through the intrinsic spellings in the
operations table below, but **for the packed form only** — the unpacked form offers `_rn` and
nothing else. The operators themselves round to nearest in both, as they do for `double`.

Using the types
---------------

```c++
#include <cuda/fpemu>
```

The CCCL FP component lives in the `cuda` namespace. The examples qualify its names rather
than using a using-directive.

Type names carry the `cuda::` prefix; the standard-named math functions are left
**unqualified** and found by argument-dependent lookup, so a body of `double` code keeps its
call sites when the type underneath is swapped:

```c++
cuda::fp64emu x = 2.0;      // implicit, as to double
auto r = sqrt(x);            // unqualified: ADL finds the fpemu overload
```

### Construction and conversion

| Conversion | Packed `fp64emu` | Unpacked `fp64emu_unpacked` |
|---|---|---|
| `double` → | implicit | cast required |
| `float` → | implicit | cast required |
| any standard integer → | implicit, at every width | cast required |
| → `double` | implicit | implicit |
| → `float` | cast required | cast required |
| → a standard integer | cast required, truncating | cast required, truncating |

The packed form is deliberately as permissive as `double` itself, which is what makes it a
drop-in — including the integer constructors, which are implicit at 64 bits as well as 32, on
the same principle that makes `long` to `double` implicit despite its potential loss. The
unpacked form asks for the cast everywhere, since entering it is a change of representation
rather than just of type — the practical consequence in real code is that `acc += 0.5` needs to
become `acc += cuda::fp64emu_unpacked{0.5}`.

`__int128` and `__uint128` are `= delete`d in both directions, and `__float128` on the way in,
rather than being absent, so the diagnostic names the rule. Only the default constructors are
`constexpr`, so a value cannot be built at compile time.

### Operations

| Operation | C++ | Intrinsic spelling |
|---|---|---|
| Add, subtract, multiply, divide | `+`, `-`, `*`, `/` and compound forms | `__dadd_<rm>`, `__dsub_<rm>`, `__dmul_<rm>`, `__ddiv_<rm>` |
| Fused multiply-add | `fma()` | `__fma_<rm>` |
| Square root | `sqrt()` | `__dsqrt_<rm>` |
| Multiply-add | `mad()` | `__mad_rn` |
| Dot product | `dot()` | |
| Complex multiply | `cmul()` | |

The CUDA-style intrinsic spellings take these types as operands and deduce the accuracy level
from them, so existing intrinsic call sites port across unchanged. The `<rm>` suffix is one of
`rn`, `rz`, `ru`, `rd`, which is how a rounding mode other than nearest is selected; `mad` is
the exception, `__mad_rn` being its only spelling.

All six comparisons follow IEEE-754 semantics, for both representations. Mixed-type
arithmetic works directly, so an `fpemu` value combines with a built-in scalar without a cast
on the scalar side. Note that `cmul()` returns `void` and writes its real and imaginary
results through two reference parameters, so it is not an expression-style call.

There are **no transcendental math functions** for `fpemu` — no `exp`, `log` or
trigonometry. Arithmetic, `fma` and `sqrt` are the surface. This is the main functional
difference from the `fpmp` types, whose header carries a full math API.

`fpemu` objects may be declared `volatile`, for the legacy CUDA pattern of holding
shared-memory scalars in volatile variables. As for `fpmp2`, support is limited to storage —
loads, stores and copies, with a bit-preserving round-trip and trivial copyability retained.
Arithmetic and comparison take `const fpemu&` and will not bind a volatile lvalue, so compute
on a non-volatile copy and store the result back.

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

Optimization guide
------------------

`fpemu` pays for `double`'s semantics in integer and FP32 instructions, so it wins only where
FP64 is rationed, and how much it wins depends on keeping the pack and unpack out of the loop
and on matching the accuracy level to the computation. The [FP optimization
guide](README_fp_optimization_guide.md#using-fpemu-efficiently) collects that advice, and its
[Choosing the type](README_fp_optimization_guide.md#choosing-the-type) section places `fpemu`
against `fpmp` and native `double` — including the three-way choice around `double` on a
rationed part, where the two representations and `fp32mp2` compete.
