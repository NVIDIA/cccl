FPMP — Multi-Precision Arithmetic on Pairs of Floats
====================================================

An `fpmp2` value represents a number as the unevaluated sum of two IEEE-754 floats,
`value = hi + lo`, which roughly doubles the available mantissa. The pair is stored
directly in the object and every operation is an error-free transformation on the two
limbs, so the extra precision comes from arithmetic the hardware already does fast rather
than from a wider format the hardware does not have.

That matters in two places. Below `double`, GPUs typically have far more FP32 throughput
than FP64, so a float pair can be both more precise than `float` and faster than native
`double`. Above `double`, there is usually no IEEE-754 binary128 hardware at all, and a
double pair reaches 104 mantissa bits out of FP64 operations for far less than a software
binary128 costs.

For the example set, see the [examples README](README.md).

Where this pays off
-------------------

The figures below are measured on a midpoint-rule π integration — 2^26 terms, five arithmetic
operations per term, no memory traffic worth speaking of — so they price arithmetic pipelines
rather than bandwidth, and every type runs identical code. "Correct digits" means correct
decimal digits of the computed integral.

**Below `double`, where FP64 throughput is rationed.** On L40S (Ada), RTX PRO 6000 Blackwell and
B300, native FP64 runs at roughly 1/35 of FP32. A float pair rides the units that are plentiful:

| Type | Correct digits | Speed vs native `double` |
|---|---|---|
| `float` | 6.5 | 34–37× |
| `fp32mp2`, `low` | 12.8 | **16.7–17.4×** |
| `fp32mp2`, `mid` | 13.8 | **11.7–11.9×** |
| `fp32mp2`, `high` | 14.5 | **6.9–7.0×** |
| `double` | 14.8 | 1.0× |

The `high` level lands within 0.4 digits of `double` at about seven times its throughput, and
the ratios hold across those three architectures to within a few percent, the arithmetic being
pure FP32. Plain `float` is not an alternative at 6.5 digits: with 24 mantissa bits it cannot
place the grid points past 2^22 terms, so its error stops improving however many terms are
added, which is the wall these types exist to get past.

**Above `double`, against software binary128.** Where FP64 runs at full rate (B200), a double
pair is the cheap way to buy digits beyond `double`:

| Type | Correct digits | Speed vs software `fp128` |
|---|---|---|
| `fp64mp2` | 17.2 | **13.2×** |
| software `fp128` | 17.2 | 1.0× |

The same answer to the last reported digit for a thirteenth of the cost — about 3.6 `double`
passes against roughly 47. Read that as the benchmark saturating rather than the two formats
being equivalent: 17.2 digits is this integration's own truncation floor, which the pair already
resolves past, so binary128's extra significand bits have nothing left to buy here. A
computation that genuinely needs more than 104 bits still needs binary128.

The ordering does depend on the hardware, and it reverses: on
the FP64-rationed parts above, the pair rides those scarce FP64 units while a software quad path
does not, and `fp128` comes out 2.4–3.1× *faster* there. Pair arithmetic above `double` wants
FP64 throughput underneath it.

**Retiring 80-bit CPU code.** Computations written for x87 double-extended — `long double` on
Linux/x86 — carry a 64-bit significand. `fp64mp2` carries 104 by the counting used below, so
precision survives that move with room to spare. What does not survive is exponent range, which
stays `double`'s ±10^308 rather than x87's ±10^4932. Where the 80-bit format was chosen for its
mantissa, this is a substitution; where it was chosen for its range, the code needs rescaling.

The types
---------

| Type | Representation | Mantissa | Range | Size |
|---|---|---|---|---|
| `fp32mp2` | float-float | 46 bits = 2×24 − 2 | float's, ~±10^38 | 8 bytes |
| `fp64mp2` | double-double | 104 bits = 2×53 − 2 | double's, ~±10^308 | 16 bytes |

Significand counts in this document always **include the implicit leading bit**, the
convention `cuda::std::numeric_limits::digits` uses: `float` has 24 bits, 23 stored plus the
implicit one, and `double` has 53. A non-overlapping pair guarantees `2p − 2` of them, hence
46 and 104. The two subtracted bits are what non-overlap costs — the halves must stay
disjoint, so the pair carries that many *contiguous* bits rather than the 48 or 106 that its
two component significands add up to. Read 46 as `2×24 − 2`, not as `2×23`: the arithmetic
coincides, but the reasoning does not.

The other convention — stored field widths, which exclude the implicit bit — is what the
`fp_custom` template parameters use, since they name IEEE-754 layouts; see
[README_fptool_custom.md](README_fptool_custom.md). Each doc says which one it means.

The two share one interface and are used identically; only the component type differs.
Note that a pair widens the mantissa but **not** the exponent range: `fp32mp2` carries
more precision than `float` while still overflowing where `float` overflows. Where the
range matters as much as the precision, `fp64mp2` is the one to reach for.

`cuda::std::numeric_limits` is specialized for both, so generic code can query them as it
does the built-in types. The reported characteristics follow from the pair model rather
than from IEEE-754, so `is_iec559` is `false`:

| Property | `fp32mp2` | `fp64mp2` |
|---|---|---|
| `digits` / `digits10` / `max_digits10` | 46 / 13 / 15 | 104 / 31 / 33 |
| `min_exponent` / `max_exponent` | -101 / 128 | -968 / 1024 |
| `epsilon()` | 2^-45 | 2^-103 |
| `min()` / `max()` | 2^-102 / `FLT_MAX` | 2^-969 / `DBL_MAX` |

The minimum exponent is raised relative to the component type because both halves have to
stay normal for the pair to carry its full mantissa.

Accuracy levels
---------------

Each width comes at a selectable accuracy level, which decides how much work goes into
keeping the trailing limb correct. The level is part of the type:

| Type | Level | What it does |
|---|---|---|
| `fp32mp2_low` | `fpmp2_accuracy::low` | Fast arithmetic with no renormalization; the limbs are allowed to drift into overlap |
| `fp32mp2_mid` | `fpmp2_accuracy::mid` | Dekker-based splitting and error accumulation |
| `fp32mp2_high` | `fpmp2_accuracy::high` | Thall-based splitting and error accumulation, the most accurate |
| `fp32mp2` | `fpmp2_accuracy::def` | The default selector, equal to `mid` — not to `high` |

with the same four names for `fp64mp2`. Two points are easy to miss. The default is `mid`,
so asking for `high` is a deliberate step up rather than the level you already have. And
the levels differ only in the trailing limb, so a result rounded to one `double` will often
look identical across levels — comparing them means printing `hi` and `lo` separately.

The level can also be chosen per operation instead of per type, which is the point of the
accuracy-explicit functions:

```c++
namespace cudax = cuda::experimental;

using ffloat = cudax::fp32mp2_low;              // low accuracy for the bulk of the work
ffloat a = ..., b = ...;

ffloat r = cudax::add<cudax::fpmp2_accuracy::high>(a, b);   // this one step at high
```

`add`, `sub`, `mul`, `div`, `fma` and `mad` all take the level this way. The result type is
the operand type, so nothing else in the expression changes — useful where a low-accuracy
type carries a whole kernel but one step, typically an argument reduction, needs more. These
call the underlying scalar API on the `(hi, lo)` pairs directly rather than instantiating a
second class specialization, which avoids the register pressure that mixing types on a GPU
otherwise causes.

Using the types
---------------

```c++
#include <cuda/fpmp>       // types, operators, sqrt, rsqrt, fma, mad
#include <cuda/fpmp_math>  // adds exp, log, trig, pow, ...; also pulls in <cuda/fpmp>
```

Include only `<cuda/fpmp>` where the transcendentals are not needed — the math header
costs compile time.

The CCCL FP component lives in `cuda::experimental` (to be promoted to `cuda::` later). The
examples abbreviate it rather than using a using-directive:

```c++
namespace cudax = cuda::experimental;
```

Two spellings then appear, and the split is deliberate. Type names and the component's own
functions — `renormalize`, the accuracy-selecting `add<>` — carry the `cudax::` prefix,
since they have no counterpart for `double` and so only occur in code written against the
component to begin with. The standard-named math functions are left **unqualified** and found
by argument-dependent lookup:

```c++
cudax::fp64mp2 x{2.0};
auto r = sqrt(x);                  // unqualified: ADL finds the fpmp2 overload
auto s = cudax::renormalize(x);    // component-specific: qualified
```

This is what lets an existing body of `double` code keep its call sites unchanged when the
type underneath is swapped. `sqrt(x)` beats `::sqrt(double)` for an `fpmp2` argument because
taking the operand as-is is an exact match while reaching the `double` overload would cost a
user-defined conversion.

### Construction and conversion

This is the one place where `fpmp2` deliberately does not behave like a built-in type, and
the first thing a new user is likely to hit. A conversion **into** the pair that cannot be
represented exactly is a narrowing conversion, and by default it must be written out:

```c++
cudax::fp32mp2 x = 1.2345678901234567;                       // error by default
cudax::fp32mp2 y = static_cast<cudax::fp32mp2>(1.2345678901234567);   // correct, and constexpr
```

The reason is that an implicit narrowing conversion has only one place to go — the
single-limb constructor, which sets `lo` to zero and reports nothing. A value would silently
arrive carrying `float` precision in a type that advertises 46 bits. Requiring the cast makes
the conversion visible and routes it through the accurate two-limb path.

What is exact for every value of the source type stays implicit, so ordinary code is not
disturbed:

| Source | into `fp32mp2` | into `fp64mp2` |
|---|---|---|
| `float` | implicit | implicit |
| `double` | **cast required** | implicit |
| `bool`, `char`, `short`, `int32_t`, `uint32_t` | implicit | implicit |
| `int64_t`, `uint64_t` | **cast required** | implicit |

So `fp64mp2 acc = 0;` and `fp32mp2 t = some_float;` compile as expected, while
`fp32mp2 t = some_double;` does not. Conversion **out** to `double` is widening and always
implicit. The cast is `constexpr`, so full-precision coefficient tables can be built at
compile time.

The same rule reaches the scalar accumulate path: `+=` and `-=` have an optimized overload
taking a single component, worth about six operations over a full pair addition, and it is
constrained the same way. `acc += 1.5f` on an `fp32mp2` is fine; `acc += 1.5` is not, because
the `double` would be truncated to `float` before being accumulated.

Setting `CCCL_FPMP_EXPLICIT_CASTS=0` restores the fully implicit model, which is worth
considering when `fpmp2` is being dropped into a large existing `double` codebase and the
edit churn matters more than the diagnostics. The conversion still takes the accurate
two-limb path when it is allowed through — the macro decides whether the conversion is
written out, not how precisely it is done.

### Operations

Arithmetic `+ - * /` and unary negation, compound assignment, and all six comparisons.
`sqrt`, `rsqrt`, `fma` and `mad` come with the core header. Mixed-type arithmetic works
directly, so an `fpmp2` combines with a built-in scalar without a cast on the scalar side.

`renormalize(x)` restores the invariant that the limbs do not overlap (`|lo| < ulp(hi)`).
Arithmetic at `low` accuracy skips that step, so a run of low-accuracy operations can leave
a value whose limbs have drifted into overlap; `renormalize` is how it gets repaired.

`<cuda/fpmp_math>` adds the transcendentals: `exp`, `log`, `log2`, `log10`, `log1p`, `pow`,
`cbrt`, `rcbrt`, `sin`, `cos`, `tan`, `sincos`, `asin`, `acos`, `atan`, `atan2`, `sinh`,
`cosh`, `tanh`, `erf`, `erfc`, `normcdfinv`, the rounding family, the min/max family, and
`icdf` for `fp32mp2` only.

### Math function accuracy

How much precision a math function actually delivers depends on the type. Every dedicated
`fp32mp2` implementation is pure float-float with no FP64 operations, which is the point on
FP64-throttled hardware, and `fp32mp2` is never limited by its backend: functions without a
dedicated implementation evaluate in `double` and split into the pair, and binary64 already
carries more significand than a float pair holds.

`fp64mp2` is the interesting case, because a double pair asks for more precision than
binary64 can express. Those functions need a binary128 backend to be evaluated in, and
whether one is active is decided per compilation pass rather than by an option:

| Function class | `fp32mp2` | `fp64mp2`, binary128 | `fp64mp2`, `double` |
|---|---|---|---|
| Exact operations and predicates | exact | exact | exact |
| Transcendentals with a binary128 backend | float-float | binary128 | binary64 |
| `erf`, `erfc`, `normcdf` | float-float | binary64 | binary64 |
| `atan2`, on the device | float-float | binary64 | binary64 |
| Functions with no binary128 backend | float-float | binary64 | binary64 |

`binary64` in an `fp64mp2` column means only the high limb carries information and `lo` is
zero: a correctly formed double-double holding no more than a `double` would. That is the one
case where the type promises more than the function delivers, so it is worth knowing which
column applies. A host-only compilation takes the binary128 column wherever `__float128` is
available; a CUDA translation unit takes it in the device pass only, and only on
architectures that can run `fp128`, so the two halves of one program can differ. Setting
`CCCL_FPMP_FP128_MATH_FALLBACK=1` puts both on the quad path, at the cost of a
quad-precision math dependency on hosts that need one.

The per-function measured accuracy and performance figures live in the specification
document rather than here.

The example
-----------

### fpmp.cu

Both types offer the same interface, so one example covers both, running one kernel per
width — `float_float_operations` and `double_double_operations` — over the same set of
operations with the same inputs, so the two outputs can be read side by side.

Demonstrated:

- construction from literals of several source types
- arithmetic: `+`, `-`, `*`, `/`
- mixed-type arithmetic — an `fpmp2` combined directly with a `double` literal, with an
  `int`, with the scalar on the left, and the `+=` scalar accumulate path
- `sqrt`, `rsqrt` and `fma`
- math functions, here `exp` and `sin`, from `<cuda/fpmp_math>`
- comparison operators
- the `hi`/`lo` components the value is stored as
- the accuracy levels: the same sum on the default and the `high` type, then that same
  high-accuracy addition applied to low-accuracy operands via `add<>`
- `renormalize()`, on a value whose limbs have been driven into overlap by a run of
  low-accuracy operations
- the same source running on both host and device

Each operation prints its inputs and its result. Nothing is checked against a reference:
the point is to show what the interface looks like in use.

Two sections of the output are worth reading closely, because they are the parts that do not
show up in a single rounded number. The accuracy section prints `hi` and `lo` separately,
since that is the only way the levels are distinguishable. The renormalization section
evaluates `a - big + big` in the low-accuracy type, which leaves the limbs overlapping, and
prints the value before and after `renormalize()` — the unnormalized form is visibly wrong
in `hi` alone, and the repair puts it back.

Beyond the example
------------------

Parts of the API are covered by the unit tests rather than by an example, because there is
little to show beyond the fact that the overloads exist:

- **Atomics and warp shuffles.** `atomicAdd` and `atomicSub` are overloaded for both types
  (`fp32mp2` via 64-bit `atomicCAS`, `fp64mp2` via 128-bit `atomicCAS`, which needs compute
  capability ≥ 9.0), as are `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync` and
  `__shfl_xor_sync` on sm_70 and later.
- **Volatile objects**, for the legacy CUDA pattern of holding shared-memory scalars in
  volatile variables. Support is limited to storage — loads, stores, copies and reading
  `hi()`/`lo()` — with a bit-preserving round-trip and trivial copyability retained. A
  volatile lvalue is not an operand: arithmetic takes `const fpmp2&`, so compute on a
  non-volatile copy and store the result back, exactly as the compiler does for a built-in
  `volatile double`.
- **The full math surface.** The example calls `exp` and `sin`; the rest of the list above
  is exercised by the test suite, and the accuracy of each is characterized in the
  specification document.

References
----------

The arithmetic implements these algorithms:

1. **Dekker, T. (1971)** "A floating-point technique for extending the available precision",
   *Numerische Mathematik* 18, 224–242.
   [DOI: 10.1007/BF01397083](https://doi.org/10.1007/BF01397083)
2. **Karp, A. H., & Markstein, P. (1997)** "High Precision Division and Square Root",
   *ACM TOMS* 23(4), 561–589. [DOI: 10.1145/279232.279237](https://doi.org/10.1145/279232.279237)
3. **Thall, A.** "Extended-Precision Floating-Point Numbers for GPU Computation".
   [PDF](http://andrewthall.org/papers/df64_qf128.pdf)
4. **Nagai et al. (2008)** "Fast Quadruple Precision Arithmetic Library on Parallel Computer
   SR11000/J2", *ICCS '08*.
5. **Ogita, T., Rump, S. M., & Oishi, S. (2005)** "Accurate Sum and Dot Product",
   *SIAM J. Sci. Comput.* 26(6), 1955–1988.
