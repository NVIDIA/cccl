.. _libcudacxx-extended-api-fp-fpmp:

fpmp: Arithmetic on Pairs of Floats
===================================

.. toctree::
   :hidden:
   :maxdepth: 1

   Examples <fpmp_example>
   Full specification <fpmp_spec>

An ``fpmp2`` value represents a number as the unevaluated sum of two IEEE-754 floats,
``value = hi + lo``, which roughly doubles the available mantissa. The pair is stored directly in
the object, and each operation is built from error-free transformations — primitives such as
``two_sum`` that recover the rounding error of a single hardware operation exactly — so the extra
precision comes from arithmetic the hardware already does fast rather than from a wider format the
hardware does not have. The exactness belongs to those primitives rather than to the operation
composed from them: a pair addition or multiply carries a small error of its own, and how small is
what the :ref:`accuracy levels <libcudacxx-extended-api-fp-fpmp-accuracy>` choose between.

That matters in two places. Below ``double``, GPUs typically have far more FP32 throughput than
FP64, so a float pair can be both more precise than ``float`` and faster than native ``double``.
Above ``double``, there is usually no IEEE-754 binary128 hardware at all, and a double pair
reaches 104 significand bits out of ordinary FP64 operations.

Which of the two applies is decided by one number, the rate at which the part runs FP64 against
FP32, and the :doc:`specification <fpmp_spec>` lists it for every platform measured — 1:2 on B200
against 1:32 on B300. ``fp64mp2`` spends FP64 throughput, so it is the cheap route to quad-like
precision on a part where FP64 runs close to FP32, and a costly one where FP64 is rationed: a
software binary128 is integer code and is not competing for the units ``fp64mp2`` needs.
``fp32mp2`` is the mirror image, and wins exactly where ``fp64mp2`` struggles.

On the parts :ref:`measured below <libcudacxx-extended-api-fp-fpmp-performance>`, that comes to an
``fp32mp2`` within a few digits of ``double`` at six to sixteen times its throughput, how many of
each depending on the accuracy level asked for, and an ``fp64mp2`` a couple of digits past
``double`` for roughly five times its time.

The types
---------

.. list-table::
   :widths: 18 22 25 20 15
   :header-rows: 1

   * - **Type**
     - **Representation**
     - **Significand**
     - **Range**
     - **Size**

   * - ``fp32mp2``
     - float-float
     - 46 bits = 2×24 − 2
     - ``float``'s, ~±10\ :sup:`38`
     - 8 bytes

   * - ``fp64mp2``
     - double-double
     - 104 bits = 2×53 − 2
     - ``double``'s, ~±10\ :sup:`308`
     - 16 bytes

Significand counts on this page always **include the implicit leading bit**, the convention
``cuda::std::numeric_limits::digits`` uses: ``float`` has 24 bits, 23 stored plus the implicit
one, and ``double`` has 53. A non-overlapping pair guarantees 2p − 2 of them, hence 46 and 104.
The two subtracted bits are what non-overlap costs — the halves must stay disjoint, so the pair
carries that many *contiguous* bits rather than the 48 or 106 that its two component significands
add up to. Read 46 as 2×24 − 2, not as 2×23: the arithmetic coincides, but the reasoning does not.

The other convention — stored field widths, which exclude the implicit bit — is what the
``fp_custom`` template parameters use, since they name IEEE-754 layouts. Each page says which one
it means.

The two types share one interface and are used identically; only the component type differs. Note
that a pair widens the significand but **not** the exponent range: ``fp32mp2`` carries more
precision than ``float`` while still overflowing where ``float`` overflows. Where the range
matters as much as the precision, ``fp64mp2`` is the one to reach for.

``cuda::std::numeric_limits`` is specialized for both, so generic code can query them as it does
the built-in types. The reported characteristics follow from the pair model rather than from
IEEE-754, so ``is_iec559`` is ``false``:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - **Property**
     - ``fp32mp2``
     - ``fp64mp2``

   * - ``digits`` / ``digits10`` / ``max_digits10``
     - 46 / 13 / 15
     - 104 / 31 / 33

   * - ``min_exponent`` / ``max_exponent``
     - -101 / 128
     - -968 / 1024

   * - ``epsilon()``
     - 2\ :sup:`-45`
     - 2\ :sup:`-103`

   * - ``min()`` / ``max()``
     - 2\ :sup:`-102` / ``FLT_MAX``
     - 2\ :sup:`-969` / ``DBL_MAX``

The minimum exponent is raised relative to the component type because both halves have to stay
normal for the pair to carry its full significand.

Retiring 80-bit CPU code
~~~~~~~~~~~~~~~~~~~~~~~~

Computations written for x87 double-extended — ``long double`` on Linux/x86 — carry a 64-bit
significand. ``fp64mp2`` carries 104 by the counting just given, so precision survives that move
with room to spare. What does not survive is exponent range, which stays ``double``'s
±10\ :sup:`308` rather than x87's ±10\ :sup:`4932`. Where the 80-bit format was chosen for its
mantissa, this is a substitution; where it was chosen for its range, the code needs rescaling.

.. _libcudacxx-extended-api-fp-fpmp-accuracy:

Accuracy levels
---------------

Each width comes at a selectable accuracy level, which decides how much work goes into keeping
the trailing limb correct. The level is part of the type:

.. list-table::
   :widths: 22 28 50
   :header-rows: 1

   * - **Type**
     - **Level**
     - **What it does**

   * - ``fp32mp2_low``
     - ``fpmp2_accuracy::low``
     - Fast arithmetic with no renormalization; the limbs are allowed to drift into overlap

   * - ``fp32mp2_mid``
     - ``fpmp2_accuracy::mid``
     - Dekker-based splitting and error accumulation, normalizing the result

   * - ``fp32mp2_high``
     - ``fpmp2_accuracy::high``
     - Thall-based splitting and error accumulation, also normalizing; the most accurate

   * - ``fp32mp2``
     - ``fpmp2_accuracy::def``
     - The default selector, equal to ``mid``

with the same four names for ``fp64mp2``. Three points are easy to miss. The default is ``mid``, so
asking for ``high`` is a deliberate step up rather than the level you already have. The levels
differ only in the trailing limb, so a result rounded to one ``double`` will often look identical
across levels — comparing them means printing ``hi`` and ``lo`` separately.

And the levels differ in the *state* they leave the result in, not only in its accuracy. ``mid``
and ``high`` finish every operation with a normalization step, so each result comes back with its
limbs non-overlapping — ``|lo| <= ulp(hi)/2``, which for ``fp32mp2`` is ``2^-24 * |hi|`` — the
invariant the rest of the interface is written against. ``low`` omits that step — it is most of
what makes it fast — so its results can carry overlapping limbs, and a chain of them drifts
further from the invariant as it goes. That is what the ``renormalize()`` function is for, and why
it is needed with ``low`` and not with the other two. Converting a value up to ``mid`` or ``high``
restores the invariant on its own, so what is left for the explicit call is a value that stays at
``low`` — see :ref:`changing the accuracy level <libcudacxx-extended-api-fp-fpmp-level-change>`.

.. seealso::
   :ref:`fpmp2_stat <libcudacxx-extended-api-fp-fptool-stat>` — the instrumented counterparts of
   these types, which **measure the limb gap and count overlaps** on real data. They are how to
   find out whether ``low`` is safe for a particular computation rather than assuming it.

The level can also be chosen per operation instead of per type, which is the point of the
accuracy-explicit functions:

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    using ffloat = cudax::fp32mp2_low;              // low accuracy for the bulk of the work
    ffloat a = ..., b = ...;

    ffloat r = cudax::add<cudax::fpmp2_accuracy::high>(a, b);   // this one step at high

``add``, ``sub``, ``mul``, ``div``, ``fma`` and ``mad`` all take the level this way. The result
type is the operand type, so nothing else in the expression changes — useful where a low-accuracy
type carries a whole kernel but one step, typically an argument reduction, needs more. These call
the underlying scalar API on the ``(hi, lo)`` pairs directly rather than instantiating a second
class specialization, which avoids the register pressure that mixing types on a GPU otherwise
causes.

Using the types
---------------

.. code-block:: cuda

    #include <cuda/fpmp>   // types, operators, sqrt, rsqrt, fma, mad, and the math functions

One header carries the whole interface, the transcendental math functions included.

The component lives in ``cuda::experimental`` namespace. Two spellings
then appear, and the convention below is worth following even though, for most of these functions,
either one compiles. The standard-named math functions are left **unqualified**, found by
argument-dependent lookup. The component's own functions have no counterpart for ``double``, so
they only occur in code written against the component to begin with, and naming the namespace says
so:

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    cudax::fp64mp2 x{2.0};
    auto r = sqrt(x);                  // unqualified: ADL finds the fpmp2 overload
    auto s = cudax::renormalize(x);    // component-specific; renormalize(x) also compiles

Leaving the standard names unqualified is what lets an existing body of ``double`` code keep its
call sites unchanged when the type underneath is swapped. ``sqrt(x)`` beats ``::sqrt(double)`` for
an ``fpmp2`` argument because taking the operand as-is is an exact match while reaching the
``double`` overload would cost a user-defined conversion.

The one case where the choice is not free is the accuracy-selecting ``add``, ``sub``, ``mul``,
``div`` and ``fma``, called with their template argument spelled out. Under C++17,
``add<fpmp2_accuracy::high>(a, b)`` unqualified does not compile: ordinary lookup has to find the
name for the ``<`` to parse as a template argument list rather than as less-than, and ADL does not
help. C++20 lifted that, so the unqualified form works there. Qualifying them is therefore what
keeps code portable across both.

.. _libcudacxx-extended-api-fp-fpmp-conversions:

Construction and conversion
---------------------------

This is the one place where ``fpmp2`` deliberately does not behave like a built-in type, and the
first thing a new user is likely to hit. A conversion **into** the pair that cannot be represented
exactly is a narrowing conversion, and by default it must be written out:

.. code-block:: cuda

    cudax::fp32mp2 x = 1.2345678901234567;                                 // error by default
    cudax::fp32mp2 y = static_cast<cudax::fp32mp2>(1.2345678901234567);    // correct, and constexpr

The reason is that an implicit narrowing conversion has only one place to go — the single-limb
constructor, which sets ``lo`` to zero and reports nothing. A value would silently arrive carrying
``float`` precision in a type that advertises 46 bits. Requiring the cast makes the conversion
visible and routes it through the accurate two-limb path.

What is exact for every value of the source type stays implicit, so ordinary code is not
disturbed:

.. list-table::
   :widths: 46 27 27
   :header-rows: 1

   * - **Source**
     - **into** ``fp32mp2``
     - **into** ``fp64mp2``

   * - ``float``
     - implicit
     - implicit

   * - ``double``
     - **cast required**
     - implicit

   * - ``bool``, ``char``, ``short``, ``int32_t``, ``uint32_t``
     - implicit
     - implicit

   * - ``int64_t``, ``uint64_t``
     - **cast required**
     - implicit

So ``fp64mp2 acc = 0;`` and ``fp32mp2 t = some_float;`` compile as expected, while
``fp32mp2 t = some_double;`` does not. The cast is ``constexpr``, so full-precision coefficient
tables can be built at compile time.

The same rule governs the way **out**. ``fp32mp2`` converts to ``double`` implicitly, because a
``double`` holds the whole pair; ``fp64mp2`` does not, because the conversion drops the low limb.
``float`` and the integer types need the cast from either.

Taken together that is the C++23 rule for extended floating-point types (P1467R9) — implicit
where the conversion is value-preserving, written out where it is not — arrived at here from the
IEEE-754 ``float``-to-``double`` analogy rather than from the paper. ``fp32mp2`` stands to
``fp64mp2`` as ``float`` stands to ``double``, and the conversions behave accordingly.

Conversions are one thing and mixed arithmetic another. A binary operator between an ``fpmp2``
and a built-in scalar converts the scalar **into** the pair and yields the pair, so
``some_fp32mp2 + some_double`` compiles rather than being rejected as an expression over two
types neither of which contains the other. The ``double`` is brought into ``float`` exponent
range as it goes, so one too large for that arrives as infinity and the addition yields NaN.
Mixing two accuracy levels, ``fp32mp2`` with ``fp32mp2_low``, matches no overload and falls back
on built-in ``double`` arithmetic. Where the other operand is a ``double`` whose magnitude is not
known to fit, convert deliberately — to ``fp64mp2`` if the range is needed.

Quad interchange sits outside that table. ``fp64mp2`` converts both ways with the library's
128-bit type ``__fpmp_fp128`` — ``__float128`` on x86, IEEE ``long double`` on aarch64. Both
directions are explicit, and both are deleted on ``fp32mp2``: a double-float holds about 48
significand bits, fewer than a ``double``, so ``double`` is its interchange type and a quad image
is asked for through one — ``(__fpmp_fp128) (double) x`` — which is exact for any pair meeting the
double-float contract.

On GCC, ``_Float128`` is frequently a *second* binary128 type: the same format as
``__fpmp_fp128``, but a distinct type with no implicit conversion between the two spellings. The
same explicit conversions therefore exist for ``_Float128`` wherever it is not already
``__fpmp_fp128``, so a ``static_cast<_Float128>`` of an ``fp64mp2`` compiles on aarch64 as well as
on x86. This second spelling requires GCC 13, the first release to accept the token in C++; Clang
offers no distinct type — it either rejects the token or aliases it to ``long double`` — so there
the ``__fpmp_fp128`` conversions are the only ones.

The same rule reaches the scalar accumulate path: ``+=`` and ``-=`` have an optimized overload
taking a single component, worth about six operations over a full pair addition, and it is
constrained the same way. ``acc += 1.5f`` on an ``fp32mp2`` is fine; ``acc += 1.5`` is not,
because the ``double`` would be truncated to ``float`` before being accumulated.

Setting ``CCCL_FPMP_EXPLICIT_CASTS=0`` restores the fully implicit model, which is worth
considering when ``fpmp2`` is being dropped into a large existing ``double`` codebase and the edit
churn matters more than the diagnostics. The conversion still takes the accurate two-limb path
when it is allowed through — the macro decides whether the conversion is written out, not how
precisely it is done.

.. _libcudacxx-extended-api-fp-fpmp-level-change:

Changing the accuracy level
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving a value between accuracy levels is a conversion of its own, and always written out, in
both directions — the level is part of the type, and it selects the algorithms downstream
arithmetic will use, so it does not change by accident:

.. code-block:: cuda

    cudax::fp32mp2_low fast = ...;
    cudax::fp32mp2     safe(fast);      // explicit; renormalizes on the way

Conversion **out of** ``low`` renormalizes. The reason is the state ``low`` leaves its results
in: they can carry overlapping limbs, while the ``mid`` and ``high`` algorithms are written
against the non-overlap invariant on their inputs. Handing them an overlapping pair compiles
without complaint and quietly returns a worse answer, so the conversion repairs the invariant
rather than leaving it to be remembered. That makes the mixed-accuracy pattern — ``low`` for the
bulk of the work, a higher level across a critical stretch — correct as written.

The step is one ``fast_two_sum`` and it is exact, so it changes how the number is stored and
never the number itself. Conversions in the other direction, **into** ``low``, are a plain limb
copy: the pair already satisfies the invariant, and moving into ``low`` is a deliberate step into
the fast regime. Where a pure retag is wanted, with no arithmetic at all, construct from the
limbs instead — ``fp32mp2{x.hi(), x.lo()}``.

One consequence worth knowing: because the conversion out of ``low`` is arithmetic rather than a
copy, it is not usable in a constant expression, while the level changes that copy still are.

Operations
----------

Arithmetic ``+ - * /`` and unary negation, compound assignment, and all six comparisons.
``sqrt``, ``rsqrt``, ``fma`` and ``mad`` come with the core header. Mixed-type arithmetic works
directly, so an ``fpmp2`` combines with a built-in scalar without a cast on the scalar side.

``renormalize(x)`` restores the non-overlap invariant, ``|lo| <= ulp(hi)/2``, which only ``low``
accuracy can leave broken; see :ref:`accuracy levels <libcudacxx-extended-api-fp-fpmp-accuracy>`
above. Converting up out of ``low`` applies it automatically, so the explicit call is for
repairing a value that stays at ``low`` accuracy.

The rest of the math surface arrives with the same header. Much of it is implemented directly in
float-float arithmetic for ``fp32mp2``, with no FP64 operation anywhere, which is the point on
hardware where FP64 is rationed:

- **Exponentials and logarithms** — ``exp``, ``exp2``, ``exp10``, ``expm1``, ``log``, ``log2``,
  ``log10``, ``log1p``
- **Powers and roots** — ``pow``, ``cbrt``, ``rcbrt``, alongside the ``sqrt`` and ``rsqrt``
  already named above
- **Trigonometric** — ``sin``, ``cos``, ``tan``, ``sincos``, ``asin``, ``acos``, ``atan``,
  ``atan2``, and the pi-scaled ``sinpi``, ``cospi``, ``sincospi``, which split the argument over
  the limb pair rather than reducing it against a radian multiple of pi
- **Hyperbolic** — ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``
- **Error functions** — ``erf``, ``erfc``
- **Rounding, magnitude and scaling** — ``ceil``, ``floor``, ``trunc``, ``round``, ``fabs``,
  ``ldexp``, ``scalbn``
- **Minimum, maximum and remainder** — ``fmin``, ``fmax``, ``min``, ``max``, ``fmod``,
  ``remainder``
- **Probability** — ``normcdfinv``, and ``icdf`` for ``fp32mp2`` only, which turns a 32- or 64-bit
  uniform integer into a Gaussian variate. Taking an integer rather than a pair, it is the one
  function here that ADL cannot find, so it has to be written ``cudax::icdf(bits)``, and its
  accuracy level is a defaulted template parameter rather than a deduced one
- **Special** — ``boys_f0``, the zeroth-order Boys function

The remaining names are present with the same signatures, but implemented generically rather than
through a dedicated float-float path: ``erfinv``, ``erfcinv``, ``erfcx``, ``normcdf``, ``tgamma``,
``lgamma``, the Bessel family ``j0``, ``j1``, ``jn``, ``y0``, ``y1``, ``yn``, ``cyl_bessel_i0``
and ``cyl_bessel_i1``, the distances ``hypot`` and ``rhypot``, the vector norms ``norm3d``,
``norm4d``, ``rnorm3d`` and ``rnorm4d``, ``rint``, ``nearbyint``, ``lrint``, ``lround``,
``llrint``, ``llround``, ``copysign``, ``scalbln``, ``frexp``, ``modf``, ``logb``, ``ilogb``,
``nextafter``, ``fdim``, ``remquo``, and the classification predicates. For the exact ones among
these — the rounding, sign and decomposition family — that costs nothing, since an exact operation
is exact however it is written; for the transcendentals it means evaluation through the backend
covered in the next section.

The predicates come in two spellings. ``fpmp_isfinite``, ``fpmp_isinf``, ``fpmp_isnan`` and
``fpmp_signbit`` are always available, while the standard names ``isfinite``, ``isinf``, ``isnan``
and ``signbit`` are exposed only where the platform has not already claimed them as macros, so the
prefixed forms are the portable choice.

Beyond the arithmetic, both types overload ``atomicAdd`` and ``atomicSub`` (``fp32mp2`` through
64-bit ``atomicCAS``, ``fp64mp2`` through 128-bit ``atomicCAS``, which needs compute capability
9.0 or later), and the warp shuffles ``__shfl_sync``, ``__shfl_up_sync``, ``__shfl_down_sync`` and
``__shfl_xor_sync`` on sm_70 and later. Volatile objects are supported as storage only — loads,
stores, copies and reading ``hi()``/``lo()``, with a bit-preserving round-trip and trivial
copyability retained. A volatile lvalue is not an operand: arithmetic takes ``const fpmp2&``, so
compute on a non-volatile copy and store the result back, exactly as the compiler does for a
built-in ``volatile double``.

Math function accuracy
----------------------

How much precision a math function actually delivers depends on the type. Every dedicated
``fp32mp2`` implementation is pure float-float with no FP64 operations, which is the point on
FP64-throttled hardware, and ``fp32mp2`` is never limited by its backend: functions without a
dedicated implementation evaluate in ``double`` and split into the pair, and binary64 already
carries more significand than a float pair holds.

``fp64mp2`` is the different case, because a double pair asks for more precision than binary64
can express. Those functions need a binary128 backend to be evaluated in, and whether one is
active is decided per compilation pass rather than by an option:

.. list-table::
   :widths: 34 22 22 22
   :header-rows: 1

   * - **Function class**
     - ``fp32mp2``
     - ``fp64mp2``\ **, binary128**
     - ``fp64mp2``\ **,** ``double``

   * - Exact operations and predicates
     - exact
     - exact
     - exact

   * - Transcendentals with a binary128 backend
     - float-float
     - binary128
     - binary64

   * - ``erf``, ``erfc``, ``normcdf``
     - float-float
     - binary64
     - binary64

   * - ``atan2``, on the device
     - float-float
     - binary64
     - binary64

   * - Functions with no binary128 backend
     - float-float
     - binary64
     - binary64

``binary64`` in an ``fp64mp2`` column means only the high limb carries information and ``lo`` is
zero: a correctly formed double-double holding no more than a ``double`` would. That is the one
case where the type promises more than the function delivers, so it is worth knowing which column
applies. A host-only compilation takes the binary128 column wherever ``__float128`` is available;
a CUDA translation unit takes it in the device pass only, and only on architectures that can run
``fp128``, so the two halves of one program can differ. Setting
``CCCL_FPMP_FP128_MATH_FALLBACK=1`` puts both on the quad path, at the cost of a quad-precision
math dependency on hosts that need one.

.. seealso::
   :doc:`Full fpmp specification <fpmp_spec>` — the measured accuracy, special-value behavior and
   performance of every function, per type and accuracy level, which this page does not repeat.

.. _libcudacxx-extended-api-fp-fpmp-performance:

Measured performance
--------------------

The figures below come from a midpoint-rule π integration — 2\ :sup:`26` terms, five arithmetic
operations per term, no memory traffic worth speaking of — so they price arithmetic pipelines
rather than bandwidth, and every type runs identical code. "Correct digits" means correct decimal
digits of the computed integral, with every step carried out in the type being measured, the
sequential reduction of the 131072 per-thread partials included: what is reported is therefore what
a program written in that type costs end to end, not the loop in isolation. Native ``double`` is
measured in the same run, as the baseline each row is reported against.

.. seealso::
   :ref:`fpmp examples <libcudacxx-extended-api-fp-fpmp-example>` — the program behind these
   figures, which selects the type it measures from the build line.

**Below** ``double``\ **, where FP64 throughput is rationed.** On these parts native FP64 runs at a
small fraction of FP32, and a float pair rides the units that are plentiful. Accuracy is a property
of the arithmetic rather than of the GPU — the three parts below agree on it to the last digit
reported — so it is one column for all of them:

.. list-table::
   :widths: 22 16 21 21 20
   :header-rows: 1

   * - **Type**
     - **Correct digits**
     - **RTX 6000 Ada**
     - **RTX PRO 6000 Blackwell**
     - **B300**

   * - ``float``
     - 3.1
     - 28.2×
     - 25.4×
     - 32.3×

   * - ``fp32mp2``, ``low``
     - 9.5
     - **15.8×**
     - **11.5×**
     - **16.0×**

   * - ``fp32mp2``, ``mid``
     - 12.0
     - **9.9×**
     - **8.6×**
     - **10.8×**

   * - ``fp32mp2``, ``high``
     - 12.7
     - **6.1×**
     - **5.5×**
     - **6.7×**

   * - ``double``
     - 14.8
     - 1.0×
     - 1.0×
     - 1.0×

Each GPU column is speed relative to native ``double`` on that part, which is why the ``double``
row reads 1.0× in all of them. A float pair lands within three digits of ``double`` at several
times its throughput, and the three levels do separate here — but they separate very unevenly, and
the shape of that is worth reading rather than the numbers alone.

The large step is from ``low`` to ``mid``: 2.5 digits, for 1.6× the time. It is not cancellation
that does it, since every term of this integrand is positive. It is that ``low`` is the one level
that omits the closing renormalization, so its two limbs are free to overlap, and the overlap
compounds along a chain of 131072 additions until part of the trailing limb is no longer carrying
information. ``mid`` and ``high`` both restore the invariant on every operation, and neither drifts.

The step from ``mid`` to ``high`` is small by comparison — 0.7 digits, for another 1.6× — and that
is what to expect on a sum of same-signed values. Both levels normalize; what ``high`` adds is the
recovery of the rounding error committed when the two trailing limbs are added, and that error only
grows into the result when something amplifies it. Cancellation is the usual amplifier and there is
none here, so ``high`` collects a fraction of a digit. On a computation that does cancel, the same
0.7 digits can be several. Which is the reason to measure a given computation rather than to assume
from a table, and the reason the program linked above takes its type from the build line.

Plain ``float`` is not an alternative, and not merely because it is 11.7 digits behind here. Its
accuracy gets *worse* as terms are added — 5.2 digits at 2\ :sup:`20` against 2.8 at
2\ :sup:`28` — so the extra work buys less answer rather than more. With 24 significand bits it can
neither place the grid points at that spacing nor carry the sum across that many additions.
``double`` holds near 14 digits over the same range. That divergence is the wall these types exist
to get past.

**Above** ``double``\ **.** The same program run with ``fp64mp2`` reaches 17.2 digits where
``double`` reaches 14.8, and takes roughly five times ``double``'s time on all three parts above: a
pair operation is a run of operations on the limbs, and on these parts those are the rationed FP64
ones.
The digits past ``double`` are therefore bought at a real price, unlike the ones below it, and how
bearable that price is follows from the FP64 throughput underneath.

References
----------

The arithmetic implements these algorithms:

1. **Dekker, T. (1971)** "A floating-point technique for extending the available precision",
   *Numerische Mathematik* 18, 224–242.
   `DOI: 10.1007/BF01397083 <https://doi.org/10.1007/BF01397083>`__
2. **Karp, A. H., & Markstein, P. (1997)** "High Precision Division and Square Root",
   *ACM TOMS* 23(4), 561–589.
   `DOI: 10.1145/279232.279237 <https://doi.org/10.1145/279232.279237>`__
3. **Thall, A.** "Extended-Precision Floating-Point Numbers for GPU Computation".
   `PDF <http://andrewthall.org/papers/df64_qf128.pdf>`__
4. **Nagai et al. (2008)** "Fast Quadruple Precision Arithmetic Library on Parallel Computer
   SR11000/J2", *ICCS '08*.
5. **Ogita, T., Rump, S. M., & Oishi, S. (2005)** "Accurate Sum and Dot Product",
   *SIAM J. Sci. Comput.* 26(6), 1955–1988.
