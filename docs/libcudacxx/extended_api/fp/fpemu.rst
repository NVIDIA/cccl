.. _libcudacxx-extended-api-fp-fpemu:

fpemu: IEEE-754 Double Precision in Software
============================================

.. toctree::
   :hidden:
   :maxdepth: 1

   Examples <fpemu_example>

``fpemu`` provides IEEE-754 double-precision arithmetic built from 32-bit integer and ``float``
operations rather than from native FP64 instructions. It is for hardware where FP64 throughput is
rationed: the emulation runs on the units the GPU has in abundance, so a ``double`` workload can be
faster in software than in the hardware path it would otherwise take.

At its default accuracy the packed form reproduces ``double`` bit for bit. The trade is therefore
throughput against the FP64 pipeline, not accuracy against ``double``. Accuracy enters only if you
step down a level deliberately, or step up to the unpacked representation.

On parts that ration FP64 at a small fraction of FP32 — the three measured
:ref:`below <libcudacxx-extended-api-fp-fpemu-performance>` do — the packed form runs at parity
with hardware ``double`` while producing its exact results, and the unpacked form runs about 1.5×
hardware ``double`` at the same accuracy. **None of this applies where FP64 already runs at full
rate**: on a part whose FP64 pipe is half of FP32, hardware ``double`` is the fast path and
emulating it only slows the mix down. That is the first thing to establish about your target.

.. _libcudacxx-extended-api-fp-fpemu-representations:

The two representations
-----------------------

.. list-table::
   :widths: 24 34 12 30
   :header-rows: 1

   * - **Type**
     - **Stored as**
     - **Size**
     - **Construction**

   * - ``fp64emu``
     - The 64-bit IEEE bit pattern, exactly as a ``double`` is stored
     - 8 bytes
     - Implicit from the built-in arithmetic types

   * - ``fp64emu_unpacked``
     - Separate sign, exponent and mantissa fields
     - 16 bytes
     - Explicit — the cast is written out

The packed form is the drop-in one: the size of a ``double``, the bit pattern of a ``double``,
converts implicitly, and at the default accuracy produces identical results. Its advantage over
``double`` is purely where the work happens.

The unpacked form exists to avoid re-packing between operations, and that buys accuracy as well as
speed. Its mantissa field extends the 53-bit binary64 significand by 9 guard bits, so intermediate
computations on values that stay unpacked run on **62 significand bits** and round to the storage
format once, at the pack, rather than after every operation.

Two consequences follow, and both matter:

- A chain of unpacked operations is **not bit-identical to** ``double``, even at
  ``fpemu_accuracy::high``, whose individual operations are. The behavior is that of computing in
  x87 double-extended precision and storing at the end. Code that has to reproduce ``double``
  exactly should use the packed form.
- 16 bytes per value against 8 can cost occupancy in kernels with many live values.

.. _libcudacxx-extended-api-fp-fpemu-accuracy:

Accuracy levels
---------------

Selected per type, and available for both representations — ``fp64emu_high``, ``fp64emu_mid``,
``fp64emu_low``, and the same three names with ``_unpacked_``:

.. list-table::
   :widths: 26 26 24 24
   :header-rows: 1

   * - **Type**
     - **Level**
     - **Mantissa**
     - **Range**

   * - ``fp64emu_high``
     - ``fpemu_accuracy::high``
     - Correctly rounded
     - Full IEEE-754: infinities, NaNs, subnormals

   * - ``fp64emu_mid``
     - ``fpemu_accuracy::mid``
     - 1–2 ulp error
     - Normal range only

   * - ``fp64emu_low``
     - ``fpemu_accuracy::low``
     - Up to half the mantissa bits lost
     - Normal range only

   * - ``fp64emu``
     - ``fpemu_accuracy::def``
     - The default selector, equal to ``high``
     - Full IEEE-754

The default is the IEEE-correct level, so the type is safe to reach for without knowing this table;
the lower levels are opt-in. This is the opposite of the ``fpmp`` convention, where the default is
the middle level — worth knowing if you use both. Stepping down costs special values as well as
precision: in the packed form ``mid`` and ``low`` handle the normal range only, and do not carry
the subnormal, NaN and infinity handling that ``high`` does. The unpacked form is better behaved
here, because the pack and unpack routines it shares across every operation are full-range at
every accuracy level — denormals normalized, infinities and NaNs encoded — so stepping down there
reduces the precision the core produces without discarding the range handling at the boundary.

What stepping down does **not** cost is exponent range, and that is what makes ``low`` more useful
than its precision alone suggests. The format is still binary64, so a ``low`` value reaches to
about ±10\ :sup:`308` where a ``float`` stops at ±10\ :sup:`38`, while its arithmetic works on the
top 24 bits of the significand — ``float``'s width. That combination, ``float`` precision over
``double``'s range, is what the level is for: a power series, say, whose terms are perfectly well
served by 24 bits of significand but whose intermediate products or factorials leave ``float``'s
exponent behind long before the sum has converged. Running the series in ``fp64emu_low`` keeps those
intermediates in range at the cheapest arithmetic the component offers — up to 2.3× native
``double``, the fastest rows in the table :ref:`below
<libcudacxx-extended-api-fp-fpemu-performance>` — rather than paying for double precision that the
computation was never going to use. Its error is biased rather than scattered, however, which is
what limits it in a long accumulation; the measurements
:ref:`below <libcudacxx-extended-api-fp-fpemu-what-the-benchmark-separates>` show the effect.

The four rounding modes ``rn`` (nearest), ``rz`` (toward zero), ``ru`` (toward +∞) and ``rd``
(toward −∞) are available at every accuracy level through the intrinsic spellings listed under
:ref:`Operations <libcudacxx-extended-api-fp-fpemu-operations>` — but **for the packed form only**.
The unpacked form offers ``_rn`` and nothing else, so a computation that needs directed rounding
needs the packed representation. The operators themselves round to nearest in both, as they do for
``double``.

Using the types
---------------

.. code-block:: cuda

    #include <cuda/fpemu>

The component lives in ``cuda::experimental``, to be promoted to ``cuda::`` later. Type names carry
the namespace; the standard-named math functions are left **unqualified** and found by
argument-dependent lookup, so a body of ``double`` code keeps its call sites when the type
underneath is swapped:

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    cudax::fp64emu x = 2.0;   // implicit, as it would be to double
    auto r           = sqrt(x);   // unqualified: ADL finds the fpemu overload

Construction and conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - **Conversion**
     - **into** ``fp64emu``
     - **into** ``fp64emu_unpacked``

   * - from ``double``
     - implicit
     - **cast required**

   * - from ``float``
     - implicit
     - **cast required**

   * - from any standard integer type
     - implicit, at every width
     - **cast required**

   * - to ``double``
     - implicit
     - implicit

   * - to ``float``
     - **cast required**
     - **cast required**

   * - to a standard integer type
     - **cast required**, truncating
     - **cast required**, truncating

The packed form is deliberately as permissive as ``double`` itself, which is what makes it a
drop-in — including the integer constructors, which are implicit at 64 bits as well as 32, on the
same principle that makes ``long`` to ``double`` implicit despite its potential loss. The unpacked
form asks for the cast everywhere, since entering it is a change of representation rather than
only of type; the practical consequence is that ``acc += 0.5`` becomes
``acc += cudax::fp64emu_unpacked{0.5}``.

``__int128`` and ``__uint128`` are ``= delete``\ d in both directions, and ``__float128`` on the way
in, rather than being absent — so the diagnostic names the rule instead of failing obscurely. Note
also that only the default constructors are ``constexpr``: the converting constructors call into the
emulation, so a value cannot be built at compile time.

.. _libcudacxx-extended-api-fp-fpemu-operations:

Operations
----------

.. list-table::
   :widths: 40 34 26
   :header-rows: 1

   * - **Operation**
     - **C++**
     - **Intrinsic spelling**

   * - Add, subtract, multiply, divide
     - ``+``, ``-``, ``*``, ``/`` and the compound forms
     - ``__dadd_<rm>``, ``__dsub_<rm>``, ``__dmul_<rm>``, ``__ddiv_<rm>``

   * - Fused multiply-add
     - ``fma()``
     - ``__fma_<rm>``

   * - Square root
     - ``sqrt()``
     - ``__dsqrt_<rm>``

   * - Multiply-add
     - ``mad()``
     - ``__mad_rn``

   * - Dot product
     - ``dot()``
     -

   * - Complex multiply
     - ``cmul()``
     -

All six comparisons follow IEEE-754 semantics, for both representations. Mixed-type arithmetic
works directly, so an ``fpemu`` value combines with a built-in scalar without a cast on the scalar
side. Note that ``cmul()`` returns ``void`` and writes its real and imaginary results through two
reference parameters, so it is not an expression-style call like the others.

The CUDA-style intrinsic spellings in the third column take these types as operands and deduce the
accuracy level from them, so existing intrinsic call sites port across unchanged. Their ``<rm>``
suffix is where a rounding mode other than nearest is asked for, as described
:ref:`above <libcudacxx-extended-api-fp-fpemu-accuracy>`. ``mad`` is the exception with no such
suffix — ``__mad_rn`` is its only spelling, for both representations.

There are **no transcendental math functions** for ``fpemu`` — no ``exp``, ``log`` or
trigonometry. Arithmetic, ``fma`` and ``sqrt`` are the surface. That is the main functional
difference from the ``fpmp`` types, whose header carries a full math API.

Objects of both representations may be declared ``volatile``, for the legacy CUDA pattern of
holding shared-memory scalars in volatile variables. As for ``fpmp2``, support is limited to
storage — loads, stores and copies, with a bit-preserving round-trip and trivial copyability
retained. Arithmetic and comparison take ``const fpemu&`` and will not bind a volatile lvalue, so
compute on a non-volatile copy and store the result back.

.. _libcudacxx-extended-api-fp-fpemu-performance:

Measured performance
--------------------

The figures below come from a midpoint-rule π integration — 2\ :sup:`26` terms, five arithmetic
operations per term, no memory traffic worth speaking of — so they price arithmetic pipelines
rather than bandwidth, and every type runs identical code. "Correct digits" means correct decimal
digits of the computed integral, with every step carried out in the type being measured, the
sequential reduction of the 131072 per-thread partials included: what is reported is therefore what
a program written in that type costs end to end, not the loop in isolation. Native ``double`` is
measured in the same run, as the baseline each row is reported against, and accuracy is a property
of the arithmetic rather than of the GPU, so it is one column for all three parts:

.. seealso::
   :ref:`fpemu examples <libcudacxx-extended-api-fp-fpemu-example>` — the program behind these
   figures, which selects the type it measures from the build line.

.. list-table::
   :widths: 28 14 16 26 16
   :header-rows: 1

   * - **Type**
     - **Correct digits**
     - **RTX 6000 Ada**
     - **RTX PRO 6000 Blackwell**
     - **B300**

   * - ``double``
     - 14.8
     - 1.00×
     - 1.00×
     - 1.00×

   * - ``fp64emu``
     - 14.8
     - 1.02×
     - 1.12×
     - 0.98×

   * - ``fp64emu_unpacked``
     - 16.1
     - **1.47×**
     - **1.65×**
     - **1.46×**

   * - ``fp64emu_mid``
     - 11.3
     - 1.43×
     - 1.61×
     - 1.46×

   * - ``fp64emu_unpacked_mid``
     - 14.0
     - **1.78×**
     - **1.96×**
     - **1.81×**

   * - ``fp64emu_low``
     - 3.1
     - 1.72×
     - 1.95×
     - 1.76×

   * - ``fp64emu_unpacked_low``
     - 3.1
     - 2.12×
     - 2.30×
     - 2.05×

Two results to read off it. The packed form at the default accuracy reproduces ``double`` bit for
bit — the two rows are not merely equal to the digit shown, they return the same value — while
running at parity with it, 0.98× to 1.12× across the three, so exact FP64 semantics are available
without using the FP64 pipe at all. And the unpacked form runs about 1.5× hardware ``double`` and
comes out 1.3 digits *ahead* of it, which is the deferred rounding
:ref:`described above <libcudacxx-extended-api-fp-fpemu-representations>` showing up twice over:
the pack/unpack tax is paid once at the boundary rather than once per operation, and the guard bits
that survive between operations absorb error that ``double`` has to round away — including the
error of the long final summation, which is where most of it is.

The emulation executes on the integer and FP32 pipes and never touches an FP64 unit, which has an
interesting consequence on parts like these: the FP64 hardware sits idle while the emulation runs,
and would still deliver its own rate if asked. Splitting a reduction across both — part of the
range on native ``double``, the rest on ``fp64emu_unpacked`` — is therefore possible in principle,
and the unpacked form is the natural one for it, its pack boundary being where the handover
belongs. No such kernel has been built and measured here, so treat that as a direction rather than
as a result.

.. _libcudacxx-extended-api-fp-fpemu-what-the-benchmark-separates:

What this benchmark can and cannot separate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The accuracy column above is one integration at one term count, and the rows do not all respond
the same way as the chain of operations gets longer. Rerunning it from 2\ :sup:`22` to
2\ :sup:`26` terms, which lengthens each thread's accumulation from 32 iterations to 512:

- ``low`` is indistinguishable from ``float`` — 3.3, 2.9 and 3.1 digits at the three term counts,
  the same figures ``float`` reports, and at some of them the same value bit for bit. That is what
  the level is: its fast path reaches 24 bits by truncating each operand's significand, so once a
  computation carries its own summation rather than handing it to something wider, ``low`` is
  ``float`` arithmetic with a wider exponent, and it degrades with term count for the same reason
  ``float`` does. Packed and unpacked agree exactly here too, the guard bits having nothing left
  to protect.
- Packed ``mid`` is pinned at 11.3 digits at every term count, its error sitting at −1.7e−11
  throughout. An error that does not move as the number of operations grows is a *relative* bias:
  each term comes out short by about the same fraction, so their sum is short by that fraction
  however many there are. A rounding error scattered about zero would instead grow with the chain.
  The 11.3 above is therefore a property of the level, not of this chain length.
- Unpacked ``mid`` is likewise flat, at 14.0, because its 1–2 ulp error lands in the guard bits
  below the stored significand rather than in the value that survives. Flat is enough to win here:
  ``double`` wanders between 13.7 and 14.8 over the same sweep, so unpacked ``mid`` is level with
  it at 2\ :sup:`26` and slightly ahead beyond that.
- At the default accuracy, packed ``fp64emu`` returns the same value as ``double`` at every term
  count, while unpacked *improves* as terms are added — 14.8, 15.9, 16.1 — pulling further ahead
  of ``double`` the longer the summation gets. The guard bits are absorbing summation error that
  ``double`` has to round away at every step, and there is more of it to absorb at every term
  count.

Which is the general caution: measure the computation you have, in the way you have it. These
figures carry the reduction, so they describe a program written end to end in one type. Reduce in
something wider instead — as a program would if it kept a more accurate accumulator — and most of
what separates the rows goes with it: the summation error disappears from both sides, ``low``
parts company with ``float``, and the unpacked form comes back level with ``double`` rather than
ahead of it. Neither measurement is the truthful one in general. The one that applies is whichever
matches where your accumulator lives.
