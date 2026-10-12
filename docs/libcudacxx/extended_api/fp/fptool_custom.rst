.. _libcudacxx-extended-api-fp-fptool-custom:

fp64_custom: Any Narrower Format, Emulated on Native FP64
=========================================================

An ``fp64_custom`` value is a ``double`` that rounds to a narrower format after every operation, so
a computation runs as though the hardware had that format. The mantissa width and the exponent
width are chosen independently, which is what makes it a tool for asking questions about an
algorithm rather than a type to write one in: how much precision does this kernel actually need,
would BF16 break it, and is it the mantissa or the dynamic range that matters.

Because the value is held in a real ``double`` throughout, the emulation costs no extra storage and
needs no hardware support for the format being modelled — including formats that no hardware
implements, and none ever need.

.. seealso::
   :ref:`fptool overview <libcudacxx-extended-api-fp-fptool>` — the header shared with the
   ``fpmp2_stat`` types, and why neither is a type to ship in.

The type
--------

.. code-block:: cuda

    fp64_custom<exponent_bits, mantissa_bits>

Note the order: **exponent first**, which is the opposite of the natural guess. Both parameters
default to the native FP64 widths, so ``fp64_custom<>`` is plain ``double``. It is an alias
template, so the angle brackets are never optional — ``fp64_custom<> x;``, never
``fp64_custom x;``. The underlying ``fp_custom`` also takes the base type as a first parameter,
but ``double`` is the only one implemented, so ``fp64_custom`` is the spelling to use.

The named formats come out as:

.. list-table::
   :widths: 20 30 25 25
   :header-rows: 1

   * - **Format**
     - **Type**
     - **Exponent**
     - **Mantissa**

   * - FP64
     - ``fp64_custom<>``
     - 11
     - 52

   * - FP32
     - ``fp64_custom<8, 23>``
     - 8
     - 23

   * - TF32
     - ``fp64_custom<8, 10>``
     - 8
     - 10

   * - FP16
     - ``fp64_custom<5, 10>``
     - 5
     - 10

   * - BF16
     - ``fp64_custom<8, 7>``
     - 8
     - 7

   * - Powers of two
     - ``fp64_custom<11, 0>``
     - 11
     - 0

Both parameters are **stored field widths** that exclude the implicit leading bit, exactly as in
the IEEE-754 layouts they name. ``mantissa_bits`` therefore sits one below the significand count
that ``cuda::std::numeric_limits::digits`` reports for the same format: FP32 is
``mantissa_bits == 23`` and ``digits == 24``. Both conventions appear across this doc set because
each is fixed by its own API — a template parameter naming a field width, ``digits`` naming a
significand — so read every bit count against the name beside it.

The exponent axis accepts **2 to 11** bits. The floor is 2 because an n-bit exponent field spends
its all-ones pattern on infinity and NaN, so one bit would leave no binades at all. The mantissa
axis accepts **0 to 52**. Reducing the mantissa rounds the value — zero mantissa bits leave only
the implicit leading 1, so every result snaps to a power of two, which is the last row above.
Reducing the exponent narrows the range instead.

Sizes equal to the native ones switch the emulation off, **per axis and independently**: at
``<11, 52>`` the whole reduction is discarded at compile time and the arithmetic is plain
``double``, while ``fp64_custom<11, 23>`` compiles the exponent phase away and keeps only the
mantissa rounding. That last spelling is the one to reach for to model a reduced mantissa at full
FP64 dynamic range, and it is half of how the two axes get separated in the
:ref:`study below <libcudacxx-extended-api-fp-fptool-custom-pi>`.

When the rounding happens
-------------------------

This is the semantic to get right before reading any output, and it is not what most people
assume. Reduction is applied **on every arithmetic operation, to both operands and to the
result** — and **never on construction**:

1. reduce the input operands
2. perform the native FP64 operation
3. reduce the result

So a freshly constructed value still holds whatever ``double`` it was given:

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    cudax::fp64_custom<8, 23> x{1.0 / 3.0};
    static_cast<double>(x);          // still the full double 1/3, unreduced
    static_cast<double>(x + zero);   // now rounded to 23 mantissa bits

The consequence worth internalizing is that an explicit constructor here is **not** reporting a
loss. It marks a value entering an emulated format; the loss arrives with the first operation.
``fp64_custom<5, 10>{1e300}`` therefore reads back as exactly ``1e300``, and becomes infinity as
soon as it is used.

Comparisons are the other place this shows: they compare the stored, unreduced bit patterns
without going through the reduction, so two values that would be equal in the emulated format can
still compare unequal if neither has been through an operation yet.

Rounding, overflow and underflow
--------------------------------

Mantissa reduction always uses IEEE-754 round-to-nearest-even. This is **not selectable** — there
is no template parameter, macro or function to change it, and no truncating or biased mode. Ties
go to even, so at zero mantissa bits 3 rounds down to 2 while 6 rounds up to 8, both landing on an
even exponent.

When the exponent is reduced, values outside the new dynamic range are clamped, and this too is
unconditional:

- **Overflow** — too large for the reduced exponent — becomes ±infinity
- **Underflow** — too small — becomes ±zero

The sign is preserved in both cases, and two details matter when reading results. Underflow is a
flush to signed zero: the reduced format has **no gradual/subnormal range**, so there is no
graceful decay near the bottom. And a clamped result skips mantissa reduction entirely, having
already been replaced. Values that arrive as NaN or infinity are never clamped, so a NaN never
degrades into an infinity.

The only way to switch clamping off is to ask for the native exponent width, which compiles the
phase away.

Compile-time or run-time sizes
------------------------------

The sizes can come from the template arguments or from a variable:

.. code-block:: cuda

    using fp_dynamic = cudax::fp64_custom<cudax::fp_custom_dynamic_size,
                                          cudax::fp_custom_dynamic_size>;

``fp_custom_dynamic_size`` can be used on either axis or both, so one axis can stay fixed while the
other is swept. The trade is the usual one. With the sizes in the type, the emulation is
specialized and generated inline, and the native format costs nothing at all. With the sizes in a
variable, one binary covers a whole sweep of formats, but the optimizer no longer knows them and
each reduction pays a memory read. Sizes start at the native FP64 values, so a program that never
calls a setter behaves exactly like ``double``.

Setting the sizes
~~~~~~~~~~~~~~~~~

Both axes have a setter and a getter, on both the host and the device — twelve functions in total,
all with a defaulted template parameter, so they are called with empty angle brackets or none:

.. list-table::
   :widths: 16 28 28 28
   :header-rows: 1

   * -
     - **Host state**
     - **Device state, from host**
     - **Device state, from a kernel**

   * - set mantissa
     - ``fp_custom_set_host_mantissa_size(int)``
     - ``fp_custom_set_device_mantissa_size(int, stream_ref)``
     - ``fp_custom_set_device_mantissa_size(int)``

   * - set exponent
     - ``fp_custom_set_host_exponent_size(int)``
     - ``fp_custom_set_device_exponent_size(int, stream_ref)``
     - ``fp_custom_set_device_exponent_size(int)``

   * - get mantissa
     - ``fp_custom_get_host_mantissa_size()``
     - ``fp_custom_get_device_mantissa_size(stream_ref)``
     - ``fp_custom_get_device_mantissa_size()``

   * - get exponent
     - ``fp_custom_get_host_exponent_size()``
     - ``fp_custom_get_device_exponent_size(stream_ref)``
     - ``fp_custom_get_device_exponent_size()``

Host and device sizes are **independent state**: setting the device mantissa leaves the host one at
52. That is deliberate, so a host reference computation can run at full precision alongside a
reduced device one.

The stream-taking forms are what host code uses to drive a device sweep, and the stream does two
jobs. It orders the write, so every kernel launched after it on that stream sees the new size and
**no synchronization is needed**. And it identifies the device, the size being per-device state, so
a multi-device program sets the size once per device. The getters, by contrast, do synchronize,
because the value has to come back to the host. Those four are also the only ones that can throw,
reporting a failed copy as ``cuda::cuda_error``; the other eight are ``noexcept``.

The kernel-side setter writes from thread 0 of block 0 only, so a whole grid calling it does not
race — but nothing makes the new size visible to the rest of the grid. It is meant for a
single-block setup kernel, or for a JIT-compiled program with no host side to set it from.

Three practical notes. Out-of-range sizes are caught by an assertion, so they are a diagnostic in a
debug configuration rather than a runtime guarantee. On the device, all translation units share one
copy of the sizes only under relocatable device code; compiled as a whole program, each gets its
own copy, so a size set for one is not seen by the others. And under NVRTC only the four
device-code accessors exist, a JIT compilation having no host side.

Using the type
--------------

The header is opt-in, because the runtime sizes above are mutable variables at namespace scope
with one copy shared by every translation unit. Define ``CCCL_ENABLE_FPTOOL`` for the whole
project — see :ref:`the component page <libcudacxx-extended-api-fp-fptool>` for why it has to be
the whole project:

.. code-block:: bash

    nvcc -DCCCL_ENABLE_FPTOOL ...

.. code-block:: cuda

    #include <cuda/fptool>     // one header for the whole feature

Everything this feature names is specific to the component and carries the namespace. Unlike the
``fpmp`` and ``fpemu`` types, there is nothing here to find by argument-dependent lookup: the
setters and getters take only an ``int`` and a stream, so there is no operand of a component type
to look in.

Construction and conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Narrowing into a reduced format is explicit by default, on the same principle as the other FP
component types — the cast marks the point where a value enters an emulated format:

.. list-table::
   :widths: 34 33 33
   :header-rows: 1

   * - **Target**
     - **from** ``double`` **or an integer**
     - **from** ``float``

   * - ``fp64_custom<>``
     - implicit
     - implicit

   * - ``fp64_custom<8, 23>``
     - **explicit**
     - implicit

   * - ``fp64_custom<5, 10>``
     - **explicit**
     - **explicit**

   * - dynamic sizes
     - **explicit**
     - **explicit**

A dynamic size is always explicit, since the format is not known at compile time. Setting
``CCCL_FP_CUSTOM_EXPLICIT_CASTS=0`` makes them all implicit, which is the adoption path when a
``double`` typedef is being swapped across an existing codebase and respelling every initialization
is the obstacle. This macro is milder than its ``fpmp`` counterpart: construction does not reduce,
so an implicit conversion here drops nothing at the conversion itself — what it drops is the
annotation.

Conversion out to ``double`` is always implicit and always exact, that being the type the value is
held in. ``operator float()`` is implicit only for formats that fit in a ``float``
(``exponent_bits <= 8 && mantissa_bits <= 23``) and explicit otherwise — but note what that does
*not* do. ``float f = x;`` compiles for every format regardless, via the implicit ``double``
conversion followed by the standard ``double``-to-``float`` narrowing. What the specifier decides is
overload resolution, and for a format that fits in a ``float``, a call whose overload set holds both
``float`` and ``double`` is ambiguous and needs a cast at the call site.

``__int128``, ``__uint128`` and ``__float128`` conversions are deleted rather than absent, so the
diagnostic names the rule instead of failing obscurely.

One conversion is missing: there is none **between two** ``fp_custom`` **formats**. A value can only
enter a format from a built-in type, so ``fp64_custom<5, 10> x{y}`` does not compile for a ``y``
that is itself an ``fp_custom``, and neither does assigning one format to another. Convert through
``double``, which the value is held in and leaves exactly:

.. code-block:: cuda

    cudax::fp64_custom<5, 10> x{static_cast<double>(y)};

Where several formats are wanted for the same number, the simplest arrangement is to keep the input
in a ``double`` and build each format from it, as the example does. Note that this route carries the
value the source was *given*, not the value its format would produce, since construction stores the
number unreduced.

Operations
~~~~~~~~~~

Arithmetic ``+ - * /`` and unary negation; compound assignment ``+= -= *= /=``; pre- and
post-increment and decrement; all six comparisons. Mixed arithmetic with a plain built-in scalar
works in either order, and the result is the ``fp_custom`` type rather than the scalar, so
``x + 2.0`` and ``3 * x`` stay in the emulated format. Two *different* ``fp_custom`` instantiations
deliberately do not mix — modelling a format means committing to it.

Unary negation is a sign-bit flip and does **not** reduce, being exact in any format.

Objects may be declared ``volatile``, for the legacy shared-memory pattern, with support limited to
storage — load, store and a bit-preserving round-trip. A volatile lvalue is not an operand, so copy
into a non-volatile local first. The type stays trivially copyable, the same size and alignment as a
``double``, and ``bit_cast`` through ``double`` is the sanctioned route to the raw bits.

The math-function gap
~~~~~~~~~~~~~~~~~~~~~

Only **sqrt and fma** have reducing implementations. They are also redeclared in ``cuda::std``, so
the qualified spellings select the reducing version rather than narrowing through
``operator double()``.

Every other math function does not. There is no ``fabs``, ``exp``, ``log``, ``sin`` or ``cos`` for
``fp_custom``, so a call like ``cuda::std::fabs(x)`` resolves through the implicit
``operator double()`` and computes at **full FP64 precision, silently**. In a sensitivity study that
is a real trap: the emulated format quietly stops applying for that part of the expression. Where an
algorithm leans on transcendentals, check what is actually being reduced before trusting the answer.

.. _libcudacxx-extended-api-fp-fptool-custom-pi:

A worked sensitivity study
--------------------------

What the tool is for is best shown by using it. The integration below is the same midpoint-rule π
computation the :ref:`fpmp <libcudacxx-extended-api-fp-fpmp>` and
:ref:`fpemu <libcudacxx-extended-api-fp-fpemu>` pages measure — 2\ :sup:`26` terms, five arithmetic
operations per term — run here through one emulated format after another, with the type alias as the
only thing that changes. "Correct digits" means correct decimal digits of the computed integral.

Two things about the method, since they bound what the numbers mean, and since they are why the
``fp64_custom<11, 52>`` row below does not match the ``double`` row on those other two pages. The
per-thread accumulation is the part that runs in the emulated format; the cross-thread reduction
and the reference are ``fp64mp2``, so the figures here are about the integration loop rather than
about the summation, whereas the ``fpmp`` and ``fpemu`` measurements keep the reduction in the type
as well. And because that reduction is more accurate than the format under test, these are the
*best* results each format can give on this algorithm, not the typical ones — which is what a
sensitivity study wants, the question being where a format starts to lose the *computation*.

Start with the mantissa, holding the exponent at its native 11 bits so that nothing is clamped:

.. list-table::
   :widths: 30 20 26 24
   :header-rows: 1

   * - **Type**
     - **Models**
     - **Result**
     - **Correct digits**

   * - ``fp64_custom<11, 52>``
     - FP64
     - 3.1415926535897936
     - 16.3

   * - ``fp64_custom<11, 23>``
     - an FP32 significand
     - 3.1415926522895461
     - 9.4

   * - ``fp64_custom<11, 10>``
     - a TF32/FP16 significand
     - 3.1249996721744537
     - 2.3

   * - ``fp64_custom<11, 7>``
     - a BF16 significand
     - 2
     - none

   * - ``fp64_custom<11, 0>``
     - powers of two
     - 0.015625
     - none

A graded response, and an unforgiving one past a point: 24 significand bits buy 9 digits, 11 bits
buy 2, and 8 bits do not survive the accumulation at all.

Now the other axis, holding the mantissa at 23 bits and squeezing the exponent:

.. list-table::
   :widths: 34 32 34
   :header-rows: 1

   * - **Type**
     - **Result**
     - **Correct digits**

   * - ``fp64_custom<11, 23>``
     - 3.1415926522895461
     - 9.4

   * - ``fp64_custom<8, 23>``
     - 3.1415926522895461
     - 9.4

   * - ``fp64_custom<7, 23>``
     - 3.1415926522895461
     - 9.4

   * - ``fp64_custom<6, 23>``
     - 3.1415926522895461
     - 9.4

   * - ``fp64_custom<5, 23>``
     - NaN
     - —

Not graded at all, but a cliff. From 11 bits of exponent down to 6 the results are **byte-identical**
— the same 17 digits, not merely the same rounded figure — and at 5 bits the computation does not
produce a number. That is the whole dynamic-range question answered in one column: this algorithm
needs six bits of exponent and does not care what it is given beyond that, so every digit it loses
to a narrow format is lost to the mantissa.

The cliff is worth understanding, because it is characteristic of the exponent axis rather than an
artifact. Five exponent bits is FP16's range, whose smallest normal is about 6.1·10\ :sup:`-5` and
whose largest finite value is 65504. The step 1/2\ :sup:`26` ≈ 1.5·10\ :sup:`-8` falls under the
first and the thread index runs past the second, so with no subnormals to soften the bottom, one
operand flushes to zero, another saturates to infinity, and their product is NaN. Range does not
degrade gracefully the way precision does — it either covers the values or it does not.

Reading the two tables together answers the question the tool exists for. Modelling FP32 as a whole
(``fp64_custom<8, 23>``) gives 9.4 digits, and the mantissa sweep shows that all of the loss from
16.3 is the significand's doing, since the exponent sweep is flat across that width. An algorithm
that had come out the other way — flat under mantissa reduction and falling off under exponent
reduction — would be one to attack with a wider-range, narrower-precision format such as BF16, and
this is the measurement that tells the two cases apart.

One number not in the tables: every reduced format above ran the integration in about 1.44 ms
against native ``double``'s 1.30 ms on an RTX 6000 Ada. Reduction is cheap, but it is a cost and
never a saving. These formats are for learning what an algorithm needs, after which the answer is
implemented in whatever the hardware actually offers.

The example
-----------

`fptool_custom.cu <https://github.com/NVIDIA/cccl/blob/main/examples/cudax/fp/fptool_custom.cu>`__
is the complete program, and it covers the interface rather than a benchmark. The sizes can be
fixed in the type or held in a variable, so it runs one kernel per choice —
``compile_time_formats`` and ``run_time_formats`` — and evaluates the same expression through both,
so the two can be read against one another. The compile-time formats are ordered by descending
mantissa to line up row by row with the run-time sweep.

It demonstrates:

- the named formats, and how far each one rounds the same sum
- construction: the native format takes a ``double`` implicitly, a reduced one marks the value
  entering it
- mixed arithmetic — a reduced value combined directly with a plain ``double``
- the reduced exponent range, where a value too large becomes infinity
- sweeping the mantissa size at run time without recompiling, and reading back the size in effect
- the same source running on both host and device, which is also where the two ways of setting a
  run-time size differ: host code sets its own size directly, device code has it set from the host
  on a stream

Each step prints its inputs and its result, and nothing is checked against a reference — the point
is to show what the interface looks like in use. Two things to notice in its output. The
powers-of-two row snaps every result to a power of two, which is the clearest demonstration that the
rounding is real and not a display artifact. And the exponent-range step shows the construction
semantics above in action: the value reads back intact, and only becomes infinity once it is used.

The example sweeps the mantissa axis at run time. The exponent axis has the same setters and would
sweep the same way; it is left out to keep the output readable.
