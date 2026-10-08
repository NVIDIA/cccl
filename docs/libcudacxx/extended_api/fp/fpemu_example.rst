.. _libcudacxx-extended-api-fp-fpemu-example:

fpemu: Examples
===============

Two programs, with different purposes. The first is the example that ships with CCCL, which shows
what each operation does and is the place to start. The second is the benchmark behind the figures
on the :ref:`fpemu page <libcudacxx-extended-api-fp-fpemu>`, which shows what the emulation costs
against the hardware it is replacing.

The example that ships with CCCL
--------------------------------

`examples/cudax/fp/fpemu.cu <https://github.com/NVIDIA/cccl/blob/main/examples/cudax/fp/fpemu.cu>`__
covers both representations with one kernel each — ``fpemu_packed_operations`` and
``fpemu_unpacked_operations`` — over the same operations with the same inputs, so the two read
almost identically and the differences stand out: the unpacked type needs a written-out cast
wherever the packed one converts implicitly. Construction, arithmetic including the mixed-type and
``+=`` scalar paths, ``sqrt`` and ``fma``, the comparisons, the accuracy levels side by side, and
conversion between the two representations are all shown, each on the host and in a kernel, with
both results printed.

Its closing section is the one result that cannot be seen in a single operation. It adds 1000
terms of ``1e-17`` onto ``1.0``. One ulp of ``1.0`` is about ``2.22e-16``, so each term is some
twenty times too small to move a ``double`` at all: ``double`` and packed ``fp64emu`` both never
leave ``1.0``, while the unpacked form's guard bits accumulate the terms and reach the exact
answer. That is the deferred rounding of the unpacked representation, in the smallest case that
makes it visible.

It builds with the other CCCL examples, and the
`sibling programs <https://github.com/NVIDIA/cccl/tree/main/examples/cudax/fp>`__ do the same for
``fpmp`` and the ``fptool`` types. Those examples are also where the CCCL runtime API usage worth
copying lives — device enumeration that degrades gracefully when there is no GPU, buffers that
release themselves, and a launch configuration built with ``cuda::make_config``.

Measuring against hardware ``double``
-------------------------------------

The benchmark is the same program as on the :ref:`fpmp examples
<libcudacxx-extended-api-fp-fpmp-example>` page, which computes

.. math::

    \pi = 4 \int_0^1 \frac{dx}{1 + x^2}

by the midpoint rule, once in native ``double`` and once in a type of your choosing, and reports
for each the value, how many decimal digits of it are correct, and how long the kernel took. It is
deliberately a poor numerical method and a good benchmark: five arithmetic operations per term with
no memory traffic worth speaking of, so what it prices is arithmetic pipelines rather than
bandwidth, and both runs execute identical code. That page carries the full listing and the
measurement details — the warm-up launch, the reduction carried out in the type under test rather
than in something wider, and the double-double reference.

Nothing in it is specific to one sub-component. The kernel is written against a type parameter,
and the conversions are written out because a narrowing conversion into these types has to be
explicit — which is exactly what the unpacked representation requires everywhere:

.. code-block:: cuda

    template <class T>
    __global__ void pi_kernel(long long n, T* partials)
    {
      const int tid      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int nthreads = static_cast<int>(gridDim.x * blockDim.x);

      const T h    = T(1.0 / static_cast<double>(n));
      const T four = T(4.0);
      const T one  = T(1.0);

      T x        = (T(static_cast<double>(tid)) + T(0.5)) * h;
      const T dx = T(static_cast<double>(nthreads)) * h;

      T acc = T(0.0);
      for (long long i = tid; i < n; i += nthreads)
      {
        acc += four / (one + x * x);
        x += dx;
      }

      partials[tid] = acc;
    }

The type and the header that declares it come from the build line, so measuring an ``fpemu`` type
is a rebuild rather than an edit. All three include roots have to come from one CCCL checkout,
because the toolkit ships its own copies of CUB and Thrust and their version checks reject being
paired with a libcudacxx from somewhere else:

.. code-block:: bash

    nvcc -std=c++20 -arch=native \
      -DPI_FP_T=cudax::fp64emu_unpacked -DPI_FP_HEADER='<cuda/fpemu>' \
      -I <cccl>/libcudacxx/include -I <cccl>/cub -I <cccl>/thrust pi.cu -o pi

On an RTX 6000 Ada, where FP64 runs at a fraction of FP32:

.. code-block:: text

    pi by the midpoint rule, 67108864 terms, 131072 threads
    on NVIDIA RTX 6000 Ada Generation, sm_89

    type                           value                    digits   time (ms)
    double                         3.1415926535897984        14.78       1.297
    cudax::fp64emu_unpacked        3.1415926535897936        16.10       0.881

    cudax::fp64emu_unpacked is 1.47x the speed of native double, for +1.3 digits

Software double precision, beating the hardware double precision on the same part on both axes —
and the accuracy it gains is the summation error that its guard bits absorb and ``double`` has to
round away. Rebuilding with ``-DPI_FP_T=cudax::fp64emu`` gives the packed form, which reproduces
``double`` bit for bit at about parity on time, and the ``_mid`` and ``_low`` names walk the
accuracy levels. The :ref:`fpemu page <libcudacxx-extended-api-fp-fpemu>` collects those figures
across the parts measured.

Two things to expect when running it. Absolute times move with clocks and with what else is on the
GPU, so the ratio is the stable quantity, not the milliseconds. And ``-DPI_LOG2_TERMS=nn`` is worth
a look here in particular: as terms are added the unpacked form keeps gaining on ``double``, which
stalls near 14 digits, while packed ``mid`` sits unmoved at 11.3 and ``low`` tracks plain ``float``
down — for the reasons given
:ref:`on the fpemu page <libcudacxx-extended-api-fp-fpemu-what-the-benchmark-separates>`.
