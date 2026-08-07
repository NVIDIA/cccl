.. _libcudacxx-standard-api-time:

Time Library
=======================

See the documentation of the standard header `\<chrono\> <https://en.cppreference.com/w/cpp/header/chrono>`_

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - ``<cuda/std/chrono>``
     - Times, dates, and clocks
     - libcu++ 1.1.0 / CCCL 2.0.0 / CUDA 12.3

Implementation-Defined Behavior
-------------------------------

`std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_
is a clock that track real-world time. In the C++ Standard, it is
unspecified whether or not this clock is monotonically increasing. In
our implementation, it is not.

To implement
`std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_,
we use:

-  `GetSystemTimePreciseAsFileTime <https://docs.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimepreciseasfiletime>`_ and
   `GetSystemTimeAsFileTime <https://docs.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemtimeasfiletime>`_
   for host code on Windows.
-  `clock_gettime(CLOCK_REALTIME, ...) <https://man7.org/linux/man-pages/man3/clock_gettime.3.html>`_ and `gettimeofday <https://man7.org/linux/man-pages/man2/gettimeofday.2.html>`_
   for host code on Linux, Android, and QNX.
-  `PTX's %globaltimer <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>`_ for device code.

`PTX's %globaltimer <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>`_
is a system clock which also happens to be monotonically increasing on today's NVIDIA GPUs
(e.g. it cannot be updated and is not changed when the host system clock changes).
However, this is not necessarily the case with respect to host threads, where updates of the system clock may occur during the execution of the program.

`PTX's %globaltimer <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>`_
is initialized from the host system clock upon device attach; that may be at program start, but it could be earlier (for example, due to CUDA persistence mode).
Since `PTX's %globaltimer <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>`_ is a system clock,
it counts real-world time, and thus it has the same tick rate as the host system clock.

There is potential for logical inconsistencies between the time that host threads and device threads observe from our
`std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_.
However, this is perfectly fine; it is an inherent property of system clocks. In fact, it is not even guaranteed that a system clock remain
consistent between different host threads, or even within the same host thread. This can occur, for example, due to Daylights Savings Time or a
time zone change.

The requirements for `Clock <https://eel.is/c++draft/time.clock.req>`_ state:

   ``C1`` denotes a clock type. ``t1`` and ``t2`` are values returned by
   ``C1::now()`` where the call returning ``t1`` `happens before <http://eel.is/c++draft/intro.multithread#def:happens_before>`_
   the call returning ``t2`` and both of these calls occur before
   ``C1::time_point::max()``.

   ``C1::is_steady`` is ``true`` if ``t1 <= t2`` is always true and the
   time between clock ticks is constant, otherwise ``false``.

The property is true for our `std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_
within device code, but it is not true for all threads. Therefore, in the NVIDIA C++ Standard Library today,
the value of the ``is_steady`` member of
`std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_
is ``false``.

`std::chrono::high_resolution_clock <https://en.cppreference.com/w/cpp/chrono/high_resolution_clock>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `std::chrono::high_resolution_clock specification <http://eel.is/c++draft/time.clock.hires>`_ states:

   Objects of ``class high_resolution_clock`` represent clocks with the shortest tick period.
   ``high_resolution_clock`` may be a synonym for ``system_clock`` or ``steady_clock``.

In the NVIDIA C++ Standard Library, `std::chrono::high_resolution_clock <https://en.cppreference.com/w/cpp/chrono/high_resolution_clock>`_
is an alias for `std::chrono::system_clock <https://en.cppreference.com/w/cpp/chrono/system_clock>`_.
This means that it counts real-world time and that ``is_steady`` is false for our
`std::chrono::high_resolution_clock <https://en.cppreference.com/w/cpp/chrono/high_resolution_clock>`_.

While our `std::chrono::high_resolution_clock <https://en.cppreference.com/w/cpp/chrono/high_resolution_clock>`_
is not heterogeneously steady, it is steady within device code, so it is suitable for performance measurement within device code.

Omissions
---------

The following facilities in section `time.syn <https://eel.is/c++draft/time.syn>`_ of ISO/IEC IS 14882 (the C++ Standard)
are not available in the NVIDIA C++ Standard Library today:

-  `std::chrono::steady_clock <https://en.cppreference.com/w/cpp/chrono/steady_clock>`_
   - a monotonically increasing clock.
-  `std::chrono::duration I/O operators <https://eel.is/c++draft/time.duration.io>`_.

`std::chrono::steady_clock <https://en.cppreference.com/w/cpp/chrono/steady_clock>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`std::chrono::steady_clock <https://en.cppreference.com/w/cpp/chrono/steady_clock>`_
is, by definition, a monotonically increasing clock (e.g. ``is_steady`` is ``true``). We do not currently have a heterogeneous steady clock.
While we have a monotonically increasing clock in host code, and our system clock
(`PTX's %globaltimer <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>`_)
is monotonically increasing in device code, it is not guaranteed that the host clocks and the device clocks are monotonically increasing with
respect to each other, due to how `PTX's %globaltimer <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>`_
is initialized. Additionally, ``%globaltime`` and the host steady clock may tick at different rates.

It may be technically possible to synchronize the clocks and to compute and adjust for the difference in tick rates. However, it would be
challenging to do so, and may introduce substantial overhead in the initialization and access of the heterogeneous clock.

As such, today we do not provide `std::chrono::steady_clock <https://en.cppreference.com/w/cpp/chrono/steady_clock>`_,
as we cannot easily provide an efficient implementation that is truly heterogeneous and conforms to the specification.

`std::chrono::duration I/O Operators <https://eel.is/c++draft/time.duration.io>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementing a heterogeneous C++ I/O streams library involves many challenges that we cannot overcome today.
