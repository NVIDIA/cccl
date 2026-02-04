.. _libcudacxx-standard-api-numerics-random:

``<cuda/std/random>``
=====================

Provided functionalities
------------------------

Random number engines:

- `std::minstd_rand0 <https://en.cppreference.com/w/cpp/numeric/random/minstd_rand0>`_
- `std::minstd_rand <https://en.cppreference.com/w/cpp/numeric/random/minstd_rand>`_
- C++26 `std::philox4x32 <https://en.cppreference.com/w/cpp/numeric/random/philox_engine.html>`_ - available from C++17 onwards
- C++26 `std::philox4x64 <https://en.cppreference.com/w/cpp/numeric/random/philox_engine.html>`_ - available from C++17 onwards

.. note::

    ``cuda::pcg64`` is provided in the non-standard ``<cuda/random>`` header. See
    :ref:`cuda::pcg64 <libcudacxx-extended-api-random-pcg64>`.

Random number distributions:

- `std::bernoulli_distribution <https://en.cppreference.com/w/cpp/numeric/random/bernoulli_distribution>`_
- `std::binomial_distribution <https://en.cppreference.com/w/cpp/numeric/random/binomial_distribution>`_
- `std::cauchy_distribution <https://en.cppreference.com/w/cpp/numeric/random/cauchy_distribution>`_
- `std::chi_squared_distribution <https://en.cppreference.com/w/cpp/numeric/random/chi_squared_distribution>`_
- `std::exponential_distribution <https://en.cppreference.com/w/cpp/numeric/random/exponential_distribution>`_
- `std::extreme_value_distribution <https://en.cppreference.com/w/cpp/numeric/random/extreme_value_distribution>`_
- `std::fisher_f_distribution <https://en.cppreference.com/w/cpp/numeric/random/fisher_f_distribution>`_
- `std::gamma_distribution <https://en.cppreference.com/w/cpp/numeric/random/gamma_distribution>`_
- `std::geometric_distribution <https://en.cppreference.com/w/cpp/numeric/random/geometric_distribution>`_
- `std::lognormal_distribution <https://en.cppreference.com/w/cpp/numeric/random/lognormal_distribution>`_
- `std::negative_binomial_distribution <https://en.cppreference.com/w/cpp/numeric/random/negative_binomial_distribution>`_
- `std::normal_distribution <https://en.cppreference.com/w/cpp/numeric/random/normal_distribution>`_
- `std::poisson_distribution <https://en.cppreference.com/w/cpp/numeric/random/poisson_distribution>`_
- `std::student_t_distribution <https://en.cppreference.com/w/cpp/numeric/random/student_t_distribution>`_
- `std::uniform_int_distribution <https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution>`_
- `std::uniform_real_distribution <https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution>`_
- `std::weibull_distribution <https://en.cppreference.com/w/cpp/numeric/random/weibull_distribution>`_

Utilities:

- `std::seed_seq <https://en.cppreference.com/w/cpp/numeric/random/seed_seq>`_
- `std::generate_canonical <https://en.cppreference.com/w/cpp/numeric/random/generate_canonical>`_


.. note::

    ``cuda::std::seed_seq`` should be used exclusively on host or exclusively on device. Do not share the same
    ``seed_seq`` instance between host and device code.


The following engines or distributions are not implemented as they are not convenient or practical to implement in CUDA device code, either due to dynamic memory allocations or large state sizes.

Not supported
-------------
- `std::random_device <https://en.cppreference.com/w/cpp/numeric/random/random_device>`_
- `std::mersenne_twister_engine <https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine>`_
  (`std::mt19937 <https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine>`_,
  `std::mt19937_64 <https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine>`_)
- `std::subtract_with_carry_engine <https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html>`_
  (`std::ranlux24_base <https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html>`_,
  `std::ranlux48_base <https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html>`_,
  `std::ranlux24 <https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html>`_,
  `std::ranlux48 <https://en.cppreference.com/w/cpp/numeric/random/subtract_with_carry_engine.html>`_)
- `std::discard_block_engine <https://en.cppreference.com/w/cpp/numeric/random/discard_block_engine.html>`_
- `std::independent_bits_engine <https://en.cppreference.com/w/cpp/numeric/random/independent_bits_engine.html>`_
- `std::shuffle_order_engine <https://en.cppreference.com/w/cpp/numeric/random/shuffle_order_engine>`_
  (`std::knuth_b <https://en.cppreference.com/w/cpp/numeric/random/shuffle_order_engine>`_)
- `std::discrete_distribution <https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution.html>`_
- `std::piecewise_constant_distribution <https://en.cppreference.com/w/cpp/numeric/random/piecewise_constant_distribution.html>`_
- `std::piecewise_linear_distribution <https://en.cppreference.com/w/cpp/numeric/random/piecewise_linear_distribution.html>`_
