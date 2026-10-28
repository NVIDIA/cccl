.. _cudax-coop-group-mapping-take:

``cuda::experimental::coop::take``
==================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <cuda::std::size_t StaticUnitCount>
  class take : public /*group-mapping-interface-for*/<take>
  {
    cuda::std::uint32_t /*unit_count_*/; // exposition-only

  public:
    __device__ explicit take(cuda::std::uint32_t unit_count) noexcept
      : /*unit_count_*/{unit_count}
    {}
  };

  template <typename T>
  take(T) -> take</*maybe-static-extent*/<T>>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::take`` is a mapping that selects only the first ``unit_count`` units from the previous mapping result, keeping the same group count. The ``unit_count`` can be either static (when the type is ``/*integral_constant-like*/``) or dynamic otherwise.

The returned mapping result is always non-exhaustive except when both ``unit_count`` and the previous mapping result's static unit count are statically known and have the same value.

The returned mapping result is always contiguous if the previous mapping result was contiguous.

Examples
--------

.. code:: cuda

  #include <cuda/std/utility>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Creates a mapping that selects only the first 32 units from the previous mapping result.
    cudax::coop::take m1{32};

    // Creates a mapping that selects only the first 8 units from the previous mapping result. The unit count will be
    // statically known.
    cudax::coop::take m2{cuda::std::cw<8>};
  }
