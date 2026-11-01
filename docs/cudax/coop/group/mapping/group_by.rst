.. _cudax-coop-group-mapping-group-by:

``cuda::experimental::coop::group_by``
======================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <cuda::std::size_t StaticUnitCount>
  class group_by : public /*group-mapping-interface-for*/<group_by>
  {
    cuda::std::uint32_t /*unit_count_*/; // exposition-only

  public:
    __device__ explicit constexpr group_by(cuda::std::uint32_t unit_count) noexcept
      : /*unit_count_*/{unit_count}
    {}
  };

  template <typename T>
  group_by(T) -> group_by</*maybe-static-extent*/<T>, true>;

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::group_by`` is a mapping that splits every group from the previous mapping result into groups of ``unit_count`` units. The ``unit_count`` can be either static (when the type is ``/*integral_constant-like*/``) or dynamic otherwise.

During exhaustive mapping, if the previous mapping result's ``unit_count`` is not divisible by ``unit_count`` without a remainder, the behaviour is undefined. During non-exhaustive mapping, all of the ranks that would not form a full group will be left without a group.

The returned mapping result is always contiguous if the previous mapping result was contiguous.

Examples
--------

.. code:: cuda

  #include <cuda/std/utility>

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Creates a mapping that splits units from the previous mapping into subgroups that consist of 32 units.
    cudax::coop::group_by m1{32};

    // Creates a mapping that splits units from the previous mapping into subgroups that consist of 4 units. The unit count
    // will be statically known.
    cudax::coop::group_by m2{cuda::std::cw<4>};

    // Creates a mapping that splits units from the previous mapping into subgroups that consist of 64 units. The unit count
    // will be statically known. If the number of units from the previous mapping is not divisible by 64 without a remainder,
    // those units will be excluded and will return an invalid mapping result.
    cudax::coop::group_by m3{cudax::coop::non_exhaustive, cuda::std::cw<64>};
  }
