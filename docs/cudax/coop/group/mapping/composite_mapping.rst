.. _cudax-coop-group-mapping-composite-mapping:

``cuda::experimental::coop::composite_mapping``
===============================================

.. code:: cuda

  namespace cuda::experimental::coop {

  template <typename... Mappings>
  class composite_mapping : public /*group-mapping-interface-for*/<composite_mapping>
  {
    cuda::std::tuple<Mappings...> /*mappings_*/; // exposition-only

  public:
    __device__ constexpr composite_mapping(const Mappings&... mappings)
      noexcept((cuda::std::is_nothrow_copy_constructible_v<Mappings> && ...))
      : /*mappings_*/{mappings...}
    {}
  };

  template <typename... Mappings>
  composite_mapping(const Mappings&...) -> composite_mapping<Mappings...>;

  template </*mapping-type*/ Lhs, /*mapping-type*/ Rhs>
  [[nodiscard]] __device__ constexpr
  composite_mapping<Lhs, Rhs> operator|(const Lhs& lhs, const Rhs& rhs)
    noexcept(cuda::std::is_nothrow_constructible_v<composite_mapping<Lhs, Rhs>, const Lhs&, const Rhs&>);

  template <typename... LhsMappings, /*mapping-type*/ Rhs>
  [[nodiscard]] __device__ constexpr
  composite_mapping<LhsMappings..., Rhs> operator|(const composite_mapping<LhsMappings...>& lhs, const Rhs& rhs)
    noexcept(cuda::std::is_nothrow_constructible_v<composite_mapping<LhsMappings..., Rhs>,
                                                    const LhsMappings&...,
                                                    const Rhs&>);

  template </*mapping-type*/ Lhs, typename... RhsMappings>
  [[nodiscard]] __device__ constexpr
  composite_mapping<Lhs, RhsMappings...> operator|(const Lhs& lhs, const composite_mapping<RhsMappings...>& rhs)
    noexcept(cuda::std::is_nothrow_constructible_v<composite_mapping<Lhs, RhsMappings...>,
                                                    const Lhs&,
                                                    const RhsMappings&...>);

  template <typename... LhsMappings, typename... RhsMappings>
  [[nodiscard]] __device__ constexpr
  composite_mapping<LhsMappings..., RhsMappings...> operator|(const composite_mapping<LhsMappings...>& lhs,
                                                              const composite_mapping<RhsMappings...>& rhs)
    noexcept(cuda::std::is_nothrow_constructible_v<composite_mapping<LhsMappings..., RhsMappings...>,
                                                    const LhsMappings&...,
                                                    const RhsMappings&...>);

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::composite_mapping`` is a mapping that combines multiple mappings together, so they can be used as one during group construction. The type itself shouldn't be used directly; users should use the pipe operator (``|``) to merge the mappings instead.

Examples
--------

.. code:: cuda

  #include <cuda/experimental/coop/group>

  namespace cudax = cuda::experimental;

  __global__ void kernel()
  {
    // Creates a mapping that groups units from the previous mapping results by 4 and then selects only 3 of those
    // threads to return a valid mapping result.
    auto m = cudax::coop::group_by{4} | cudax::coop::take{3};
  }
