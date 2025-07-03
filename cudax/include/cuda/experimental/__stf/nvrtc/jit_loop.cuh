template <typename shape_t, typename static_types_tup, typename F>
__device__ void jit_loop(F f, static_types_tup args)
{
  size_t _i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t _step = blockDim.x * gridDim.x;
  const size_t n     = shape_t::size();

  auto explode_args = [&](auto&&... data) {
    auto const explode_coords = [&](auto&&... coords) {
      f(coords..., data...);
    };
    // For every linearized index in the shape
    for (; _i < n; _i += _step)
    {
      ::cuda::std::apply(explode_coords, shape_t::index_to_coords(_i));
    }
  };
  ::cuda::std::apply(explode_args, ::std::move(args));
}
