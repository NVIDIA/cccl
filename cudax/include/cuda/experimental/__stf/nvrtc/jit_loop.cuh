template <typename shape_t, typename static_types_tup, typename F>
__device__ void jit_loop(F f, static_types_tup args)
{
  const shape_t static_shape;

  //const auto targs = ::cuda::std::make_tuple(static_arg0, static_arg1);

  size_t _i          = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t _step = blockDim.x * gridDim.x;
  const size_t n     = static_shape.size();

  auto explode_args = [&](auto&&... data) {
    auto const explode_coords = [&](auto&&... coords) {
      f(coords..., data...);
    };
    // For every linearized index in the shape
    for (; _i < n; _i += _step)
    {
      ::cuda::std::apply(explode_coords, static_shape.index_to_coords(_i));
    }
  };
  ::cuda::std::apply(explode_args, ::std::move(args));
}
