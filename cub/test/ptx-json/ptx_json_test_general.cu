#include <cub/detail/ptx-json/json.cuh>

// CHECK: {"a":1,"b":2,"c":[1,2,"a"]}

__device__ auto& fn()
{
  return ptx_json::id<ptx_json::string("test-json-id")>() =
           ptx_json::object<(ptx_json::key<"a">() = ptx_json::value<1>()),
                            (ptx_json::key<"b">() = ptx_json::value<2>()),
                            (ptx_json::key<"c">() = ptx_json::array<1, 2, ptx_json::string("a")>())>();
}
