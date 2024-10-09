// For GCC 9 (< 10), <execution> unconditionally includes a TBB header file.
// If GCC < 10 was not built with TBB support, this causes a build error.
//
// Above issue not present when compiling with nvc++
// Always needs this for `is_execution_policy`
// #if (! defined(__GNUC__)) || (__GNUC__ > 9)
#include <execution>
// #endif

#include <type_traits>

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{
namespace impl
{
// the execution policy used for default serial inline implementations
struct inline_exec_t
{};

// The execution policy used when no execution policy is provided
// It must be remapped to some other execution policy, which the default mapper does
struct default_exec_t
{};

// helpers
template <class T>
struct is_inline_exec : std::false_type
{};
template <>
struct is_inline_exec<inline_exec_t> : std::true_type
{};
template <class T>
inline constexpr bool is_inline_exec_v = is_inline_exec<T>::value;
} // namespace impl
} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

namespace std::experimental
{

template <class ExecutionPolicy, bool is_known_exec_policy = std::is_execution_policy<ExecutionPolicy>::value>
struct is_execution_policy : std::false_type
{};

template <class ExecutionPolicy>
struct is_execution_policy<ExecutionPolicy, true> : std::true_type
{};

template <class ExecutionPolicy>
static constexpr bool is_execution_policy_v = ::std::experimental::is_execution_policy<ExecutionPolicy>::value;

template <>
struct is_execution_policy<__p1673_version_0::linalg::impl::inline_exec_t> : std::true_type
{};

template <>
struct is_execution_policy<__p1673_version_0::linalg::impl::default_exec_t> : std::true_type
{};

} // namespace std::experimental

#if defined(__LINALG_ENABLE_CUBLAS) || defined(__LINALG_ENABLE_BLAS)
#  include <experimental/__p1673_bits/exec_policy_wrapper_nvhpc.hpp>
#endif

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{
template <class T>
auto execpolicy_mapper(T)
{
  return std::experimental::linalg::impl::inline_exec_t();
}
} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std
