#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_PAR_NO_SYNC_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_PAR_NO_SYNC_HPP_

#include <execution>
#include <type_traits>

template <typename _ExecutionPolicy>
struct no_sync_policy {
    _ExecutionPolicy __exec_;

public:
    no_sync_policy(_ExecutionPolicy&& __exec) : __exec_(__exec) {}
    no_sync_policy(_ExecutionPolicy const& __exec) : __exec_(__exec) {}

    operator _ExecutionPolicy() { return __exec_; }
};

namespace std::experimental {
template <>
struct is_execution_policy<no_sync_policy<std::execution::parallel_policy>> : std::true_type {};
}  // namespace std::experimental

template <typename _ExecutionPolicy>
auto no_sync(_ExecutionPolicy&& __exec) {
    using __Striped = std::remove_cv_t<std::remove_reference_t<_ExecutionPolicy>>;
    return no_sync_policy<__Striped>((_ExecutionPolicy &&) __exec);
}

#endif
