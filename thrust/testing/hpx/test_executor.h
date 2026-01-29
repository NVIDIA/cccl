/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <hpx/execution.hpp>

struct test_sync_executor
{
  using execution_category = hpx::execution::sequenced_execution_tag;

  template <typename F, typename... Ts>
  friend decltype(auto)
  tag_invoke(hpx::parallel::execution::sync_execute_t, test_sync_executor const&, F&& f, Ts&&... ts)
  {
    return hpx::invoke(std::forward<F>(f), std::forward<Ts>(ts)...);
  }
};

namespace hpx::execution::experimental
{

template <>
struct is_one_way_executor<test_sync_executor> : std::true_type
{};

} // namespace hpx::execution::experimental
