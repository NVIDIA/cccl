// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "checked_env_default_queries.h"

using namespace default_query_test;

int main(int, char**)
{
  // Defaults do not excuse customizations that the input explicitly advertises.
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  (void) cuda::checked_env<get_stop_token_t>(advertised_empty_a{});
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  (void) cuda::checked_env<get_stop_token_t>(ex::env{advertised_empty_b{}});

  // A default cannot repair an advertised customization with a mismatched argument signature.
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  (void) cuda::checked_env<rvalue_query>(invalid_argument_advertisement{});

  // A real provider must not conceal a missing customization advertised by another component.
  const auto first = ex::env{advertised_empty_a{}, provider{{1, true}}};
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  ex::__detail::__validate_env<decltype(first)>();
  const auto last = ex::env{provider{{1, true}}, advertised_empty_b{}};
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  ex::__detail::__validate_env<decltype(last)>();
  return 0;
}
