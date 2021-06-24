# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

#! /usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <image>"
  exit 1
fi


curl -s -S "https://registry.hub.docker.com/v2/repositories/${@}/tags/?page_size=100" | jq '."results"[]["name"]' | sort

