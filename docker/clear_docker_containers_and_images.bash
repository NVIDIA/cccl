# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

#! /usr/bin/env bash

docker container rm $(docker container ls -a | awk '{ print $1 }' | grep -v CONTAINER) > /dev/null 2>&1
docker image rm $(docker image ls -a | awk '{ print $3 }'  | grep -v IMAGE) > /dev/null 2>&1

exit 0

