# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import json
import os
import pickle

_ENABLE_CACHE = "CCCL_ENABLE_CACHE" in os.environ
if _ENABLE_CACHE:
    _CACHE_LOCATION = os.path.join(os.path.expanduser("~"), ".cache", "cccl")
    if not os.path.exists(_CACHE_LOCATION):
        os.makedirs(_CACHE_LOCATION)


# We use
# json.dumps to serialize args/kwargs to a string
# hashlib to compute the hash
def json_hash(*args, **kwargs):
    hasher = hashlib.sha1()
    hasher.update(json.dumps([args, kwargs]).encode("utf-8"))
    return hasher.hexdigest()


def disk_cache(func):
    def cacher(*args, **kwargs):
        if _ENABLE_CACHE:
            # compute hash(args, kwargs)
            h = json_hash(*args, **kwargs)
            # if file exist...
            if os.path.isfile(os.path.join(_CACHE_LOCATION, h)):
                # open it
                with open(os.path.join(_CACHE_LOCATION, h), "rb") as f:
                    out = pickle.load(f)
                # return cache
                return out
            else:
                # compute output
                out = func(*args, **kwargs)
                # store to file
                with open(os.path.join(_CACHE_LOCATION, h), "wb") as f:
                    pickle.dump(out, f)
                return out
        else:
            return func(*args, **kwargs)

    return cacher
