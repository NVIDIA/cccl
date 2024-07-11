# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tempfile
import re
from collections import namedtuple

version = namedtuple('version', ('major', 'minor'))
code = namedtuple('code', ('kind', 'version', 'data'))
symbol = namedtuple('symbol', ('kind', 'name'))
dim3 = namedtuple('dim3', ('x', 'y', 'z'))

def make_binary_tempfile(content, suffix):
    tmp = tempfile.NamedTemporaryFile(mode='w+b', suffix=suffix, buffering=0)
    tmp.write(content)
    return tmp

def check_in(name, arg, set):
    if arg not in set:
        raise ValueError(f"{name} must be in {set} ; got {name} = {arg}")

def check_not_in(name, arg, set):
    if arg in set:
        raise ValueError(f"{name} must not be any of those value {set} ; got {name} = {arg}")

def check_contains(set, key):
    if key not in set:
        raise ValueError(f"{key} must be in {set}")

def check_dim3(name, arg):
    if len(arg) != 3:
        raise ValueError(f"{name} should be a length-3 tuple ; got {name} = {arg}")

def find_unsigned(name, txt):
    regex = re.compile(f'.global .align 4 .u32 {name} = ([0-9]*);', re.MULTILINE)
    found = regex.search(txt)
    if found is None: # TODO: improve regex logic
        regex = re.compile(f'.global .align 4 .u32 {name};', re.MULTILINE)
        found = regex.search(txt)
        if found is not None:
            return 0
        else:
            raise ValueError(f"{name} not found in text")
    else:
        return int(found.group(1))

def find_mangled_name(name, txt):
    regex = re.compile(f'[_a-zA-Z0-9]*{name}[_a-zA-Z0-9]*', re.MULTILINE)
    return regex.search(txt).group(0)

def find_dim2(name, txt):
    return (find_unsigned(f'{name}_x', txt), find_unsigned(f'{name}_y', txt))

def find_dim3(name, txt):
    return (find_unsigned(f'{name}_x', txt), find_unsigned(f'{name}_y', txt), find_unsigned(f'{name}_z', txt))
