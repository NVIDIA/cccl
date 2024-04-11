#!/bin/bash
set -e

## Usage: dump_and_check test.a test.cu PREFIX
input_archive=$1
input_testfile=$2
input_prefix=$3

cuobjdump --dump-sass $input_archive | FileCheck --check-prefix $input_prefix $input_testfile
