#!/bin/bash
set -e

## Usage: dump_and_check filter test.ptx test.cu JSON_ID
input_filter=$1
input_ptx=$2
input_testfile=$3
input_json_id=$4

$input_filter $input_ptx $input_json_id | FileCheck $input_testfile
