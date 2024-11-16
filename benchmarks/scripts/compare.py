#!/usr/bin/env python3

import os
import sys
import cccl
import argparse
import numpy as np
import pandas as pd


def get_filenames_map(arr):
    if not arr:
        return []

    prefix = arr[0]
    for string in arr:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break

    return {string: string[len(prefix):] for string in arr}


def is_finite(x):
    if isinstance(x, float):
        return x != np.inf and x != -np.inf
    return True


def filter_by_problem_size(df):
    min_elements_pow2 = 28
    if 'Elements{io}[pow2]' in df.columns:
        df['Elements{io}[pow2]'] = df['Elements{io}[pow2]'].astype(int)
        df = df[df['Elements{io}[pow2]'] >= min_elements_pow2]
    return df


def filter_by_offset_type(df):
    if 'OffsetT{ct}' in df.columns:
        df = df[(df['OffsetT{ct}'] == 'I32') | (df['OffsetT{ct}'] == 'U32')]
    return df


def filter_by_type(df):
    if 'T{ct}' in df:
        # df = df[df['T{ct}'].str.contains('64')]
        df = df[~df['T{ct}'].str.contains('C')]
    elif 'KeyT{ct}' in df:
        # df = df[df['KeyT{ct}'].str.contains('64')]
        df = df[~df['KeyT{ct}'].str.contains('C')]
    return df


def alg_dfs(file):
    result = {}
    storage = cccl.bench.StorageBase(file)
    for algname in storage.algnames():
        subbench_df = None
        for subbench in storage.subbenches(algname):
            df = storage.alg_to_df(algname, subbench)
            df = df.map(lambda x: x if is_finite(x) else np.nan)
            df = df.dropna(subset=['center'], how='all')
            df = filter_by_type(filter_by_offset_type(filter_by_problem_size(df)))
            df['Noise'] = df['samples'].apply(lambda x: np.std(x) / np.mean(x)) * 100
            df['Mean'] = df['samples'].apply(lambda x: np.mean(x))
            df = df.drop(columns=['samples', 'center', 'bw', 'elapsed'])
            if subbench_df is None:
                subbench_df = df
            else:
                subbench_df = pd.concat([subbench_df, df])
        fused_algname = algname + '.' + subbench
        if fused_algname in result:
            result[fused_algname] = pd.concat([result[fused_algname], subbench_df])
        else:
            result[fused_algname] = subbench_df

    for algname in result:
        if result[algname]['cccl'].nunique() != 1:
            raise ValueError(f"Multiple CCCL versions in one db '{algname}'")
        result[algname] = result[algname].drop(columns=['cccl'])

    return result


def file_exists(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"The file '{value}' does not exist.")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument('reference', type=file_exists, help='Reference database file.')
    parser.add_argument('compare', type=file_exists, help='Comparison database file.')
    return parser.parse_args()


def compare():
    args = parse_args()
    reference_df = alg_dfs(args.reference)
    compare_df = alg_dfs(args.compare)
    for alg in reference_df.keys() & compare_df.keys():
        print()
        print()
        print(f'# {alg}')
        merge_columns = [col for col in reference_df[alg].columns if col not in ['Noise', 'Mean']]
        df = pd.merge(reference_df[alg], compare_df[alg], on=merge_columns, suffixes=('Ref', 'Cmp'))
        df['Diff'] = df['MeanCmp'] - df['MeanRef']
        df['FDiff'] = (df['Diff'] / df['MeanRef']) * 100

        for _, row in df[['ctk', 'gpu', 'variant']].drop_duplicates().iterrows():
            ctk_version = row['ctk']
            variant = row['variant']
            gpu = row['gpu']
            case_df = df[(df['ctk'] == ctk_version) & (df['gpu'] == gpu) & (df['variant'] == variant)]
            case_df = case_df.drop(columns=['ctk', 'gpu', 'variant'])
            print()
            print(f'## CTK {ctk_version} GPU {gpu} ({variant})')
            print()
            print(case_df.to_markdown(index=False))


if __name__ == "__main__":
    compare()
