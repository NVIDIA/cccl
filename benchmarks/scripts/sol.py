#!/usr/bin/env python3

import os
import sys
import cccl
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def alg_dfs(files, min_elements_pow2=0):
    result = {}
    for file in files:
        storage = cccl.bench.StorageBase(file)
        for algname in storage.algnames():
            for subbench in storage.subbenches(algname):
                subbench_df = None
                df = storage.alg_to_df(algname, subbench)
                df = df.map(lambda x: x if is_finite(x) else np.nan)
                df = df.dropna(subset=['center'], how='all')
                if 'Elements{io}[pow2]' in df.columns:
                    df['Elements{io}[pow2]'] = df['Elements{io}[pow2]'].astype(int)
                    df = df[df['Elements{io}[pow2]'] >= min_elements_pow2]
                df = df.filter(items=['ctk', 'cccl', 'gpu', 'variant', 'bw'])
                df['variant'] = df['variant'].astype(str) + " ({})".format(file)
                df['bw'] = df['bw'] * 100
                if subbench_df is None:
                    subbench_df = df
                else:
                    subbench_df = pd.concat([subbench_df, df])
            fused_algname = algname + '.' + subbench
            if fused_algname in result:
                result[fused_algname] = pd.concat([result[fused_algname], subbench_df])
            else:
                result[fused_algname] = subbench_df

    return result


def alg_bws(dfs):
    medians = None
    for algname in dfs:
        df = dfs[algname]
        df['alg'] = algname
        if df is None:
            medians = df
        else:
            medians = pd.concat([medians, df])
    medians['hue'] = medians['ctk'].astype(str) + ' ' + medians['cccl'].astype(
        str) + ' ' + medians['gpu'].astype(str) + ' ' + medians['variant']
    return medians.drop(columns=['ctk', 'cccl', 'gpu', 'variant'])


def file_exists(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"The file '{value}' does not exist.")
    return value


def plot_sol(medians):
    ax = sns.boxenplot(data=medians, x='alg', y='bw', hue='hue')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, rotation_mode='anchor', ha='right')
    plt.show()


def sol():
    min_elements_pow2 = 28
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument('files', type=file_exists, nargs='+', help='At least one file is required.')
    plot_sol(alg_bws(alg_dfs(parser.parse_args().files, min_elements_pow2)))


if __name__ == "__main__":
    sol()
