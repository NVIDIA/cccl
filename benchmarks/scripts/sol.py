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
    storages = {}
    algnames = set()
    filenames_map = get_filenames_map(files)
    for file in files:
        storage = cccl.bench.StorageBase(file)
        algnames.update(storage.algnames())
        storages[filenames_map[file]] = storage

    result = {}
    for algname in algnames:
        for subbench in storage.subbenches(algname):
            subbench_df = None
            for file in storages:
                storage = storages[file]
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
            result[algname + '.' + subbench] = subbench_df

    return result


def alg_bws(dfs):
    medians = None
    for algname in dfs:
        df = dfs[algname]
        df['alg'] = algname
        median = df.groupby([col for col in df.columns if col != 'bw'])['bw'].median().reset_index()
        if medians is None:
            medians = median
        else:
            medians = pd.concat([medians, median])
    medians['hue'] = medians['ctk'].astype(str) + ' ' + medians['cccl'].astype(
        str) + ' ' + medians['gpu'].astype(str) + ' ' + medians['variant']
    return medians.drop(columns=['ctk', 'cccl', 'gpu', 'variant'])


def file_exists(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"The file '{value}' does not exist.")
    return value


def plot_sol(medians):
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(data=medians, x='alg', y='bw', hue='hue')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, rotation_mode='anchor', ha='right')
    plt.savefig('sol.png')


def sol():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument('files', type=file_exists, nargs='+', help='At least one file is required.')
    plot_sol(alg_bws(alg_dfs(parser.parse_args().files)))


if __name__ == "__main__":
    sol()
