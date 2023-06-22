#!/usr/bin/env python3

import os
import re
import cub
import math
import argparse
import itertools
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats.mstats import hdquantiles

pd.options.display.max_colwidth = 100

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = itertools.cycle(default_colors)
color_map = {}

precision = 0.01
sensitivity = 0.5


def get_bench_columns():
    return ['variant', 'elapsed', 'center', 'samples']


def get_extended_bench_columns():
    return get_bench_columns() + ['speedup', 'base_samples']


def compute_speedup(df):
    bench_columns = get_bench_columns()
    workload_columns = [col for col in df.columns if col not in bench_columns]
    base_df = df[df['variant'] == 'base'].drop(columns=['variant']).rename(
        columns={'center': 'base_center', 'samples': 'base_samples'})
    base_df.drop(columns=['elapsed'], inplace=True)

    merged_df = df.merge(
        base_df, on=[col for col in df.columns if col in workload_columns])
    merged_df['speedup'] = merged_df['base_center'] / merged_df['center']
    merged_df = merged_df.drop(columns=['base_center'])
    return merged_df


def get_ct_axes(df):
    ct_axes = []
    for col in df.columns:
        if '{ct}' in col:
            ct_axes.append(col)

    return ct_axes


def get_rt_axes(df):
    rt_axes = []
    excluded_columns = get_ct_axes(df) + get_extended_bench_columns()

    for col in df.columns:
        if col not in excluded_columns:
            rt_axes.append(col)

    return rt_axes


def ct_space(df):
    ct_axes = get_ct_axes(df)

    unique_ct_combinations = []
    for _, row in df[ct_axes].drop_duplicates().iterrows():
        unique_ct_combinations.append({})
        for col in ct_axes:
            unique_ct_combinations[-1][col] = row[col]

    return unique_ct_combinations


def extract_case(df, ct_point):
    tuning_df_loc = None

    for ct_axis in ct_point:
        if tuning_df_loc is None:
            tuning_df_loc = (df[ct_axis] == ct_point[ct_axis])
        else:
            tuning_df_loc = tuning_df_loc & (df[ct_axis] == ct_point[ct_axis])

    tuning_df = df.loc[tuning_df_loc].copy()
    for ct_axis in ct_point:
        tuning_df.drop(columns=[ct_axis], inplace=True)

    return tuning_df


def extract_rt_axes_values(df):
    rt_axes = get_rt_axes(df)
    rt_axes_values = {}

    for rt_axis in rt_axes:
        rt_axes_values[rt_axis] = list(df[rt_axis].unique())

    return rt_axes_values


def extract_rt_space(df):
    rt_axes = get_rt_axes(df)
    rt_axes_values = []
    for rt_axis in rt_axes:
        values = df[rt_axis].unique()
        rt_axes_values.append(["{}={}".format(rt_axis, v) for v in values])
    return list(itertools.product(*rt_axes_values))


def filter_variants(df, group):
    rt_axes = get_rt_axes(df)
    unique_combinations = set(
        df[rt_axes].drop_duplicates().itertuples(index=False))
    group_combinations = set(
        group[rt_axes].drop_duplicates().itertuples(index=False))
    has_all_combinations = group_combinations == unique_combinations
    return has_all_combinations


def extract_complete_variants(df):
    return df.groupby('variant').filter(functools.partial(filter_variants, df))


def compute_workload_score(rt_axes_values, rt_axes_ids, weight_matrix, row):
    rt_workload = []
    for rt_axis in rt_axes_values:
        rt_workload.append("{}={}".format(rt_axis, row[rt_axis]))
    
    weight = cub.bench.get_workload_weight(rt_workload, rt_axes_values, rt_axes_ids, weight_matrix)
    return row['speedup'] * weight


def compute_variant_score(rt_axes_values, rt_axes_ids, weight_matrix, group):
    workload_score_closure = functools.partial(compute_workload_score, rt_axes_values, rt_axes_ids, weight_matrix)
    score_sum = group.apply(workload_score_closure, axis=1).sum()
    return score_sum


def extract_scores(df):
    rt_axes_values = extract_rt_axes_values(df)
    rt_axes_ids = cub.bench.compute_axes_ids(rt_axes_values)
    weight_matrix = cub.bench.compute_weight_matrix(rt_axes_values, rt_axes_ids)

    score_closure = functools.partial(compute_variant_score, rt_axes_values, rt_axes_ids, weight_matrix)
    grouped = df.groupby('variant')
    scores = grouped.apply(score_closure).reset_index()
    scores.columns = ['variant', 'score']

    stat = grouped.agg(mins = ('speedup', 'min'),
                       means = ('speedup', 'mean'), 
                       maxs = ('speedup', 'max'))
    
    result = pd.merge(scores, stat, on='variant')
    return result.sort_values(by=['score'], ascending=False)


def distributions_are_different(alpha, row):
    ref_samples = row['base_samples']
    cmp_samples = row['samples']

    # H0: the distributions are not different
    # H1: the distribution are different
    _, p = mannwhitneyu(ref_samples, cmp_samples)

    # Reject H0
    return p < alpha


def remove_matching_distributions(alpha, df):
    closure = functools.partial(distributions_are_different, alpha)
    return df[df.apply(closure, axis=1)]


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


def iterate_case_dfs(args, callable):
    storages = {}
    algnames = set()
    filenames_map = get_filenames_map(args.files)
    for file in args.files:
        storage = cub.bench.StorageBase(file)
        algnames.update(storage.algnames())
        storages[filenames_map[file]] = storage

    pattern = re.compile(args.R)

    for algname in algnames:
        if not pattern.match(algname):
            continue

        case_dfs = {}
        for file in storages:
            storage = storages[file]
            df = storage.alg_to_df(algname)
            with pd.option_context('mode.use_inf_as_na', True):
                df = df.dropna(subset=['center'], how='all')

            for _, row in df[['ctk', 'cub']].drop_duplicates().iterrows():
                ctk_version = row['ctk']
                cub_version = row['cub']
                ctk_cub_df = df[(df['ctk'] == ctk_version) &
                                (df['cub'] == cub_version)]

                for gpu in ctk_cub_df['gpu'].unique():
                    target_df = ctk_cub_df[ctk_cub_df['gpu'] == gpu]
                    target_df = target_df.drop(columns=['ctk', 'cub', 'gpu'])
                    target_df = compute_speedup(target_df)

                    for ct_point in ct_space(target_df):
                        point_str = ", ".join(["{}={}".format(k, ct_point[k]) for k in ct_point])
                        case_df = extract_complete_variants(extract_case(target_df, ct_point))
                        case_df['variant'] = case_df['variant'].astype(str) + " ({})".format(file)
                        if point_str not in case_dfs:
                            case_dfs[point_str] = case_df
                        else:
                            case_dfs[point_str] = pd.concat([case_dfs[point_str], case_df])
        
        for point_str in case_dfs:
            callable(algname, point_str, case_dfs[point_str])


def case_top(alpha, N, algname, ct_point_name, case_df):
    print("{}[{}]:".format(algname, ct_point_name))

    if alpha < 1.0:
        case_df = remove_matching_distributions(alpha, case_df)

    case_df = extract_complete_variants(case_df)
    print(extract_scores(case_df).head(N))


def top(args):
    iterate_case_dfs(args, functools.partial(case_top, args.alpha, args.top))


def case_coverage(algname, ct_point_name, case_df):
    num_variants = cub.bench.Config().variant_space_size(algname)
    num_covered_variants = len(case_df['variant'].unique())
    coverage = (num_covered_variants / num_variants) * 100
    case_str = "{}[{}]".format(algname, ct_point_name)
    print("{} coverage: {} / {} ({:.4f}%)".format(
          case_str, num_covered_variants, num_variants, coverage))


def coverage(args):
    iterate_case_dfs(args, case_coverage)


def qrde_hd(samples):
    """
    Computes quantile-respectful density estimation based on the Harrell-Davis 
    quantile estimator. The implementation is based on the following post:
    https://aakinshin.net/posts/qrde-hd by Andrey Akinshin
    """
    min_sample, max_sample = min(samples), max(samples)
    num_quantiles = math.ceil(1.0 / precision)
    quantiles = np.linspace(precision, 1 - precision, num_quantiles - 1)
    hd_quantiles = [min_sample] + list(hdquantiles(samples, quantiles)) + [max_sample]
    width = [hd_quantiles[idx + 1] - hd_quantiles[idx] for idx in range(num_quantiles)]
    p = 1.0 / precision
    height = [1.0 / (p * w) for w in width]
    return width, height


def extract_peaks(pdf):
    peaks = []
    for i in range(1, len(pdf) - 1):
        if pdf[i - 1] < pdf[i] > pdf[i + 1]:
            peaks.append(i)
    return peaks


def extract_modes(samples):
    """
    Extract modes from the given samples based on the lowland algorithm:
    https://aakinshin.net/posts/lowland-multimodality-detection/ by Andrey Akinshin
    Implementation is based on the https://github.com/AndreyAkinshin/perfolizer
    LowlandModalityDetector class.
    """
    mode_ids = []

    widths, heights = qrde_hd(samples)
    peak_ids = extract_peaks(heights)
    bin_area = 1.0 / len(heights)

    x = min(samples)
    peak_xs = []
    peak_ys = []
    bin_lower = [x]
    for idx in range(len(heights)):
        if idx in peak_ids:
            peak_ys.append(heights[idx])
            peak_xs.append(x + widths[idx] / 2)
        x += widths[idx]
        bin_lower.append(x)

    def lowland_between(mode_candidate, left_peak, right_peak):
        left, right = left_peak, right_peak
        min_height = min(heights[left_peak], heights[right_peak])
        while left < right and heights[left] > min_height:
            left += 1
        while left < right and heights[right] > min_height:
            right -= 1

        width = bin_lower[right + 1] - bin_lower[left]
        total_area = width * min_height
        total_bin_area = (right - left + 1) * bin_area
        
        if total_bin_area / total_area < sensitivity:
            mode_ids.append(mode_candidate)
            return True
        return False

    previousPeaks = [peak_ids[0]]
    for i in range(1, len(peak_ids)):
        currentPeak = peak_ids[i]
        while previousPeaks and heights[previousPeaks[-1]] < heights[currentPeak]:
            if lowland_between(previousPeaks[0], previousPeaks[-1], currentPeak):
                previousPeaks = []
            else:
                previousPeaks.pop()

        if previousPeaks and heights[previousPeaks[-1]] > heights[currentPeak]:
            if lowland_between(previousPeaks[0], previousPeaks[-1], currentPeak):
                previousPeaks = []

        previousPeaks.append(currentPeak)

    mode_ids.append(previousPeaks[0])
    return mode_ids


def hd_displot(samples, label, ax):
    if label not in color_map:
        color_map[label] = next(color_cycle)
    color = color_map[label]
    widths, heights = qrde_hd(samples)
    mode_ids = extract_modes(samples)

    min_sample, max_sample = min(samples), max(samples)
    
    xs = [min_sample]
    ys = [0]

    peak_xs = []
    peak_ys = []

    x = min(samples)
    for idx in range(len(widths)):
        xs.append(x + widths[idx] / 2)
        ys.append(heights[idx])
        if idx in mode_ids:
            peak_ys.append(heights[idx])
            peak_xs.append(x + widths[idx] / 2)
        x += widths[idx]

    xs = xs + [max_sample]
    ys = ys + [0]

    ax.fill_between(xs, ys, 0, alpha=0.4, color=color)

    quartiles_of_interest = [0.25, 0.5, 0.75]

    for quartile in quartiles_of_interest:
        bin = int(quartile / precision) + 1
        ax.plot([xs[bin], xs[bin]], [0, ys[bin]], color=color)

    ax.plot(xs, ys, label=label, color=color)
    ax.plot(peak_xs, peak_ys, 'o', color=color)
    ax.legend()


def displot(data, ax):
    for variant in data:
        hd_displot(data[variant], variant, ax)


def variant_ratio(data, variant, ax):
    if variant not in color_map:
        color_map[variant] = next(color_cycle)
    color = color_map[variant]

    variant_samples = data[variant]
    base_samples = data['base']

    variant_widths, variant_heights = qrde_hd(variant_samples)
    base_widths, base_heights = qrde_hd(base_samples)

    quantiles = []
    ratios = []

    base_x = min(base_samples)
    variant_x = min(variant_samples)

    for i in range(1, len(variant_heights) - 1):
        base_x += base_widths[i] / 2
        variant_x += variant_widths[i] / 2
        quantiles.append(i * precision)
        ratios.append(base_x / variant_x)
    
    ax.plot(quantiles, ratios, label=variant, color=color)
    ax.axhline(1, color='red', alpha=0.7)
    ax.legend()
    ax.tick_params(axis='both', direction='in', pad=-22)


def ratio(data, ax):
    for variant in data:
        if variant != 'base':
            variant_ratio(data, variant, ax)


def case_variants(pattern, mode, algname, ct_point_name, case_df):
    title = "{}[{}]:".format(algname, ct_point_name)
    df = case_df[case_df['variant'].str.contains(pattern, regex=True)].reset_index(drop=True)
    rt_axes = get_rt_axes(df)
    rt_axes_values = extract_rt_axes_values(df)

    vertical_axis_name = rt_axes[0]
    if 'Elements{io}[pow2]' in rt_axes:
        vertical_axis_name = 'Elements{io}[pow2]'
    horizontal_axes = rt_axes
    horizontal_axes.remove(vertical_axis_name)
    vertical_axis_values = rt_axes_values[vertical_axis_name] 

    vertical_axis_ids = {}
    for idx, val in enumerate(vertical_axis_values):
        vertical_axis_ids[val] = idx

    def extract_horizontal_space(df):
        values = []
        for rt_axis in horizontal_axes:
            values.append(["{}={}".format(rt_axis, v) for v in df[rt_axis].unique()])
        return list(itertools.product(*values))

    if len(horizontal_axes) > 0:
        idx = 0
        horizontal_axis_ids = {}
        for point in extract_horizontal_space(df):
            horizontal_axis_ids[" / ".join(point)] = idx
            idx = idx + 1

    num_rows = len(vertical_axis_ids)
    num_cols = max(1, len(extract_horizontal_space(df)))

    if num_rows == 0:
        return

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, gridspec_kw = {'wspace': 0, 'hspace': 0})


    for _, vertical_row_description in df[[vertical_axis_name]].drop_duplicates().iterrows():
        vertical_val = vertical_row_description[vertical_axis_name]
        vertical_id = vertical_axis_ids[vertical_val]
        vertical_name = "{}={}".format(vertical_axis_name, vertical_val)

        vertical_df = df[df[vertical_axis_name] == vertical_val]

        for _, horizontal_row_description in vertical_df[horizontal_axes].drop_duplicates().iterrows():
            horizontal_df = vertical_df

            for axis in horizontal_axes:
                horizontal_df = horizontal_df[horizontal_df[axis] == horizontal_row_description[axis]]

            horizontal_id = 0

            if len(horizontal_axes) > 0:
                horizontal_point = []
                for rt_axis in horizontal_axes:
                    horizontal_point.append("{}={}".format(rt_axis, horizontal_row_description[rt_axis]))
                horizontal_name = " / ".join(horizontal_point)
                horizontal_id = horizontal_axis_ids[horizontal_name]
                ax=axes[vertical_id, horizontal_id]
            else:
                ax=axes[vertical_id]
                ax.set_ylabel(vertical_name)

            data = {}
            for _, variant in horizontal_df[['variant']].drop_duplicates().iterrows():
                variant_name = variant['variant']
                if 'base' not in data:
                    data['base'] = horizontal_df[horizontal_df['variant'] == variant_name].iloc[0]['base_samples']
                data[variant_name] = horizontal_df[horizontal_df['variant'] == variant_name].iloc[0]['samples']

            if mode == 'pdf':
                # sns.histplot(data=data, ax=ax, kde=True)
                displot(data, ax)
            else:
                ratio(data, ax)

            if len(horizontal_axes) > 0:
                ax=axes[vertical_id, horizontal_id]
                if vertical_id == (num_rows - 1):
                    ax.set_xlabel(horizontal_name)
                if horizontal_id == 0:
                    ax.set_ylabel(vertical_name)
                else:
                    ax.set_ylabel('')

    for ax in axes.flat:
        ax.set_xticklabels([])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



def variants(args, mode):
    pattern = re.compile(args.variants_pdf) if mode == 'pdf' else re.compile(args.variants_ratio)
    iterate_case_dfs(args, functools.partial(case_variants, pattern, mode))


def file_exists(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"The file '{value}' does not exist.")
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument(
        '-R', type=str, default='.*', help="Regex for benchmarks selection.")
    parser.add_argument(
        '--list-benches', action=argparse.BooleanOptionalAction, help="Show available benchmarks.")
    parser.add_argument(
        '--coverage', action=argparse.BooleanOptionalAction, help="Show variant space coverage.")
    parser.add_argument(
        '--top', default=7, type=int, action='store', nargs='?', help="Show top N variants with highest score.")
    parser.add_argument(
        'files', type=file_exists, nargs='+', help='At least one file is required.')
    parser.add_argument(
        '--alpha', default=1.0, type=float)
    parser.add_argument(
        '--variants-pdf', type=str, help="Show matching variants data.")
    parser.add_argument(
        '--variants-ratio', type=str, help="Show matching variants data.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.list_benches:
        cub.bench.list_benches()
        return

    if args.coverage:
        coverage(args)
        return
    
    if args.variants_pdf:
        variants(args, 'pdf')
        return

    if args.variants_ratio:
        variants(args, 'ratio')
        return

    top(args)


if __name__ == "__main__":
    main()
