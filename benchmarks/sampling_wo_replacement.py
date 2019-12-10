# -*- coding: utf-8 -*-
# Copyright 2019 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark sampling without replacement.

The active learning procedure requires randomly generating candidate
trials using sampling without replacement, while also using a
non-uniform weighting scheme. This component represents a non-trivial
bottleneck.

The new implementation is not always faster than using a for loop with
np.choice. Use this function to benchmark your use case.

See: https://medium.com/ibm-watson/
        incredibly-fast-random-sampling-in-python-baf154bd836a
"""

import json
import time

import numpy as np

from psiz.generator import choice_wo_replace
from psiz.benchmark import system_info, benchmark_filename


def main():
    """Run benchmark."""
    # Benchmark settings.
    n_restart = 50
    n_keep = 10

    # Problem settings.
    n_stimuli = 1000
    n_trial = 10000
    n_reference = 8

    candidate_idx = np.arange(1, n_stimuli)
    np.random.seed(123)
    candidate_prob = np.random.random(len(candidate_idx))
    candidate_prob = candidate_prob / np.sum(candidate_prob)

    # Initialize.
    time_list_old = []
    time_list_new = []

    for i_restart in range(n_restart):
        start_time = time.time()
        stimulus_set = choice_wo_replace(candidate_idx, (n_trial, n_reference), candidate_prob)
        duration = time.time() - start_time
        time_list_new.append(duration)

    for i_restart in range(n_restart):
        stimulus_set = np.zeros([n_trial, n_reference])
        start_time = time.time()
        for i_trial in range(n_trial):
            stimulus_set[i_trial, :] = np.random.choice(
                candidate_idx, (1, n_reference), replace=False,
                p=candidate_prob
            )
        duration = time.time() - start_time
        time_list_old.append(duration)

    stats_old = bench_stats(time_list_old, n_keep)
    stats_new = bench_stats(time_list_new, n_keep)

    sys_info = system_info()

    test_info = {}
    test_info['n_stimuli'] = n_stimuli
    test_info['n_trial'] = n_trial
    test_info['n_reference'] = n_reference
    test_info['n_restart'] = n_restart
    test_info['duration_old'] = str_bench_stats(stats_old)
    test_info['duration_new'] = str_bench_stats(stats_new)
    test_info['speedup_new_over_old'] = stats_new['best']['mean'] / stats_old['best']['mean']

    report = {
        'test': test_info,
        'system': sys_info
    }
    fp_report = benchmark_filename()
    with open(fp_report, 'w') as outfile:
        json.dump(report, outfile)


def bench_stats(time_list, n_keep):
    """Compute benchmark statistics."""
    time_list = np.sort(time_list)
    time_list_best = time_list[0:n_keep]

    result = {
        'all': {
            'mean': np.mean(time_list),
            'median': np.median(time_list),
            'std': np.std(time_list)
        },
        'best': {
            'mean': np.mean(time_list_best),
            'median': np.median(time_list_best),
            'std': np.std(time_list_best)
        }
    }
    return result


def str_bench_stats(stats):
    """Output string for benchmark statistics."""
    p0 = first_significant_digit(stats['best']['median'])
    p1 = first_significant_digit(stats['best']['std'])

    smallest = np.min([p0, p1]) - 1

    s = '{0:.0f} {3} {1:.0f} x 10e{2} s (median {3} std.)'.format(
        stats['best']['median'] * 10**(-smallest),
        stats['best']['std'] * 10**(-smallest),
        smallest,
        u"\u00B1"
    )
    return s


def manage_benchmark_times(best_time_list, new_time):
    """Mange benchmark times."""
    best_time_list = np.sort(best_time_list)
    locs = np.less(new_time, best_time_list)
    if np.sum(locs) > 0:
        x = 1
    return best_time_list


def first_significant_digit(d):
    """Determine location of first significant digit."""
    p = int(np.ceil(np.log10(d)))
    return p


if __name__ == "__main__":
    main()
