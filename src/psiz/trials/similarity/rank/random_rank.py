# -*- coding: utf-8 -*-
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
# ============================================================================
"""Generate rank-type unjudged similarity judgment trials.

Classes:
    RandomRank: Concrete class for generating Rank similarity trials.

"""

import copy
import multiprocessing
import sys
from time import time

import numpy as np

from psiz.data import sample_qr_sets
from psiz.trials.similarity.docket_generator import DocketGenerator
from psiz.trials.similarity.rank.rank_docket import RankDocket


class RandomRank(DocketGenerator):
    """Trial generator that samples query-reference trials.

    Users can guide the sampling process by providing a matrix of
    weights. The generative process can be distributed across multiple
    workers.

    NOTE: When sampling without replacement (i.e., replace=False) you
    must be mindful of the number of unique trials possible. If
    requesting more unique trials than possible, you may be surprised
    when fewer trials are returned.

    """

    def __init__(
        self,
        indices,
        n_reference=2,
        n_select=1,
        w=None,
        replace=True,
        n_highest=None,
        n_worker=1,
        mask_zero=False,
        verbose=0,
    ):
        """Initialize.

        Args:
            indices: A scalar inter or 1D array-like of integers
                indicating the eligible indices. If scalar, index array
                is instantiated as `np.arange(indices)`.
            n_reference (optional): A scalar indicating the number of
                references for each trial.
            n_select (optional): A scalar indicating the number of
                selections an agent must make.
            w (optional): A non-negative square matrix, specifying the
                sampling weight of a particular stimulus. The diagonal
                and off-diagonal elements serve different purposes. The
                diagonal elements indicate the probability of selecting
                a stimulus as a query. These elements are only used when
                calling the `generate` method. Each row of the off-
                diagonal elements corresponds to a particular query and
                indicates the probability of sampling a stimulus to
                serve as a reference for that query. For eample element
                `w[i, j]` indicates the probability stimulus `j` will
                be sampled as a reference for query `i`. By default, all
                stimuli are given equal weight.
                shape=(n_idx, n_idx)
            replace (optional): Boolean indicating if sampled trials
                should be unique. The default `replace=True` does not
                guarantee unique trials.
            n_highest (optional): An integer specifying the number of
                highest probability references that are eligible for
                selection. All references are eligible by default. For
                large problems it can be useful to set this value to
                something less than the total number of stimuli. For
                example, if `n_highest=100`, only the 100 references
                with the highest probability will be considered and all
                other references will be masked.
            n_worker (optional): The number of unique workers (CPUs) to
                divide the work amongst. By default, only one worker is
                used.
            mask_zero (optional): A Boolean indicating if zero should
                be interpretted as a mask value in `stimulus_set`. By
                default, `mask_zero=False`.
            verbose (optional): The verbosity of output.

        """
        DocketGenerator.__init__(self)

        # Check if argument is sclar or array-like.
        indices = np.array(indices, copy=False)
        if indices.ndim == 0:
            indices = np.arange(indices)
        elif indices.ndim != 1:
            raise ValueError("Argument `indices` must be 1D.")
        self.indices = indices
        self.n_idx = len(indices)

        if n_reference > self.n_idx:
            raise ValueError("Argument `n_reference` must be less than `n_idx`")
        if n_select > n_reference:
            raise ValueError("Argument `n_select` must be less than `n_reference`")

        # Sanitize inputs.
        self.n_reference = np.int32(n_reference)
        self.n_select = np.int32(n_select)

        # Check weight matrix `w`.
        if w is None:
            w = np.ones([self.n_idx, self.n_idx]) / self.n_idx
        else:
            # Make sure `w` is a square matrix.
            assert w.shape[0] == w.shape[1]
            assert w.shape[0] == self.n_idx
        self.w = w

        self.replace = replace
        self.n_highest = n_highest
        self.n_worker = int(np.maximum(n_worker, 1))
        self.mask_zero = mask_zero
        self.verbose = verbose

    def generate(self, n_trial, per_query=False):
        """Generate trials.

        Generative behavior is toggled via `per_query`.

        Args:
            n_trial: A scalar indicating the number of trials to
                generate.
            per_query (optional): Boolean indicating if the provided
                `n_trial` should be interpreted as the number of trials
                per query. The default (False) means that queries and
                references are sampled to create a total of `n_trial`
                trials. If `True`, `n_trial` trials will be
                stochastically sampled for each stimulus in proportion
                to the weights on the diagonal of `w`.

        Returns:
            A RankDocket object.

        """
        # Create a contiguous array of indices.
        query_idx_list = np.arange(self.n_idx)
        # Determine eligable queries from diagonal of `w`.
        w_diag = np.diag(self.w)
        bidx = np.greater(w_diag, 0.0)
        query_idx_list = query_idx_list[bidx]
        w_diag = w_diag[bidx]

        if len(bidx) == 0:
            raise ValueError(
                "No queries are eligable. You must have some non-zero values"
                "on the diagonal of `w`."
            )

        if per_query:
            # Generate `n_trial` for each query.
            n_trial_per_query_list = np.full([len(query_idx_list)], n_trial)
        else:
            # Draw query index counts.
            w_diag = w_diag / np.sum(w_diag)
            rng = np.random.default_rng()
            n_trial_per_query_list = rng.multinomial(n_trial, w_diag)

        stimulus_set = self._uniprocess_generate(query_idx_list, n_trial_per_query_list)
        # TODO fix segfault issue with multiprocessing
        # stimulus_set = self._multiprocess_generate(
        #     query_idx_list, n_trial_per_query_list
        # )
        n_trial_total = stimulus_set.shape[0]
        n_select = np.full([n_trial_total], self.n_select)

        # Convert from contiguous indices to user-provided indices.
        stimulus_set = self.indices[stimulus_set]

        return RankDocket(stimulus_set, n_select=n_select, mask_zero=self.mask_zero)

    def _uniprocess_generate(self, query_idx_list, n_trial_per_query_list):
        """Uniprocessing strategy."""
        start_s = time()

        stimulus_set = []
        for query_idx, n_trial_per_query in zip(query_idx_list, n_trial_per_query_list):
            w_q = self.w[query_idx]
            # Set query index to zero to prohibit sampling query as reference.
            w_q[query_idx] = 0.0
            # Mask references (if any) below the specified limit.
            if self.n_highest is not None:
                ref_priority = _mask_lowest(w_q, self.n_highest)
            else:
                ref_priority = copy.copy(w_q)
            stimulus_set_q = sample_qr_sets(
                query_idx,
                self.n_reference,
                n_trial_per_query,
                ref_priority,
                replace=self.replace,
            )
            stimulus_set.append(stimulus_set_q)

        stimulus_set = np.concatenate(stimulus_set, axis=0)

        if self.verbose > 0:
            duration_s = time() - start_s
            print(
                "Docket assembly: n_trial {0} | duration {1:.0f} s".format(
                    stimulus_set.shape[0], duration_s
                )
            )

        return stimulus_set

    def _multiprocess_generate(self, query_idx_list, n_trial_per_query_list):
        """Multiprocessing setup and teardown."""
        # Create a Queue for exception handling.
        shared_exception = multiprocessing.Queue()

        # Partition the requested queries into sub-groups for consumption by
        # the worker pool.
        n_query = len(query_idx_list)
        worker_boundaries = np.linspace(0, n_query, self.n_worker + 1, dtype=int)
        start_s = time()

        with multiprocessing.Manager() as manager:
            stimulus_set_shared = manager.list([])

            process_list = []
            for idx in range(self.n_worker):
                query_idx_list_sub = query_idx_list[
                    worker_boundaries[idx] : worker_boundaries[idx + 1]
                ]
                n_trial_per_query_list_sub = n_trial_per_query_list[
                    worker_boundaries[idx] : worker_boundaries[idx + 1]
                ]
                process_list.append(
                    multiprocessing.Process(
                        target=_worker_generate,
                        args=(
                            query_idx_list_sub,
                            stimulus_set_shared,
                            self.w,
                            self.n_reference,
                            self.n_highest,
                            n_trial_per_query_list_sub,
                            self.replace,
                            shared_exception,
                        ),
                    )
                )

            for p in process_list:
                p.start()

            for p in process_list:
                p.join()

            #  Check for raised exceptions.
            e_list = [shared_exception.get() for _ in process_list]
            for e in e_list:
                if e is not None:
                    raise e

            stimulus_set = np.concatenate(stimulus_set_shared, axis=0)

        if self.verbose > 0:
            duration_s = time() - start_s
            print(
                "Docket assembly: n_trial {0} | duration {1:.0f} s".format(
                    stimulus_set.shape[0], duration_s
                )
            )

        return stimulus_set


def _worker_generate(
    query_idx_list,
    stimulus_set,
    w,
    n_reference,
    n_highest,
    n_trial_per_query_list,
    replace,
    shared_exception,
):
    """Launch worker sub-process.

    Assemble complete stimulus set for a list of pre-selected query
    indices (`query_idx_list`).

    Args:
        query_idx_list:
        stimulus_set:
        w:
        n_reference:
        n_highest:
        n_trial_per_query_list:
        replace:
        shared_exception:

    """
    try:
        stimulus_set_batch = []
        for query_idx, n_trial_per_query in zip(query_idx_list, n_trial_per_query_list):
            w_q = w[query_idx]
            # Set query index to zero to prohibit sampling query as reference.
            w_q[query_idx] = 0.0
            # Mask references (if any) below the specified limit.
            if n_highest is not None:
                ref_priority = _mask_lowest(w_q, n_highest)
            else:
                ref_priority = copy.copy(w_q)
            stimulus_set_q = sample_qr_sets(
                query_idx, n_reference, n_trial_per_query, ref_priority, replace=replace
            )
            stimulus_set_batch.append(stimulus_set_q)

        stimulus_set.extend(stimulus_set_batch)

        shared_exception.put(None)

    except Exception as e:
        tb = sys.exc_info()[2]
        shared_exception.put(e.with_traceback(tb))


def _mask_lowest(arr, n_unmasked, mask_value=0):
    """Mask lowest value entries.

    Args:
        arr: A 1D array of values.
        n_unmasked: The number of entries to leave unmasked.
        mask_value (optional): The mask value.

    Returns:
        arr_masked: A 1D array with mask applied.

    """
    # Sort highest to lowest.
    nn_idx = np.argsort(-arr)
    # Select lowest values.
    nn_idx = nn_idx[n_unmasked:]
    # Mask lowest values.
    arr_masked = copy.copy(arr)
    arr_masked[nn_idx] = mask_value
    return arr_masked
