# -*- coding: utf-8 -*-
# Copyright 2023 The PsiZ Authors. All Rights Reserved.
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

"""Data module.

Classes:
    WeightedRank

"""

import copy
import multiprocessing
import sys
from time import time

import numpy as np

from psiz.data.generators.rank_generator import RankGenerator
from psiz.data.sample_qr_sets import sample_qr_sets
from psiz.data.contents.rank import Rank


class WeightedRank(RankGenerator):
    """Generator that generates Rank content.

    Users can guide the sampling process by providing a matrix of
    weights. The generative process can be distributed across multiple
    workers.

    NOTE: When sampling *without* replacement (i.e., replace=False) you
    must be mindful of the number of unique samples possible. If
    requesting more unique samples than possible, you may be surprised
    when fewer samples are returned.

    """

    def __init__(self, n_worker=1, **kwargs):
        """Initialize.

        Args:
            n_worker (optional): The number of unique workers (CPUs) to
                divide the work amongst. By default, only one worker is
                used.

        """
        super(WeightedRank, self).__init__(**kwargs)
        self.n_worker = int(np.maximum(n_worker, 1))

    def generate(
        self, n_sample, sequence_length=1, w=None, per_query=False, replace=True
    ):
        """Generate samples.

        Generative behavior is toggled via `per_query`.

        Args:
            n_sample: A scalar indicating the number of samples to
                generate.
            sequence_length (optional): The length of a sample
                sequence. By default, every sample has a sequence
                length of 1.
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
                elements are given equal weight.
                shape=(n_element, n_element)
            per_query (optional): Boolean indicating if the provided
                `n_sample` should be interpreted as the number of
                samples per query. The default (False) means that
                queries and references are sampled to create a total of
                `n_sample` samples. If `True`, `n_sample` samples will
                be stochastically sampled for each stimulus in
                proportion to the weights on the diagonal of `w`.
            replace (optional): Boolean indicating if samples should be
                unique. The default `replace=True` does not
                guarantee unique samples.

        Returns:
            A `psiz.data.Rank` object.

        """
        # Make sure `w` is a square matrix.
        w = self._validate_w(w)

        # Determine eligible queries from diagonal of `w`.
        query_idx_list = self.element_indices
        w_diag = np.diag(w)
        bidx = np.greater(w_diag, 0.0)
        query_idx_list = query_idx_list[bidx]
        w_diag = w_diag[bidx]

        if len(bidx) == 0:
            raise ValueError(
                "No queries are eligible. You must provide an alternative `w` to create eligible queries."
            )

        if per_query:
            # Generate `n_sample` for each query.
            n_sample_per_query_list = np.full([len(query_idx_list)], n_sample)
        else:
            # Draw query index counts.
            w_diag = w_diag / np.sum(w_diag)
            rng = np.random.default_rng()
            n_sample_per_query_list = rng.multinomial(n_sample, w_diag)

        if self.n_worker > 1:
            # TODO
            raise NotImplementedError(
                "Multi-worker functionality is not yet implemented."
            )
            stimulus_set = self._multiprocess_generate(
                w, replace, query_idx_list, n_sample_per_query_list
            )
        else:
            stimulus_set = _worker_generate(
                query_idx_list,
                copy.copy(w),
                self.n_reference,
                n_sample_per_query_list,
                replace,
            )
        # Convert to raw elements.
        stimulus_set = self.raw_elements[stimulus_set]
        content = Rank(stimulus_set, n_select=self.n_select)
        return content

    def _validate_w(self, w):
        """Validate argument `w`.

        Make sure provided weight matrix is valid. Instantiate uniform
        weight matrix if none provided.

        Args:
            w: A 2D array of weights.

        Returns
            w

        """
        if w is None:
            w = np.ones([self.n_element, self.n_element]) / self.n_element
        else:
            if w.shape[0] != w.shape[1]:
                raise ValueError("The argument `w` must be a square matrix.")
            if w.shape[0] != self.n_element:
                raise ValueError(
                    "The shape of `w` must agree with the number of elements "
                    "provided."
                )
        return w

    def _multiprocess_generate(
        self, w, replace, query_idx_list, n_sample_per_query_list
    ):
        """Multiprocessing setup and teardown."""
        # Create a Queue for exception handling.
        shared_exception = multiprocessing.Queue()

        # Partition the requested queries into subsets to be allocated to
        # individual workers in the worker pool.
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
                n_sample_per_query_list_sub = n_sample_per_query_list[
                    worker_boundaries[idx] : worker_boundaries[idx + 1]
                ]
                process_list.append(
                    multiprocessing.Process(
                        target=_worker_wrapper,
                        args=(
                            query_idx_list_sub,
                            stimulus_set_shared,
                            w,
                            self.n_reference,
                            n_sample_per_query_list_sub,
                            replace,
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
                "Docket assembly: n_sample {0} | duration {1:.0f} s".format(
                    stimulus_set.shape[0], duration_s
                )
            )

        return stimulus_set


def _worker_wrapper(
    query_idx_list,
    stimulus_set,
    w,
    n_reference,
    n_sample_per_query_list,
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
        n_sample_per_query_list:
        replace:
        shared_exception:

    """
    try:
        stimulus_set_batch = []
        for query_idx, n_sample_per_query in zip(
            query_idx_list, n_sample_per_query_list
        ):
            w_q = w[query_idx]
            # Set query index to zero to prohibit sampling query as reference.
            w_q[query_idx] = 0.0
            ref_priority = copy.copy(w_q)
            stimulus_set_q = sample_qr_sets(
                query_idx,
                n_reference,
                n_sample_per_query,
                ref_priority,
                replace=replace,
            )
            stimulus_set_batch.append(stimulus_set_q)

        stimulus_set.extend(stimulus_set_batch)

        shared_exception.put(None)

    except Exception as e:
        tb = sys.exc_info()[2]
        shared_exception.put(e.with_traceback(tb))


def _worker_generate(
    query_idx_list,
    w,
    n_reference,
    n_sample_per_query_list,
    replace,
):
    """Launch worker sub-process.

    Assemble complete stimulus set for a list of pre-selected query
    indices (`query_idx_list`).

    Args:
        query_idx_list:
        w:
        n_reference:
        n_sample_per_query_list:
        replace:

    """
    stimulus_set_batch = []
    for query_idx, n_sample_per_query in zip(query_idx_list, n_sample_per_query_list):
        w_q = w[query_idx]
        # Set query index to zero to prohibit sampling query as reference.
        w_q[query_idx] = 0.0
        ref_priority = copy.copy(w_q)
        stimulus_set_q = sample_qr_sets(
            query_idx, n_reference, n_sample_per_query, ref_priority, replace=replace
        )
        stimulus_set_batch.append(stimulus_set_q)
    return np.concatenate(stimulus_set_batch, axis=0)
