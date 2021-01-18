
# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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
"""Module of utility functions.

Classes:
    ProgressBarRe: A progress bar displayed in the terminal.

Functions:
    affine_mvn: Affine transformation of multivariate normal
        distribution.
    pairwise_matrix: Return the similarity matrix characterizing
        the embedding.
    matrix_comparison: Compute correlation between two matrices.
    compare_models: Compare the similarity structure between two
        embedding models.
    rotation_matrix: Returns a two-dimensional rotation matrix.
    procrustes_2d: Attempt to allign two sets of 2D points by finding
        an affine transformation.
    choice_wo_replace: Efficient sampling without replacement.
    standard_split: Standard 80-10-10 split of observations.
    pad_2d_array: Pad a 2D array with a value.

"""

import copy
import datetime
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


def affine_mvn(loc, cov, r=None, t=None):
    """Affine transformation of multivariate normal.

    Performs the following operations:
        loc_affine = loc @ r + t
        cov_affine = r^T @ cov @ r

    Arguments:
        loc: Location parameters.
            shape=(1, n_dim)
        cov: Covariance.
            shape=(n_dim, n_dim)
        r: Rotation matrix.
            shape=(n_dim, n_dim)
        t: Transformation vector.
            shape=(1, n_dim)

    Returns:
        loc_affine: Rotated location parameters.
        cov_affine: Rotated covariance.

    NOTE:
        This implementation hits the means with a rotation matrix on
        the RHS, allowing the rows to correspond to an instance and
        columns to correspond to dimensionality. The more conventional
        pattern has rows corresponding to dimensionality, in which case
        the code would be implemented as:

        ```
        loc_affine = np.matmul(r, loc) + t
        cov_affine = np.matmul(r, np.matmul(cov, np.transpose(r)))
        ```

    """
    n_dim = loc.shape[0]

    if t is None:
        t = 0
    if r is None:
        r = np.eye(n_dim)

    loc_affine = np.matmul(loc, r) + t
    cov_affine = np.matmul(np.transpose(r), np.matmul(cov, r))
    return loc_affine, cov_affine


def assess_convergence(
        obs, model, n_stimuli, n_dim, n_partition=10, n_back=2, n_shuffle=3,
        n_restart=50, score='r2', verbose=0):
    """Evaluate if a sufficient number of observations have been collected.

    In general, more observations improve model inference. However,
    there are generally diminishing returns to collecting more data.
    This function determines the impact of adding a little bit more
    data. Ideally, you should have sufficient data that adding more
    does not dramatically alter the structure of the inferred
    embedding. Structural changes are assessed by comparing the pair-
    wise similarity matrix of different embeddings.

    Arguments:
        obs: A RankObservations object.
        model: A PsychologicalEmbedding object.
        n_stimuli: The number of stimuli.
        n_dim: The dimensionality of the embedding.
        n_partition (optional): The number of partitions.
        n_back (optional):  The number of partitions to evaluate. Can
            be [1, n_partition-1].
        n_shuffle (optional): The number of times to shuffle and
            repeat tha analysis.
        n_restart (optional): The number of restarts to use when
            fitting the embeddings.
        score (optional): Measure to use when comparing two similarity
            matrices.
        verbose (optional): Verbosity.

    Returns:
        summary: Dictionary of results.
            n_trial_array: Number of trials in each set.

    """
    # TODO CRITICAL
    # Check arguments.
    n_back = np.maximum(n_back, 1)
    n_back = np.minimum(n_back, n_partition-1)

    n_trial_array = np.linspace(0, obs.n_trial, n_partition + 1)
    n_trial_array = np.ceil(n_trial_array[1:]).astype(int)

    val = np.ones([n_shuffle, n_partition]) * np.nan
    for i_shuffle in range(n_shuffle):
        if verbose > 0:
            print('  Shuffle {0}'.format(i_shuffle + 1))
        # Randomize observations.
        rand_idx = np.random.permutation(obs.n_trial)
        obs = obs.subset(rand_idx)

        # Infer embeddings for an increasing number of observations.
        first_part = True
        emb_list = [None] * n_partition
        for i_part in range(n_partition-n_back-1, n_partition):
            include_idx = np.arange(0, n_trial_array[i_part])
            curr_obs = obs.subset(include_idx)
            curr_emb = model(n_stimuli, n_dim=n_dim)
            curr_emb.fit(curr_obs, n_restart=n_restart, verbose=verbose)
            emb_list[i_part] = curr_emb

            if not first_part:
                simmat_0 = pairwise_matrix(
                    emb_list[i_part - 1].similarity,
                    emb_list[i_part - 1].z
                )
                simmat_1 = pairwise_matrix(
                    emb_list[i_part].similarity, emb_list[i_part].z
                )
                val[i_shuffle, i_part] = matrix_comparison(
                    simmat_0, simmat_1, score=score
                )
                if verbose > 0:
                    print('    {0} | measure: {1:.2f}'.format(
                        i_part, val[i_shuffle, i_part]
                    ))
            else:
                first_part = False

    return {"n_trial_array": n_trial_array, "val": val, "measure": score}


def pairwise_matrix(kernel_fn, z):
    """Return a pairwise similarity matrix.

    Arguments:
        kernel_fn: A kernel function.
        z: An embedding.

    Returns:
        A 2D array where element s_{i,j} indicates the similarity
            between the ith and jth stimulus.

    """
    n_stimuli = z.shape[0]

    xg = np.arange(n_stimuli)
    a, b = np.meshgrid(xg, xg)
    a = a.flatten()
    b = b.flatten()

    z_a = z[a, :]
    z_b = z[b, :]
    pmat = kernel_fn(z_a, z_b)
    pmat = pmat.reshape(n_stimuli, n_stimuli)
    return pmat


def matrix_comparison(mat_a, mat_b, score='r2', elements='upper'):
    """Return a comparison score between two square matrices.

    Arguments:
        mat_a: A square matrix.
        mat_b: A square matrix the same size as mat_a
        score (optional): The type of comparison to use.
        elements (optional): Which elements to use in the computation.
            The options are upper triangular elements (upper), lower
            triangular elements (lower), or off-diagonal elements
            (off).

    Returns:
        The comparison score.

    Notes:
        'r2'
        When computing R^2 values of two similarity matrices, it is
        assumed, by definition, that the corresponding diagonal
        elements are the same between the two matrices being compared.
        Therefore, the diagonal elements are not included in the R^2
        computation to prevent inflating the R^2 value. On a similar
        note, including both the upper and lower triangular portion
        does not artificially inflate R^2 values for symmetric
        matrices.

    """
    n_row = mat_a.shape[0]
    idx_upper = np.triu_indices(n_row, 1)
    idx_lower = np.triu_indices(n_row, 1)
    if elements == 'upper':
        idx = idx_upper
    elif elements == 'lower':
        idx = idx_lower
    elif elements == 'off':
        idx = (
            np.hstack((idx_upper[0], idx_lower[0])),
            np.hstack((idx_upper[1], idx_lower[1])),
        )
    else:
        raise ValueError(
            'The argument to `elements` must be "upper", "lower", or "off".')

    if score == 'pearson':
        score, _ = pearsonr(mat_a[idx], mat_b[idx])
    elif score == 'r2':
        rho, _ = pearsonr(mat_a[idx], mat_b[idx])
        score = rho**2
        # score = r2_score(mat_a[idx], mat_b[idx])
    elif score == 'mse':
        score = np.mean((mat_a[idx] - mat_b[idx])**2)
    else:
        raise ValueError(
            'The provided `score` argument is not valid.')
    return score


def compare_models(
        model_a, model_b, group_id_a=0, group_id_b=0, score='r2',
        elements='upper'):
    """Compare two psychological embeddings.

    Arguments:
        model_a:  A psychological embedding model.
        model_b:  A psychological embedding model.
        group_id_a (optional):  A particular group ID to use when
            computing the similarity matrix for model_a.
        group_id_b (optional):  A particular group ID to use when
            computing the similarity matrix for model_b.

    Returns:
        R^2 value of the two similarity matrices derived from each
            embedding.

    """
    def sim_func_a(z_q, z_ref):
        return model_a.similarity(z_q, z_ref, group_id=group_id_a)

    def sim_func_b(z_q, z_ref):
        return model_b.similarity(z_q, z_ref, group_id=group_id_b)

    simmat_a = pairwise_matrix(sim_func_a, model_a.z)
    simmat_b = pairwise_matrix(sim_func_b, model_b.z)

    r_squared = matrix_comparison(
        simmat_a, simmat_b, score=score, elements=elements
    )
    return r_squared


def rotation_matrix(theta):
    """Return 2D rotation matrix.

    Arguments:
        theta: Scalar value indicating radians of rotation.

    """
    return np.array((
        (np.cos(theta), -np.sin(theta)),
        (np.sin(theta), np.cos(theta)),
    ))


def procrustes_2d(z0, z1, n_restart=10, scale=True, seed=None):
    """Perform Procrustes superimposition.

    Align two sets of coordinates using an affine transformation.

    Attempts to find the affine transformation (composed of a rotation
    matrix `r` and a transformation vector `t`) for `z1` such that
    `z1_affine` closely matches `z0`. Closeness is measured using MSE.

        z1_affine = np.matmul(z1, r) + t

    This algorithm only works with 2D coordinates (i.e., n_dim=2).

    Arguments:
        z0: The first set of points.
            shape = (n_point, n_dim)
        z1: The second set of points.
            shape = (n_point, n_dim)
        n_restart (optional): A scalar indicating the number of
            restarts for the optimization routine.
        scale (optional): Boolean indicating if scaling is permitted
            in the affine transformation. By default scaling is
            allowed, generating a full Procrustes superimposition. Set
            to false to yield a partial Procrustes superimposition.
        seed (optional): Seed for random number generator.

    Returns:
        r: A rotation matrix.
            shape=(n_dim, n_dim)
        t: A transformation vector.
            shape=(1, n_dim)

    """
    # Settings
    n_dim = 2
    flip_list_2d = [
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ]

    # Center matrices.
    z0 = z0 - np.mean(z0, axis=0, keepdims=True)
    z1 = z1 - np.mean(z1, axis=0, keepdims=True)

    # Scale for stability
    stability_scaling = 10. / (np.max(z0) - np.min(z0))
    z0 = stability_scaling * z0
    z1 = stability_scaling * z1

    if seed is not None:
        np.random.seed(seed)

    def assemble_r_t(params):
        """Assemble.

        Arguments
            params:
                trans_x: Translation in x.
                trans_y: Translation in y.
                s: Scaling factor.
                r: Rotation matrix, in terms of theta radians.
                f: Flip (reflection).

        Returns:
            fsr: Compined flip, scale, and rotation matrix.
                shape=(2, 2)
            t: A translation vector.
                shape=(1, 2)

        """
        # Convert scale, rotation and flip parameters to matrices.
        s = params[3] * np.eye(n_dim)
        r = rotation_matrix(params[2])
        f = np.array([[np.sign(params[4]), 0], [0, np.sign(params[5])]])

        # Combined scale, rotation, and flip into one matrix.
        fsr = np.matmul(f, np.matmul(s, r))

        # Convert translation parameters to vector.
        t = np.array([params[0], params[1]])
        t = np.expand_dims(t, axis=0)

        return fsr, t

    # In order to avoid impossible rotation matrices, perform optimization
    # on rotation components separately (theta, scaling, mirror).
    def objective_fn(params, z0, z1):
        if np.any(np.isnan(params)):
            loss = -np.inf
        else:
            r, t = assemble_r_t(params)
            # Apply affine transformation.
            z1_affine = np.matmul(z1, r) + t
            loss = np.mean(
                    np.sum(
                        (z0 - z1_affine)**2, axis=1
                    )
            )
            # Could also define loass as MAE, since MSE may chase
            # outliers.
            # loss = np.mean(np.sum(np.abs(z0 - z1_affine), axis=1))

        # print(
        #     'loss: {3} | {0:.2g} {1:.2g} {2:.2g}'.format(
        #         params[0], params[1], params[2], loss
        #     )
        # )

        return loss

    # t_0, t_1, theta, scaling, flip
    params_best = np.array((0., 0., 0., 1.))
    loss_best = np.inf
    for _ in range(n_restart):
        # Perform a restart for each possible axis flip combination.
        for flip in flip_list_2d:
            # Make initial translation vector one that aligns the two
            # centroids.
            (tr_x0, tr_y0) = np.mean(z1, axis=0) - np.mean(z0, axis=0)
            theta0 = 2 * np.pi * np.random.rand(1)[0]
            if scale:
                s0 = np.random.rand(1)[0] + .5
                s_bnds = (0., None)
            else:
                s0 = 1
                s_bnds = (1., 1.)
            fx0 = flip[0]
            fy0 = flip[1]

            params0 = np.array((tr_x0, tr_y0, theta0, s0, fx0, fy0))
            bnds = (
                (None, None),
                (None, None),
                (0., 2.01 * np.pi),
                s_bnds,
                (fx0, fx0),
                (fy0, fy0)
            )
            res = minimize(
                objective_fn, params0, args=(z0, z1), bounds=bnds
                # method='SLSQP'
                # method='L-BFGS-B'
                )
            params_candidate = res.x
            loss_candidate = res.fun
            if loss_candidate < loss_best:
                loss_best = loss_candidate
                params_best = params_candidate

    r, t = assemble_r_t(params_best)
    return r, t/stability_scaling


def procrustes_rotation(z0, z1, scale=True):
    """Perform Procrustes superimposition.

    Align two sets of coordinates (`z0` and `z1`) by finding the
    optimal rotation matrix `r` that rotates `z0` into `z1`. Both `z0`
    and `z1` are centered first.

    `z0_rot = z0 @ r`

    Arguments:
        z0: The first set of points.
            shape = (n_point, n_dim)
        z1: The second set of points. The data matrices `z0` and `z1`
            must have the same shape.
            shape = (n_point, n_dim)
        n_restart (optional): A scalar indicating the number of
            restarts for the optimization routine.
        scale (optional): Boolean indicating if scaling is permitted
            in the affine transformation. By default scaling is
            allowed, generating a full Procrustes superimposition. Set
            to false to yield a partial Procrustes superimposition.

    Returns:
        r: A rotation matrix that operates on the *centered* data.
            shape=(n_dim, n_dim)

    """
    n_dim = z0.shape[1]

    # Ensure data matrices are centered.
    z0 = z0 - np.mean(z0, axis=0, keepdims=True)
    z1 = z1 - np.mean(z1, axis=0, keepdims=True)

    # Compute cross-covariance matrix.
    m = np.matmul(np.transpose(z0), z1)

    # Compute SVD of covariance matrix.
    # NOTE: h = u @ np.diag(s) @ vh = (u * s) @ vh
    u, s, vh = np.linalg.svd(m, hermitian=False)

    # Aseemble rotation matrix (does not include scaling).
    r = u @ vh

    if scale:
        # Determine scaling factor.
        z0_rot = np.matmul(z0, r)
        norm_z0 = np.sqrt(np.sum((z0_rot**2), axis=1))
        norm_z1 = np.sqrt(np.sum((z1**2), axis=1))
        scale_factor = norm_z1 / norm_z0
        scale_factor = np.mean(scale_factor)
        r = r @ (np.eye(n_dim) * scale_factor)

    return r


def choice_wo_replace(a, size, p):
    """Fast sampling without replacement.

    Arguments:
        a: An array indicating the eligable elements.
        size: A tuple indicating the number of independent samples and
            the number of draws (without replacement) for each sample.
            The tuple is ordered such that
            size = (n_sample, sample_size).
        p: An array indicating the probabilites associated with drawing
            a particular element. User provided probabilities are
            already assumed to sum to one. Probability p[i] indicates
            the probability of drawing index a[i].

    Returns:
        result: A 2D array containing the drawn elements.
            shape=(n_sample, sample_size)

    See: https://medium.com/ibm-watson/
        incredibly-fast-random-sampling-in-python-baf154bd836a

    """
    n_sample = size[0]
    sample_size = size[1]

    # Replicate probabilities as many times as `n_sample`
    replicated_probabilities = np.tile(p, (n_sample, 1))

    # Get random shifting numbers & scale them correctly.
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]

    # Shift by numbers and find largest (by finding the smallest of the
    # negative).
    shifted_probabilities = random_shifts - replicated_probabilities
    samples = np.argpartition(
        shifted_probabilities, sample_size, axis=1
    )[:, :sample_size]

    return a[samples]


def standard_split(obs, shuffle=False, seed=None):
    """Creata a standard 80-10-10 split of the observations.

    Arguments:
        obs: A set of observations.
        shuffle (optional): Boolean indicating if the data should be
            shuffled before splitting.
        seed: Integer to seed randomness.

    Returns:
        obs_train: A train set (80%).
        obs_val: A validation set (10%).
        obs_test: A test set (10%).

    """
    skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=seed)
    (train_idx, holdout_idx) = list(
        skf.split(obs.stimulus_set, obs.config_idx)
    )[0]
    obs_train = obs.subset(train_idx)
    obs_holdout = obs.subset(holdout_idx)
    skf = StratifiedKFold(n_splits=2, shuffle=shuffle, random_state=seed)
    (val_idx, test_idx) = list(
        skf.split(obs_holdout.stimulus_set, obs_holdout.config_idx)
    )[0]
    obs_val = obs_holdout.subset(val_idx)
    obs_test = obs_holdout.subset(test_idx)
    return obs_train, obs_val, obs_test


def pad_2d_array(arr, n_column, value=-1):
    """Pad 2D array with columns composed of -1.

    Argument:
        arr: A 2D array denoting the stimulus set.
        n_column: The total number of columns that the array should
            have.
        value (optional): The value to use to pad the array.

    Returns:
        Padded array.

    """
    n_trial = arr.shape[0]
    n_pad = n_column - arr.shape[1]
    if n_pad > 0:
        pad_mat = value * np.ones((n_trial, n_pad), dtype=np.int32)
        arr = np.hstack((arr, pad_mat))
    return arr


class ProgressBarRe(object):
    """Display a progress bar in terminal."""

    def __init__(
            self, total, prefix='', decimals=1, length=100,
            fill='█'):
        """Initialize.

        Arguments:
            iteration: Integer indicating current iteration.
            total: Integer indicating total iterations.
            prefix (optional): String that is used as prefix.
            suffix (optional): String that is used as suffix.
            decimals (optional): Integer indicating a positive number
                of decimals in percent complete.
            length (optional): Integer indicating the character length
                of the progress bar.
            fill (optional): String indicating the bar fill character.

        """
        self.total = total
        self.prefix = prefix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.n_call = 0
        self.start_s = 0
        self.total_s = 0

    def _start(self):
        """Start timer."""
        self.start_s = time.time()

    def _stop(self):
        """Stop time."""
        self.total_s = time.time() - self.start_s

    def update(self, iteration):
        """Update progress bar to display current iteration."""
        # Start time if this is the first call.
        if self.n_call == 0:
            self._start()
        self.n_call = self.n_call + 1

        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (iteration / float(self.total))
        )

        elapsed_time = time.time() - self.start_s
        if iteration == 0:
            time_per_iter = 0.0
        else:
            time_per_iter = elapsed_time / iteration

        eta_s = np.round((self.total - iteration) * time_per_iter)
        ett_s = np.round(self.total * time_per_iter)

        eta_str = str(datetime.timedelta(seconds=eta_s))
        ett_str = str(datetime.timedelta(seconds=ett_s))

        filledLength = int(self.length * iteration // self.total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print(
            '\r    {0} |{1}| {2}% | ETA: {3} | ETT: {4}'.format(
                self.prefix, bar, percent, eta_str, ett_str
            ), end='\r'
        )
        # Print a new line on completion.
        if iteration == self.total:
            self._stop()
            print()


def generate_group_matrix(n_row, group_idx=[]):
    """Generate group ID data structure.

    The first column is reserved and is composed of all zeros.
    Additional columns are optional and determined by the user.

    Arguments:
        n_row: The number of rows.
        group_idx (optional): Array-like integers indicating
            hierarchical group index information. For example, `[4,3]`
            indicates that the first hierarchical level has `group_id`
            4 and the second hierarchical level has `group_id` 3.

    Returns:
        group_matrix: The completed group matrix where each column
            corresponds to a group level and each row a datum.
            shape=(n_row,1+len(group_id))

    """
    group_matrix = np.hstack(([0], group_idx))
    group_matrix = np.expand_dims(group_matrix, axis=0)
    group_matrix = np.repeat(group_matrix, n_row, axis=0)
    return group_matrix


# TODO add tests
def pairwise_index_dataset(
        n_data, batch_size=None, elements='upper', group_idx=[],
        mask_zero=False, subsample=None, seed=252):
    """Assemble pairwise combinations.

    Arguments:
        n_data: A scalar indicating the number of unique concepts.
        batch_size (optional): Determines the batch size. By default,
            the batch size will be set to the number of combinations.
        elements (optional): Determines which combinations in the pairwise
            matrix will be used. Can be one of 'all', 'upper', 'lower',
            or 'off'.
        group_idx (optional): Array-like integers indicating
            hierarchical group index information. For example, `[4,3]`
            indicates that the first hierarchical level has `group_id`
            4 and the second hierarchical level has `group_id` 3.
        mask_zero (optional): Whether the zero index should be masked
            and all indices incremented by one.
        subsample: A float ]0,1] indicating the proportion of all pairs
            that should be retained. By default all pairs are retained.
        seed: Integer controlling which pairs are subsampled.

    Returns:
        Tensorflow Dataset.

    """
    if elements == 'all':
        idx = np.meshgrid(np.arange(n_data), np.arange(n_data))
        idx_0 = idx[0].flatten()
        idx_1 = idx[1].flatten()
    elif elements == 'upper':
        idx = np.triu_indices(n_data, 1)
        idx_0 = idx[0]
        idx_1 = idx[1]
    elif elements == 'lower':
        idx = np.triu_indices(n_data, 1)
        idx_0 = idx[0]
        idx_1 = idx[1]
    elif elements == 'off':
        idx_upper = np.triu_indices(n_data, 1)
        idx_lower = np.triu_indices(n_data, 1)
        idx = (
            np.hstack((idx_upper[0], idx_lower[0])),
            np.hstack((idx_upper[1], idx_lower[1])),
        )
        idx_0 = idx[0]
        idx_1 = idx[1]
    del idx

    n_pair = len(idx_0)
    if subsample is not None:
        np.random.seed(seed)
        idx_rand = np.random.permutation(n_pair)
        n_pair = int(np.ceil(n_pair * subsample))
        idx_rand = idx_rand[0:n_pair]
        idx_0 = idx_0[idx_rand]
        idx_1 = idx_1[idx_rand]

    if mask_zero:
        idx_0 = idx_0 + 1
        idx_1 = idx_1 + 1

    idx_0 = tf.constant(idx_0, dtype=tf.int32)
    idx_1 = tf.constant(idx_1, dtype=tf.int32)

    if batch_size is None:
        batch_size = np.minimum(10000, n_pair)

    ds_info = {
        'n_pair': n_pair,
        'batch_size': batch_size,
        'n_batch': np.ceil(n_pair / batch_size),
        'elements': elements,
        'mask_zero': mask_zero
    }

    group_matrix = generate_group_matrix(n_pair, group_idx=group_idx)
    group_matrix = tf.constant(group_matrix, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        ((idx_0, idx_1, group_matrix))
    ).batch(
        batch_size, drop_remainder=False
    )
    return ds, ds_info


def pairwise_similarity(stimuli, kernel, ds_pairs):
    """Return the similarity between stimulus pairs.

    Arguments:
        stimuli: A psiz.keras.layers.Stimuli object.
        kernel: A psiz.keras.layers.Kernel object.
        ds_pairs: A TF dataset object that yields a 3-tuple composed
            of stimulus index i, sitmulus index j, and group
            membership indices.

    Returns:
        s: A tf.Tensor of similarities between stimulus i and stimulus
            j (using the requested group-level parameters from the
            stimuli layer and the kernel layer).
            shape=([sample_size,] n_pair)

    Notes:
        The `n_sample` property of the Stimuli layer and Kernel layer
            must agree.

    """
    s = []
    for x_batch in ds_pairs:
        z_0 = stimuli([x_batch[0], x_batch[2]])
        z_1 = stimuli([x_batch[1], x_batch[2]])
        s.append(
            kernel([z_0, z_1, x_batch[2]])
        )

    # Concatenate along pairs dimension (i.e., the last dimension).
    s = tf.concat(s, axis=-1)
    return s
