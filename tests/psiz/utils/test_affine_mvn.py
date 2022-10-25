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
"""Module for testing utils.py."""

import numpy as np

from psiz.utils import affine_mvn, rotation_matrix


def test_defaults():
    """Test default arguments."""
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    loc1, cov1 = affine_mvn(loc0, cov0)

    desired_loc = loc0
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = cov0
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_translation():
    """Test translation."""
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = np.array([
        [1., 0.],
        [0., 1.]
    ])
    t = np.array([.1, .1])
    loc1, cov1 = affine_mvn(loc0, cov0, r, t)

    desired_loc = np.array([.3, -.1])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_rotation_0():
    """Test rotation matrix.

    Mirror x-coordinate.

    """
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = np.array([
        [-1., 0.],
        [0., 1.]
    ])
    loc1, cov1 = affine_mvn(loc0, cov0, r)

    desired_loc = np.array([-.2, -.2])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_rotation_1():
    """Test rotation matrix.

    Mirror through origin.

    """
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = np.array([
        [-1., 0.],
        [0., -1.]
    ])
    loc1, cov1 = affine_mvn(loc0, cov0, r)

    desired_loc = np.array([-.2, .2])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_rotation_2():
    """Test rotation matrix.

    Rotate pi radians (clockwise).

    """
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = rotation_matrix(np.pi)
    loc1, cov1 = affine_mvn(loc0, cov0, r)

    desired_loc = np.array([-.2, .2])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_rotation_3():
    """Test rotation matrix.

    Rotate pi/2 radians (clockwise).

    """
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = rotation_matrix(np.pi / 2)
    loc1, cov1 = affine_mvn(loc0, cov0, r)

    desired_loc = np.array([-.2, -.2])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1.2, 0.],
        [0., 1.]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_rotation_plus_translation():
    """Test rotation matrix.

    Rotate pi/2 radians (clockwise).

    """
    loc0 = np.array([0.2, -0.2])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = rotation_matrix(np.pi / 2)
    t = np.array([.1, .1])
    loc1, cov1 = affine_mvn(loc0, cov0, r, t)

    desired_loc = np.array([-.1, -.1])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1.2, 0.],
        [0., 1.]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)


def test_rotation_3_w_singleton_dims():
    """Test rotation matrix.

    Rotate pi/2 radians (clockwise).
    Use singleton dimensions for `loc0`.

    """
    loc0 = np.array([[0.2, -0.2]])
    cov0 = np.array([
        [1., 0.],
        [0., 1.2]
    ])
    r = rotation_matrix(np.pi / 2)
    loc1, cov1 = affine_mvn(loc0, cov0, r)

    desired_loc = np.array([[-.2, -.2]])
    np.testing.assert_array_almost_equal(loc1, desired_loc)

    desired_cov = np.array([
        [1.2, 0.],
        [0., 1.]
    ])
    np.testing.assert_array_almost_equal(cov1, desired_cov)

    # Test with non-singleton translation vector which should
    # broadcast correctly.
    t = np.array([0., 0.])
    loc1, cov1 = affine_mvn(loc0, cov0, r, t)
    np.testing.assert_array_almost_equal(loc1, desired_loc)
