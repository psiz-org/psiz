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
"""Module for testing utils.py."""

import pytest

import numpy as np
import psiz.utils


@pytest.fixture(scope="module")
def z0():
    """Create random set of points."""
    z0 = np.array([
        [0.46472851, 0.09534286],
        [0.90612827, 0.21031482],
        [0.46595517, 0.92022067],
        [0.51457351, 0.88226988],
        [0.24506303, 0.75287697],
        [0.69773745, 0.25095083],
        [0.71550351, 0.14846334],
        [0.24825323, 0.96021703],
        [0.85497989, 0.9114596],
        [0.35982138, 0.85040905]
    ])
    return z0


def test_simple_rotation_0(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (without scaling or reflection).
    s = np.array([[1, 0], [0, 1]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=True
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    np.testing.assert_almost_equal(z1_centered, z0_rot, decimal=2)


def test_simple_rotation_1(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (without scaling or reflection).
    s = np.array([[1, 0], [0, 1]])
    r = psiz.utils.rotation_matrix(-np.pi/2.1)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=True
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    np.testing.assert_almost_equal(z1_centered, z0_rot, decimal=2)


def test_scaled_rotation(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling).
    s = np.array([[2, 0], [0, 2]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=True
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    np.testing.assert_almost_equal(z1_centered, z0_rot, decimal=2)


def test_scaled_rotation_no_scale(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling).
    s = np.array([[2, 0], [0, 2]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=False
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    z0_rot_desired = .5 * z1_centered
    np.testing.assert_almost_equal(z0_rot_desired, z0_rot, decimal=2)


def test_x_reflection_rotation(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling and reflection).
    s = np.array([[-1, 0], [0, 1]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=True
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    np.testing.assert_almost_equal(z1_centered, z0_rot, decimal=2)


def test_y_reflection_rotation(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling and reflection).
    s = np.array([[1, 0], [0, -1]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=True
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    np.testing.assert_almost_equal(z1_centered, z0_rot, decimal=2)


def test_xy_reflection_rotation(z0):
    """Test procrustean solution for simple problem."""
    # Assemble rotation matrix (with scaling and reflection).
    s = np.array([[-1, 0], [0, -1]])
    r = psiz.utils.rotation_matrix(np.pi/4)
    rs = np.matmul(s, r)

    # Center `z0`.
    z0_centered = z0 - np.mean(z0, axis=0, keepdims=True)

    # Apply rotation to centered `z0` data.
    z1 = np.matmul(z0_centered, rs)
    z1_centered = z1

    # Attempt to recover original set of points.
    r_recov = psiz.utils.procrustes_rotation(
        z0, z1, scale=True
    )
    z0_rot = np.matmul(z0_centered, r_recov)

    np.testing.assert_almost_equal(z1_centered, z0_rot, decimal=2)
