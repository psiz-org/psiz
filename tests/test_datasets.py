# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
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

"""Module for testing `datasets.py`.

Todo:
    * Test Catalog creation, save, and load.

"""

import pytest
import numpy as np

from psiz import datasets


class TestCatalog:
    """Test class Catalog."""

    def test_initialization(self):
        """Test initialization of class."""
        # Normal.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg']
        catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        np.testing.assert_array_equal(catalog.stimuli.id.values, stimulus_id)
        np.testing.assert_array_equal(
            catalog.stimuli.filepath.values, stimulus_filepath)
        assert catalog.n_stimuli == 6

        # Bad input shape.
        stimulus_id = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_id` must be a 1D array of integers.')

        # A non-integer stimulus_id value.
        stimulus_id = np.array([0, 1, 2., 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_id` must be a 1D array of integers.')

        # Zero stimulus_id not present.
        stimulus_id = np.array([1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_id` must contain a contiguous set of '
            'integers [0, n_stimuli[.')

        # Two stimulus_id's not present.
        stimulus_id = np.array([0, 1, 2, 5])
        stimulus_filepath = ['r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/f.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_id` must contain a contiguous set of '
            'integers [0, n_stimuli[.')

        # Bad shape.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            ['r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg'],
            ['r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg']]
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_filepath` must have the same shape as '
            '`stimulus_id`.')

        # Mismatch in number (too few).
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_filepath` must have the same shape as '
            '`stimulus_id`.')

        # Mismatch in number (too many).
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = [
            'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg',
            'f/g.jpg']
        with pytest.raises(Exception) as e_info:
            catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        assert str(e_info.value) == (
            'The argument `stimulus_filepath` must have the same shape as '
            '`stimulus_id`.')

        # Must be a list.
        # stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        # stimulus_filepath = np.array([
        #     'r/a.jpg', 'r/b.jpg', 'r/c.jpg', 'r/d.jpg', 'r/e.jpg', 'r/f.jpg',
        #     'f/g.jpg'])

        # A non-string stimlus_path value.
        # stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        # stimulus_filepath = np.array([0, 1, 2, 3, 4, 5])
        # with pytest.raises(Exception) as e_info:
        #     catalog = datasets.Catalog(stimulus_id, stimulus_filepath)

    def test_persistence(self, tmpdir):
        """Test object persistence."""
        # Create Catalog object.
        stimulus_id = np.array([0, 1, 2, 3, 4, 5])
        stimulus_filepath = np.array(
            ['r/a', 'r/b12', 'r/c', 'r/d', 'r/e', 'r/f'], dtype='S10')
        catalog = datasets.Catalog(stimulus_id, stimulus_filepath)
        # Save Catalog.
        fn = tmpdir.join('catalog_test.hdf5')
        catalog.save(fn)
        # Load the saved catalog.
        loaded_catalog = datasets.load_catalog(fn)
        # Check that the loaded Docket object is correct.
        assert catalog.n_stimuli == loaded_catalog.n_stimuli
        np.testing.assert_array_equal(
            catalog.stimuli.id.values,
            loaded_catalog.stimuli.id.values)
        np.testing.assert_array_equal(
            catalog.stimuli.filepath.values,
            loaded_catalog.stimuli.filepath.values)
