#############
PsiZ Datasets
#############

:Author: Brett D. Roads

Overview
========

A small number of PsiZ-compatible datasets are hosted at the OSF repository `psiz-datasets <https://osf.io/cn2s3/>`_. These datasets can be obtained in raw form---where each trial sequence is stored as a separate JSON file inside a zipped directory---or as pre-formatted TensorFlow Datasets.

Pre-formatted Datasets
----------------------
Datasets can be loaded as pre-formatted TensorFlow Datasets using the `psiz-datasets Python package <https://github.com/psiz-org/psiz-datasets>`_, which can be installed via :code:`pip install psiz-datasets`.  See the `psiz-datasets README <https://github.com/psiz-org/psiz/blob/main/README.md>`_ for additional package information. See the sections below for instructions on loading specific datasets.

Naming Convention
-----------------
Datasets follow a two part naming convention. The first part refers to the name of the stimulus dataset. The second part refers to the human behavior collected while using the stimulus dataset. For example, :file:`birds16_rank2019` refers to the stimulus dataset `birds16` and a set of collected behavior refered to as `rank2019`.

Stimuli Pointers
----------------
Rather than passing around filenames; the data refers to specific
stimuli using indices. If using the raw data files (zipped directory from OSF), the index mappings can be found in :file:`stimuli.txt`. If using the the pre-formatted TensorFlow Datasets, the index mappings are accessible by using :code:`with_info=True` to return an additional :code:`info` object when loading the dataset. The index mapping dictionary is located at :code:`info.metadata['stimuli']`.

Data Timesteps
--------------
The TensorFlow Datasets can be loaded *with* or *without* a timestep axis by appending :code:`/with_timestep` or :code:`/without_timestep` to the dataset name when using :code:`tfds.load` (see examples below). By default, the dataset is loaded with a timestep axis. If loaded without a timestep axis, the timestep axis is simply unrolled into the batch axis.

Structuring for Training
------------------------
Your application will likely require that the loaded datasets be structured into inputs, targets, and sample weights for training. This is easily achieved using :code:`tf.data.Dataset.map` function. See `birds16_rank2019` below for an example.

birds16_rank2019
================
`Raw Data Files <https://osf.io/ujv4h/>`__

To load the pre-formatted TensorFlow Dataset:

.. code-block:: python

    import tensorflow_datasets as tfds
    import psiz_datasets.birds16_rank2019
    ds, info = tfds.load(
        'birds16_rank2019/with_timestep',
        split="train",
        with_info=True
    )

Example of dataset formatting:

.. code-block:: python

    def format_data_for_training(sample):
        """Format sample as (x, y, w) tuple."""
        x = {
            'given2rank1_stimulus_set': sample['given2rank1_stimulus_set'],
            'given8rank2_stimulus_set': sample['given8rank2_stimulus_set'],
        }
        y = {
            'given2rank1_outcome': sample['given2rank1_outcome'],
            'given8rank2_outcome': sample['given8rank2_outcome'],
        }
        w = {
            'given2rank1_outcome': sample['given2rank1_sample_weight'],
            'given8rank2_outcome': sample['given8rank2_sample_weight'],
        }
        return (x, y, w)

    tfds_all = tfds_all.map(
        lambda sample: format_data_for_training(sample)
    )

ilsvrc2012_val_hsj
==================

`Raw Data Files <https://osf.io/7f96y/>`__

To load the pre-formatted TensorFlow Dataset:

.. code-block:: python

    import tensorflow_datasets as tfds
    import psiz_datasets.ilsvrc2012_val_hsj
    ds, info = tfds.load(
        'ilsvrc2012_val_hsj/with_timestep',
        split="train",
        with_info=True
    )

skin_lesions2018_rank2018
=========================

`Raw Data Files <https://osf.io/mw75h/>`__

To load the pre-formatted TensorFlow Dataset:

.. code-block:: python

    import tensorflow_datasets as tfds
    import psiz_datasets.ilsvrc2012_val_hsj
    ds, info = tfds.load(
        'skin_lesions2018_rank2018/with_timestep',
        split="train",
        with_info=True
    )
