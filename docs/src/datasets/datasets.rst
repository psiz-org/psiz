########
Datasets
########

:Author: Brett D. Roads

Overview
========

A small number of PsiZ-compatible datasets can be obtained from the OSF repository `psiz-datasets <https://osf.io/cn2s3/>`_. Each trial sequence is stored as a separate JSON file.

Pre-formatted Datasets
----------------------
Datasets can be loaded as pre-formatted TensorFlow Datasets using the `psiz-datasets Python package <https://github.com/psiz-org/psiz-datasets>.` See the `psiz-datasets README <https://github.com/psiz-org/psiz/blob/main/README.md>` for install instructions. See the sections below for instructions on loading specific datasets.

Stimuli Pointers
----------------
Rather than passing around filenames; the data refers to specific
stimuli using indices. The mapping from indices to filenames is determined by a separate file :file:`stimuli.txt`.

Naming Convention
-----------------
Datasets follow a two part naming convention. The first part refers to the name of the stimulus dataset. The second part refers to the human behavior collected while using the stimulus dataset. For example, :file:`birds16_rank2019` refers to the stimulus dataset `birds16` and a set of collected behavior refered to as `rank2019`.

birds16_rank2019
================

To load the pre-formatted TensorFlow Dataset:

.. code-block:: python
    import psiz_datasets.birds16_rank2019
    ds, info = tfds.load('birds16_rank2019', split="train", with_info=True)

ilsvrc2012_val_hsj
==================

To load the pre-formatted TensorFlow Dataset:

.. code-block:: python
    import psiz_datasets.ilsvrc2012_val_hsj
    ds, info = tfds.load('ilsvrc2012_val_hsj', split="train", with_info=True)

skin_lesions2018_rank2018
=========================

To load the pre-formatted TensorFlow Dataset:

.. code-block:: python
    import psiz_datasets.ilsvrc2012_val_hsj
    ds, info = tfds.load(
        'skin_lesions2018_rank2018', split="train", with_info=True
    )
