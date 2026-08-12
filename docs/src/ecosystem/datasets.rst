#############
PsiZ Datasets
#############

:Author: Brett D. Roads

Overview
========

A small number of PsiZ-compatible datasets are hosted at the OSF repository `psiz-datasets <https://osf.io/cn2s3/>`_. These datasets can be obtained in raw form---where each trial sequence is stored as a separate JSON file inside a zipped directory---or as pre-formatted TensorFlow Datasets.

v0.14 Dataset Artifacts and Runtime Ingestion
=============================================

Starting in v0.14, PsiZ supports a backend-neutral dataset artifact format based on:

* Parquet table files.
* A PsiZ-managed :code:`manifest.json` with strict schema/version checks.
* Split management in a separate :code:`split_assignments` table.
* Optional normalized :code:`stimuli` and :code:`participants` dimensions for
    metadata that should not be duplicated in every observation row.

The runtime ingestion path is backend-neutral using Keras :code:`PyDataset` with Tier A minimal adapters:

* :code:`Dataset.tensorflow()`.
* :code:`Dataset.torch()`.
* :code:`Dataset.numpy()`.
* :code:`Dataset.arrow()`.

For component-built datasets created with :code:`psiz.data.Dataset([...])`, prefer the
Dataset runtime methods:

* :code:`Dataset.tensorflow()`
* :code:`Dataset.torch()`
* :code:`Dataset.numpy()`
* :code:`Dataset.arrow()`

The legacy :code:`Dataset.export(export_format="tfds")` path remains available for
transition but now emits :code:`DeprecationWarning`.

Create and load a PsiZ dataset artifact
---------------------------------------

For component-built datasets, use :code:`Dataset.save(...)` and
:code:`psiz.data.load(...)` as the canonical public API.

.. code-block:: python

    import numpy as np
    import psiz

    stimulus_set = np.array(
        [[1, 2, 3], [4, 5, 6]],
        dtype=np.int32,
    )
    content = psiz.data.Rank(stimulus_set, n_select=1, name="stimulus_set")

    outcome_idx = np.array([0, 1], dtype=np.int32)
    sample_weight = np.array([[1.0], [0.5]], dtype=np.float32)
    outcome = psiz.data.SparseCategorical(
        outcome_idx,
        depth=2,
        sample_weight=sample_weight,
        name="outcome",
    )

    dataset = psiz.data.Dataset([content, outcome])

    artifact_dir = "./example_dataset.psiz"
    dataset.save(
        artifact_dir,
        dataset_id="example_dataset",
        split_set_id="split_set_v1",
    )

    ds = psiz.data.load(
        artifact_dir,
        split_set_id="split_set_v1",
        split_labels=["train"],
    )

    # Optional Tier A adapters.
    tf_dataset = ds.tensorflow()
    torch_dataset = ds.torch()
    numpy_payload = ds.numpy()
    arrow_table = ds.arrow()

Normalized dimensions
---------------------

Artifacts may include a :code:`stimuli` dimension with one row per stimulus.
Its required columns are a preserved, non-null integer :code:`stimulus_id` and
a non-null dataset-root-relative :code:`filepath`; additional dataset metadata
columns are retained as passthrough fields. For set-valued stimulus inputs,
:code:`observation_stimuli` can provide one row per observation, feature, and
position, with foreign-key references to :code:`observations` and :code:`stimuli`.

Artifacts may also include a :code:`participants` dimension. The required
:code:`participant_id` is a PsiZ-managed, non-null :code:`int64` surrogate key,
not a raw provider identifier. :code:`n_sequence` and :code:`n_trial` record
validated counts derived from :code:`observations`; external IDs and other
demographic or provenance fields may be retained as optional passthrough
columns. Sensitive participant columns can be named in the table's
:code:`sensitive_columns` manifest metadata.

These dimensions are validated when an artifact is loaded: primary keys must
be unique, references must resolve, file integrity metadata must match, and
participant counts must agree with the observation facts. Invalid contracts
fail explicitly rather than silently dropping rows or falling back to a
runtime-specific loader.

Compatibility note:

Module-level helpers such as :code:`psiz.data.write_dataset_artifact_from_samples(...)`
and :code:`psiz.data.load_dataset(...)` remain available for compatibility and
low-level workflows.

Migrate existing TensorFlow dataset workflows
---------------------------------------------

Use the migration API to convert existing TensorFlow dataset workflows into PsiZ-managed Parquet artifacts.

.. code-block:: python

    import psiz

    report = psiz.migration.migrate_dataset_from_tfds(
        source=tf_dataset,
        output_dir="./migrated_dataset.psiz",
        split_set_id="split_set_v1",
        validate=True,
        dataset_id="my_dataset",
    )

    # Includes table counts, integrity metadata, and parity diagnostics.
    print(report["status"])

Deprecation and transition note
-------------------------------

TensorFlow-first dataset paths remain supported for transition, but new work should prefer PsiZ dataset artifacts and runtime adapters listed above.

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
