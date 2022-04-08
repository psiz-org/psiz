##########################
Hosted Datasets and Models
##########################

:Author: Brett D. Roads

A select number of observations and pretrained models can be obtained from
PsiZ's OSF repository `psiz-datasets <https://osf.io/cn2s3/>`_.

General Information
===================


Catalogs
--------

Rather than passing around filenames; trials and models refer to specific
stimuli using indices. The mapping from indices to filenames is managed by
a :py:class:`psiz.Catalog` object. A :py:class:`psiz.Catalog` object has a
:py:meth:`save` method which saves the object to disk using the
:file:`hdf5` format. This file is typically named :file:`catalog.hdf5`.

To use a pre-defined catalog first download the :file:`catalog.hdf5` file and
then load it:

.. code-block:: python

    catalog = psiz.catalog.load_catalog('local/path/to/catalog.hdf5')
    filepaths = catalog.filepath()

    # If you want filepaths that are fully resolved to a local copy of the
    # of the stimulus dataset, set the `common_path` attribute.
    catalog.common_path = 'local/path/to/dataset/`
    filepaths = catalog.filepath()

The order of the stimuli in :code:`filepaths` corresponds to the indices used
in a trial object. For example, if a trial refers to a particular stimulus
using index :code:`23`, this corresponds to :code:`filepaths[23]`.

Mapping embedding layers to filenames is slightly more complicated because
they may have masking enabled. Please see `Stimuli Point Estimates`_ below.


Observations
------------

Observation files are typically named used the format:
:file:`obs-<input_id>.hdf5`. After downloading from OSF, observations can be
loaded:

.. code-block:: python

    import psiz

    fp_obs = 'local/path/to/obs.hdf5'
    obs = psiz.trials.load_trials(fp_obs)

Alternatively, you can quickly load observations and the corresponding catalog
without having to manually download by using the
:py:mod:`psiz.datasets` module. For example, observations for the ImageNet-HSJ
validation set can be loaded in the following way:

.. code-block:: python

    import psiz

    (obs, catalog) = psiz.datasets.load_dataset('ilsvrc_val_v0_2')    


Models
------

Models are stored using the TensorFlow SavedModel format. The TensorFlow
SavedModel format creates a directory and places various assets inside the
directory (e.g., model weights).

Embedding models are typically saved using the naming convention:
:file:`emb-<arch_id>-<input_id>-<n_dim>-<split>`.

- :file:`arch_id`: The model architecture ID.
- :file:`input_id`: The input data ID.
- :file:`n_dim`: An integer indicated the dimensionality of the embedding.
- :file:`split`: An (zero-indexed) integer indicating the data split if a
  split was used. If no split was performed this is :file:`x`.

For example,
:file:`emb-0-195-4-0` would indicate a model using :file:`arch_id=0`,
:file:`input_id=195`, :file:`n_dim=4`, and :file:`split=0`.


Loading a Pretrained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^

After downloading the desired model from OSF
(e.g., :file:`val/models/psiz0.5.0_tf2.4.x/emb-0-195-4-0`), load the model in the
following way:

.. code-block:: python

    import tensorflow as tf

    fp_model = 'local/path/to/emb-0-195-4-0`
    model = tf.keras.models.load_model(fp_model)

.. note::
    Currently, the only way to use a model is to download it from OSF. In
    the future, models may be hosted on TensorFlow Hub.


Stimuli Point Estimates
^^^^^^^^^^^^^^^^^^^^^^^

One often wants point estimates for the stimuli. Obtaining point estimates is
slightly different for models trained using MLE versus variational inference.
After loading the model (like above), you can obtain MLE point estimates in
the following way:

.. code-block:: python

    # Get the maximum likelihood estimates of the embedding coordinates.
    z = model.stimuli.embeddings
    if model.stimuli.mask_zero:
        z = z_mode[1:]


For a model trained with variational inference, you can retrieve the posterior
modes:

.. code-block:: python

    # Get the posterior modes of the embedding coordinates.
    z = model.stimuli.embeddings.mode()
    if model.stimuli.mask_zero:
        z = z[1:]

.. warning::
    TensorFlow and PsiZ embedding layers accept the optional argument
    :code:`mask_zero`. If :code:`mask_zero=True`, the first embedding
    coordinate is a meaningless placeholder. Care must be taken to remove this
    placeholder if using the embedding coordinates in a downstream
    application. The code snippets above demonstrate how one could write
    generic code that removes a masking coordinate if it exists.

After removing any masking coordinate, this means that :code:`z[0]` maps to
:code:`filepaths[0]` (from the catalog loading example in `Catalogs`_).


Domain-Specific Information
===========================

ImageNet-HSJ
------------

Validation Set
^^^^^^^^^^^^^^

- OSF link: https://osf.io/7f96y/
- :file:`catalog.hdf5` link: https://osf.io/bf3e2/

ImageNet validation set models are stored at :file:`val/models/psiz0.5.0_tf2.4.x​`.
These models were created using an active learning procedure that
intelligently selects which trials to ask human participants. Data collection
occured across multiple rounds. Each round involved model inference, trial
selection, and judgment collection. In each round, three models were inferred
to create an ensemble model.

Model directories are named using the format:
:file:`emb-<arch_id>-<input_id>-<n_dim>-<split>`.

- :code:`arch_id`: Identifier for model architecture.

    - :code:`arch_id=0`: Models trained using variational inference, L2
      distance, and exponential similarity kernel.

- :code:`input_id`: Indicates the active learning round.
- :code:`split`: Indicates the training/validation split that was used to
  encourage diversity of solutions in the ensemble.

Models for two rounds are currently hosted on OSF.

- **Round 118 (aka v0.1)**: Includes observations for 1,000 stimuli subset of the
  ImageNet validation set. There is one stimulus for each class.
- **Round 195 (aka v0.2)**: Includes observations for 50,000 stimuli.

.. note::
    Data collection is ongoing. A final round (v1.0) will be added in the near
    future.

.. note::
    Please see this `CVPR 2021 paper <https://ieeexplore.ieee.org/document/9578028>`_ for
    additional details regarding the ImageNet validation models.

The mapping of embedding indices to files is determined by the ImageNet
validation :file:`catalog.hdf5` object (https://osf.io/bf3e2/). This object
uses standard ImageNet class and file names, but does **not** use the
standard ImageNet ordering.
