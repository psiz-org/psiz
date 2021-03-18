#################
Pretrained Models
#################

A select number of pretrained models can be obtained PsiZ's
`OSF repository <https://osf.io/7f96y/>`_.

Models are stored as directories using TensorFlow's SavedModel format.

Model directories are named using the following format:

`emb-<arch_id>-<input_id>-<n_dim>-<split>`

For example, `emb-0-195-4-0` would indicate a model using `arch_id=0`,
`input_id=195`, `n_dim=4`, and `split=0`.


ImageNet Validation Set
=======================

ImageNet validation set models are stored at `val/models/psiz0.5.0_tf2.4.x​`.
These models were trained using active learning that intelligently selects
which trials to ask human participants. Data collection occurs across multiple
rounds involving model inference, trial selection, and judgment collection. In
each round, three models were inferred to create an ensemble model.

Models for two rounds are currently hosted on OSF.

- Round 118 (v0.1): This includes observations for 1,000 stimuli subset of the
ImageNet validation set. There is one stimulus for each class.
- Round 195 (v0.2): This includes observations for 50,000 stimuli.

.. note::
    Data collection is ongoing. A final round will be added in the near
    future.

.. note::
    Please see the `ArXiv paper <https://arxiv.org/abs/2011.11015>`_ for
    additional details.

Using a Pretrained Model
------------------------

After downloading the desired model from OSF
(e.g., val/models/psiz0.5.0_tf2.4.x/emb-0-195-4-0), load the model in the
following way:

.. code-block:: python
    import tensorflow as tf

    fp_model = 'local/path/to/emb-0-195-4-0`
    model = tf.keras.models.load_model(fp_model)
    # Get the posterior modes of the embedding.
    z_mode = model.stimuli.embeddings.mode()  # shape=(50001, 4)
    # The embedding layer use mask_zero=True, so the first embedding
    # coordinate is a meaningless placeholder.
    z_mode = z_mode[1:]  # shape=(50000, 4)

The mapping of embedding indices to files is handled by the ImageNet
validation `catalog.hdf5` object (https://osf.io/bf3e2/). This object uses the
standard class and file names, but does *not* use the standard ordering.

After downloading the catalog.hdf5 file, you can load it in the following way:

.. code-block:: python
    catalog = psiz.catalog.load_catalog('local/path/to/catalog.hdf5')
    filepaths = catalog.filepath()

    # If you want filepaths thar are fully resolved to your local copy
    # of the ImageNet validation set, set the `common_path` attribute.
    catalog.common_path = 'local/path/to/imagenet/val/`
    filepaths = catalog.filepath()
