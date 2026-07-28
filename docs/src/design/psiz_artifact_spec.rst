PsiZ .psiz artifact specification v1.0.0
======================================

This document defines the minimal contract for PsiZ-controlled artifact
directories. The design goal is to make the layout deterministic,
backend-agnostic, and portable for long-term asset viability and sharing on
platforms such as Hugging Face.
In practice, this is intended to reduce long-term storage rot risk by keeping
PsiZ assets less sensitive to backend-specific format changes over time.
It is also intended to closely align with hosting patterns used by platforms
like Hugging Face so sharing and reusing artifacts is straightforward.

Required layout
---------------

Every artifact directory should contain the following core files:

- README.md
- config.json
- model.safetensors
- model_index.json
- metadata.json
- LICENSE

Minimal config example
-----------------------

.. code-block:: json

   {
     "artifact_type": "psiz_model",
     "format_name": "psiz",
     "format_version": "1.0.0",
     "backend": "torch",
     "architecture": {
       "class_name": "ExampleModel",
       "module": "example"
     },
     "license": {
       "name": "Apache-2.0",
       "policy": "include"
     },
     "model_config": {
       "class_name": "...",
       "config": {"...": "..."}
     }
   }

Model index examples
---------------------

Minimal point-estimate model example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "weight_format": "safetensors",
     "weight_file": "model.safetensors",
     "weights": [
       {"name": "loc", "key": "weight_00000", "shape": [2], "dtype": "float32"}
     ]
   }

Minimal three-level hierarchical model example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "weight_format": "safetensors",
     "weight_file": "model.safetensors",
     "weights": [
       {"name": "global/loc", "key": "weight_00000", "shape": [2], "dtype": "float32"},
       {"name": "global/scale", "key": "weight_00001", "shape": [2], "dtype": "float32"},
       {"name": "intermediate/loc", "key": "weight_00002", "shape": [4, 2], "dtype": "float32"},
       {"name": "intermediate/scale", "key": "weight_00003", "shape": [4, 2], "dtype": "float32"},
       {"name": "leaf/loc", "key": "weight_00004", "shape": [8, 2], "dtype": "float32"},
       {"name": "leaf/scale", "key": "weight_00005", "shape": [8, 2], "dtype": "float32"}
     ]
   }

Python save/load usage
----------------------

Canonical approach: use the public function helpers from :code:`psiz.keras`:

.. code-block:: python

   import psiz

   psiz.keras.save_psiz_model(model, "my_model.psiz")
   loaded_model = psiz.keras.load_psiz_model("my_model.psiz")

For subclasses of :code:`psiz.keras.StochasticModel`, a method-based form is
also available as optional convenience:

.. code-block:: python

   model.save_psiz("my_model.psiz")
   loaded_model = MyModel.load_psiz("my_model.psiz")

Backend resolution follows PsiZ precedence:

1. Explicit :code:`backend_override`
2. Active Keras backend
3. PsiZ default backend (:code:`torch`)

Schema validation is strict and includes required files, semver major-version
compatibility, and model-index to safetensors integrity checks.

Lifecycle policy: checkpoints vs artifacts
------------------------------------------

PsiZ intentionally distinguishes training checkpoints from durable artifacts:

1. Training checkpoints should use Keras-native checkpointing for fault
  tolerance and training resume behavior.
2. PsiZ `.psiz` artifacts should be used for finalized model export,
  reproducibility, long-term storage stability, and sharing.

In other words, checkpoint files and `.psiz` directories serve different
lifecycle goals and are expected to coexist in the same project workflow.
For a compact end-to-end example, see the checkpoint and export snippet in
:doc:`../tutorials/vi`.

Versioning rules
----------------

- format_version uses semver.
- The first component (major) must match the supported v1.0.0 contract.
- Major-version mismatches raise validation errors.

Validation behavior
-------------------

The validator raises an error for any of the following conditions:

- Missing required core files.
- Invalid or non-object JSON payloads.
- Unsupported major-version format.
- Missing required config fields.
- Invalid model index structure when a model_index.json file is present.
