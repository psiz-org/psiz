PsiZ .psiz artifact specification v1.0.0
======================================

This document defines the minimal contract for PsiZ-controlled artifact
directories. The design goal is to make the layout deterministic,
backend-agnostic, and portable for long-term asset viability and sharing on
platforms such as Hugging Face.

Required layout
---------------

Every artifact directory should contain the following core files:

- README.md
- config.json
- model.safetensors
- LICENSE

Optional PsiZ-specific companion files may also be included:

- metadata.json
- model_index.json

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
     }
   }

Model index examples
---------------------

Minimal point-estimate model example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "weights": [
       {"name": "loc", "shape": [2]}
     ]
   }

Minimal three-level hierarchical model example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "weights": [
       {"name": "global/loc", "shape": [2]},
       {"name": "global/scale", "shape": [2]},
       {"name": "intermediate/loc", "shape": [4, 2]},
       {"name": "intermediate/scale", "shape": [4, 2]},
       {"name": "leaf/loc", "shape": [8, 2]},
       {"name": "leaf/scale", "shape": [8, 2]}
     ]
   }

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
