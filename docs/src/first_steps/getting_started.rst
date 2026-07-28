###############
Getting Started
###############

:Author: Brett D. Roads


Purpose
=======

PsiZ provides computational tools for modeling how people perceive the world. The primary use case of PsiZ is to infer psychological representations from human behavior (e.g., similarity judgments). The package integrates cognitive theory with modern computational methods. 


What's in a name?
=================

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter :math:`\Psi` (psi) is often used to denote the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

Installation
============

PsiZ is hosted on PyPI and is easily installed using :code:`pip`. Alternatively, you can install using :code:`git`.

System Requirements
-------------------
* Python 3.10-3.13
* Backend runtime (choose one or more): TensorFlow, PyTorch, or JAX.
* cuDNN & CUDA are only required when using GPU-enabled backend builds. For TensorFlow GPU installs, consult the `TF compatibility matrix <https://www.tensorflow.org/install/source#gpu>`_.

Install using PyPI
------------------

.. code:: bash

    pip install psiz

Install backend runtime dependencies (choose one):

.. code:: bash

    pip install "psiz[backend-tensorflow]"

.. code:: bash

    pip install "psiz[backend-torch]"

.. code:: bash

    pip install "psiz[backend-jax]"

Set your backend before importing :code:`keras` or :code:`psiz`:

.. code:: bash

    export KERAS_BACKEND=torch

Install using git
-----------------
You can also install PsiZ via `git`. You first clone the PsiZ repository from GitHub to your local machine and then install via `pip`.

.. code:: bash

    git clone https://github.com/psiz-org/psiz.git
    pip install /local/path/to/psiz


Design Philosophy
=================

PsiZ is built around Keras and supports TensorFlow, PyTorch, and JAX backends. PsiZ focuses on providing mid-level objects that subclass :py:class:`keras.layers.Layer`. PsiZ aims to follow the principle of *progressive disclosure of complexity* to enable low-friction startup and opt-in flexibility.


What next?
==========

If deciding where to go next, you have a few options.

Tutorials
---------
We recommend starting with the "Beginner Tutorial - Part 1", which provides gentle walk-through of Psiz's core use case. After that, check out "Beginner Tutorial - Part 2".

Code Examples
-------------
If you are comfortable with PsiZ, you can dive into some script-based `examples <https://github.com/psiz-org/psiz/tree/main/examples>`_. The examples forgo the verbose explanations used in the tutorials in order to provide useful starting points for creating your own scripts.

Source Code
-----------
If you feel like diving into the deep end, you can explore the source code on GitHub, which contains detailed docstrings and comments.
