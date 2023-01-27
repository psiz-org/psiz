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
* Python 3.9, 3.10
* cuDNN & CUDA: If using a conda virtual environment, you probably want to install cuDNN and CUDA libraries using :code:`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`. Replace the specified versions with those appropriate for your setup based on the `TF compatibility matrix <https://www.tensorflow.org/install/source#gpu>`_.   See the `TF Install Guide <https://www.tensorflow.org/install/pip>`_ for the latest recommendation.

Install using PyPI
------------------

.. code:: bash

    pip install psiz

Install using git
-----------------
You can also install PsiZ via `git`. You first clone the PsiZ repository from GitHub to your local machine and then install via `pip`.

.. code:: bash

    git clone https://github.com/psiz-org/psiz.git
    pip install /local/path/to/psiz


Design Philosophy
=================

PsiZ is built using the TensorFlow ecosystem and strives to closely follow  TensorFlow and Keras idioms, therefore inheriting all of the powerful functionality of TensorFlow and Keras. PsiZ focuses on providing mid-level objects that subclass :py:class:`tf.keras.layers.Layer`. PsiZ aims to follow the principle of *progressive disclosure of complexity* to enable low-friction startup and opt-in flexibility.


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
