############
Introduction
############

:Author: Brett D. Roads


Purpose
=======

PsiZ provides computational tools for modeling how people perceive the world. The primary use case of PsiZ is to infer psychological representations from human behavior (e.g., similarity judgments). The package integrates cognitive theory with modern computational methods. 


What's in a name?
=================

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter :math:`\Psi` (psi) is often used to denote the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.

Installation
============

PsiZ is hosted on PyPI and is easily installed using `pip`. Alternatively, you can insall using `git`.

System Requirements
-------------------
* Python 3.9, 3.10
* cuDNN & CUDA: Please see the `TF compatibility matrix <https://www.tensorflow.org/install/source#gpu>`_ for compatible versions. If using a conda virtual environment, you may want to install CUDA librarires using `conda install -c anaconda cudatoolkit=11.2` (replace 11.2 with the version appropriate for your TensorFlow version).

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

PsiZ is built using the TensorFlow ecosystem and strives to closely follow  TensorFlow and Keras idioms, therefore inheriting all of the powerful functionality of TensorFlow and Keras. PsiZ aims to provide useful top-level and mid-level objects for use in cognitive models. Package-defined models (top-level) are implemented by subclassing :py:class:`tf.keras.Model`. Model components (mid-level) are implemented by subclassing :py:class:`tf.keras.layers.Layer`.


What next?
==========

If deciding what to read next, you have a few of options.

Quick Start
-----------
If you would like to get a sense of how the major pieces fit together, check out the minimal working example in the Quick Start.

Beginner Tutorial
------------------
A thorough and gentle walk-through of Psiz's basic use case.

Tutorials
------------------
Explore PsiZ's most powerful applications.

Code Examples
-------------
Dive into pure code `examples <https://github.com/psiz-org/psiz/tree/main/examples>`_ that demonstrate different applications of PsiZ.
