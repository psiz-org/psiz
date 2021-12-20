############
Introduction
############

:Author: Brett D. Roads


Purpose
=======

PsiZ provides computational tools for modeling how people perceive the world. The primary use case of PsiZ is to infer psychological representations from human behavior (e.g., similarity judgments). The package integrates cognitive theory with modern computational methods. 


What's in a name?
=================

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as shorthand for the term *psychological embedding*. The greek letter :math:`\Psi` (psi) is often used to represent the field of psychology and the matrix variable **Z** is often used in machine learning to denote a latent feature space.


Design Philosophy
=================

PsiZ is built using the TensorFlow ecosystem and strives to closely follow  TensorFlow and Keras idioms, therefore inheriting all of the powerful functionality of TensorFlow and Keras. PsiZ aims to provide useful top-level and mid-level objects for use in cognitive models. Package-defined models (top-level) are implemented by subclassing :py:class:`tf.keras.Model`. Model components (mid-level) are implemented by subclassing :py:class:`tf.keras.layers.Layer`.


What next?
==========

If deciding what to read next, you have a few of options.

Quick Start
-----------
If you would like to get a sense of how the major pieces fit together, check out the minimum working example in the Quick Start.

Notebook Tutorials
------------------
For a more thorough and gentle introduction, you can explore our series of notebook tutorials. These tutorials are written as Jupyter notebooks which can be viewed as documentation or actively run using the corresponding source files on GitHub.

Code Examples
-------------
If you would like to dive into some pure code examples that demonstrate different applications of PsiZ, check out the `examples <https://github.com/roads/psiz/tree/main/examples>`_.
