############
Introduction
############

:Author: Brett D. Roads


What's in a name?
=================

The name PsiZ (pronounced like the word *size*, /sʌɪz/) is meant to serve as
shorthand for the term *psychological embedding*. The greek letter Psi is
often used to represent the field of psychology and the matrix variable **Z**
is often used in machine learning to denote a latent feature space.


Purpose
=======

PsiZ provides computational tools for inferring continuous, multivariate
stimulus representation from similarity relations. It integrates cognitive
theory with contemporary computational methods.


Design Philosophy
=================

PsiZ is built using the TensorFlow ecosystem and strives to follow TensorFlow
and Keras idioms as closely as possible.

PsiZ aims to provide useful top-level and mid-level objects for use in
cognitive models. Package-defined models (top-level) are implemented by
subclassing :py:class:`tf.keras.Model`. Model components (mid-level) are
implemented by subclassing :py:class:`tf.keras.layers.Layer`.


Examples
========

Check out the `examples <https://github.com/roads/psiz/tree/main/examples>`_
to explore other ways to take advantage of the various features that PsiZ
offers.
