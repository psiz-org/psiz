##########################
The `mask_zero` Convention
##########################

:Author: Brett D. Roads

Motivation
==========

In some instances, it is necessary to mask data. For example, two sequences may not be the same length or efficient computation requires well-shaped arrays. In the case of embeddings, where indices are mapped vectors, one must designate an index value to serve as a mask. TensorFlow's :py:class:`keras.layers.Embedding` designates `0` as an optional mask which is toggled using the optional `mask_zero` Boolean argument.

Complications and Goals
=======================

PsiZ includes non-TensorFlow classes for organizing stimulus and trial data (i.e., Psiz data classes). The choice of using `0` as a mask has implications for these classes and PsiZ in general.

1. When converting from a PsiZ data class to a TensorFlow dataset, the interpretation of the indices should be the same.
2. The meaning of the zero index should be explicitly clear for users to minimize misinterpretation.
3. When stacking trial objects, the zero index must have the same meaning for both objects.
4. When stacking trial objects of different shapes, a mask value must be used.

Solution
========

The chosen solution aims to be *consistent* and *explicit*. To be consistent with TensorFlow, the optional argument `mask_zero` is added to all relevant classes. Like the TensorFlow :py:class:`keras.layers.Embedding` layer, `mask_zero` is `False` by default. This ensures that mask zero cases are explicit in the code. Although `mask_zero=False` by default means more verbose code for many models, explicit is better than implicit, and will help new users understand PsiZ mechanics.

Details
*******

#. The mask value is always 0 if masking is used. Users are not provided with the option to  use a custom mask value.
#. The `mask_zero` argument has the simplist interaction with other arguments. Some functions and classes allow users to specify how many stimuli are present (i.e., `n_stimuli`). If `mask_zero` is True, then the `n_stimuli` provided by the user must take into account that there is an additional "mask stimulus".
#. When stacking trials, check to make sure the objects have compatible `mask_zero` attributes. If not, throw error to make user explicitly aware of issue.
#. User is responsible inspecting indices of PsiZ `Catalog` object and incrementing indices if their model uses zero masking.
#. The `mask_zero` argument is added only when strictly necessary. Where possible, PsiZ functions require users to provide a list of eligible indices instead of providing a `mask_zero` argument. This makes the functions more general (e.g., `psiz.utils.pairwise_indices`). The `Catalog` class does not include a `mask_zero` argument since it is not necessary.

Rejected Solutions
==================

#. Require all users to treat index 0 as a mask. While this is a tempting strategy since it is the simplest to implement and maintain, it is counter-intuitive to new users.
#. Masking is on by default but can be turned off. Less explicit than other solutions and opposite convention of TensorFlow.
#. Use -1 (or nan) as a mask value so that interpretation of stimulus indices is the same regardless of masking. Since TensorFlow :py:class:`keras.layers.Embedding` layers use a mask value of 0, this requires a different interpretation of indices in PsiZ data objects and TensorFlow. This inconsistency fosters confusion. As for nan, it is not defined for integers and creates a saving complications for hdf5 files.
