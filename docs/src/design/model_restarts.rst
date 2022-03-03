##############
Model Restarts
##############

:Author: Brett D. Roads

Motivation
==========

Unfortunately, embeddings are susceptible to discovering local optima during inference. While the usual tricks help, such as stochastic gradient decent, one often requires multiple restarts with different initializations to be confident in the solution.

Solution
========

In an effort to shield users from the burden of writing restart logic, PsiZ includes a :py:class:`psiz.keras.Restarter` object that implements a :py:meth:`fit` method similar to :py:class:`tf.keras.Model`. The state of most TensorFlow objects can be reset using a serialization/deserialization strategy. However, :py:class:`tf.keras.callbacks` do not implement this serialization strategy.

To patch this problem, PsiZ implements a subclass of :py:class:`tf.keras.callbacks.Callback`, which adds a :py:meth:`reset` method.

.. note::
    Subclassing :py:class:`tf.keras.callbacks.Callback` is considered a burdensome solution because it forces users to create subclasses for all callbacks they want to use with :py:class:`Restarter`. This strategy will likely change when a more elegant solution is found. 
