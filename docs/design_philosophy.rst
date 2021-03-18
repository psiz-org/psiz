#################
Design Philosophy
#################

PsiZ is built using the TensorFlow ecosystem and strives to follow TensorFlow
idioms as closely as possible.

Package-defined models are built by sub-classing
:py:class:`tf.keras.Model`. Components of a model are built using the
:py:class:`tf.keras.layers.Layer` API. A free parameter is implemented as a
:py:class:`tf.Variable`.

If you would like to contribute, please see
`CONTRIBUTING <https://github.com/roads/psiz/blob/master/CONTRIBUTING.md>`_ on
the GitHub repo for additional guidance.

PsiZ Models
===========
In PsiZ, a model can be thought as having two major components. The first
component is a psychological embedding which describes how the agent of
interest perceives similarity between a set of stimuli. This component
includes a conventional embedding (representing the stimuli coordinates in
psychological space) and a kernel that defines how simialrity is computed
between the embedding coordinates. The second component describes how
similarities are converted into an observed behavior, such as rankings or
ratings.

Deviations from TensorFlow
==========================

Embeddings are susceptible to local optima. While the usual tricks help, such
as stochastic gradient decent, one often requires multiple restarts with
different initializations to be confident in the solution. In an effort to
shield users from the burden of writing restart logic, PsiZ includes a
:py:class:`psiz.keras.Restarter` object that implements a :py:meth:`fit`
method similar to :py:class:`tf.keras.Model`. The state of most TensorFlow
objects can be reset using a serialization/deserialization strategy. However,
:py:class:`tf.keras.callbacks` do not implement this serialization strategy.
To patch this problem, PsiZ implements a subclass of
:py:class:`tf.keras.callbacks.Callback`, which adds a :py:meth:`reset` method.

.. note::
    Subclassing :py:class:`tf.keras.callbacks.Callback` is considered a
    burdensome solution because it forces users to create subclasses for all
    callbacks they want to use with :py:class:`Restarter`. This strategy will
    likely change when a more elegant solution is found. 
