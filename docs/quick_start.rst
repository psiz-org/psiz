###########
Quick Start
###########

:Author: Brett D. Roads

Similarity relations can be collected using a variety of paradigms. You will
need to use the appropriate model for your data. In addition to a model
choice, you need to provide two additional pieces of information:

1. The observed similarity relations (referred to as observations or *obs*).
2. The number of unique stimuli that will be in your embedding (`n_stimuli`).

The following minimalist example uses a :py:class:`psiz.keras.models.Rank`
psychological embedding to model a predefined set of ordinal similarity
relations.

.. literalinclude:: minimal_example.py
   :language: python
   :linenos:
