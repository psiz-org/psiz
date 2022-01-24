###########
Quick Start
###########

:Author: Brett D. Roads


Installation
============

If you haven't already, install the PsiZ python python package using ``pip install psiz``. Check out PsiZ's GitHub README for additional installation guidance.

Minimum Working Example
=======================

Behavioral data can be collected using a variety of paradigms. You will need to use the appropriate model for your data. 

If using ranked (ordinal) similarity judgements, you need to provide two additional pieces of information:

1. The observed similarity relations (referred to as observations or *obs*).
2. The number of unique stimuli that will be in your embedding (`n_stimuli`).

The following minimalist example uses a :py:class:`psiz.keras.models.Rank` psychological embedding to model a previously collected set of ordinal similarity relations.

.. literalinclude:: minimal_example.py
   :language: python
   :linenos:
