#############################
Data: Trials and Observations
#############################

:Author: Brett D. Roads

Inference is performed by fitting a model to a set of observed behavioral
outcomes. Two types of behavioral outcomes are currently supported:
*rank* similarity judgments and *rate* similarity judgments.


Rank
====

In the simplest case, an observation is obtained from a trial consisting of
three stimuli: a *query* stimulus (:math:`q`) and two *reference* stimuli
(:math:`a` and :math:`b`). An agent selects the reference stimulus that they
believe is more similar to the query stimulus. For this simple trial, there
are only two possible outcomes: either reference :math:`a` or :math:`b` is
selected. If the agent selected reference :math:`a`, then the observation for
trial :math:`i` would be recorded as the vector: 

.. math::
    D_{i} = [q, a, b]

Alternatively, if the agent had selected reference :math:`b`, the observation
would be recorded as:

.. math::
    D_{i} = [q, b, a]

Rank trials composed of one query and two reference stimuli, are sometimes
referred to as *triplet* trials. In addition to simple triplet trials, PsiZ
can handle a number of different rank trial configurations. A trial may have
2-8 reference stimuli and an agent may be required to select and rank more
than one reference stimulus. An Open Access article detailing rank trials is
available at https://link.springer.com/article/10.3758/s13428-019-01285-3.


Rate
====

In the simplest case, an observation is obtained from a trial consisting of
two stimuli. An agent provides a numerical rating regarding the similarity
between the stimuli.

.. note::
    This trial type is relatively new with many features still being developed
    and tested.


Other trial types
=================

While rankings and ratings include a large body of research, other types of
observations are possible. For example, a *sort* trial which requires
participants to sort a set of stimuli into an artbitrary number of different
piles. Other trials are constantly under consideration but constitute a sizable
time investment. Discussions and pull requests are strongly encouraged for new
trial types.
