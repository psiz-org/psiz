################################
Data: Content, Group, & Behavior
################################

:Author: Brett D. Roads

For the purpose of modeling human behavior, data can be conceptualized as having three pieces: the *content* of the trial, the *group* membership of the participant, and the participant's observed *behavior*. The content and observed behavior depends on the trial type.

The `psiz.data` module provides lightweight classes that act as an optional on-ramp for users. The module provides the *content* classes `psiz.data.Rate` and `psiz.data.Rank`.

Rank
====

In the simplest case, an observation is obtained from a trial consisting of
three stimuli: a *query* stimulus (:math:`q`) and two *reference* stimuli
(:math:`a` and :math:`b`). An agent selects the reference stimulus that they
believe is more similar to the query stimulus. For this simple trial, there
are only two possible outcomes: either reference :math:`a` or :math:`b` is
selected.

Rank trials composed of one query and two reference stimuli, are sometimes referred to as *triplet* trials. In addition to simple triplet trials, PsiZ can handle a number of different rank trial configurations.

Some common rank trial configurations:

**2-rank-1**: Participant saw three images where one image was a 
*query* and the other two images were *references*. Participants selected 
the reference they considered most similarity to the query. There are two
possible outcomes.

**8-rank-2**: Participant saw nine images where one image was a 
*query* and the other eight images were *references*. Participants selected 
the two reference they considered most similarity to the query and also
indicated the order (rank) of their selections. There are 56 possible 
outcomes.


Rate
====

In the simplest case, an observation is obtained from a trial consisting of
two stimuli. An agent provides a numerical rating regarding the similarity
between the stimuli.


Other trial types
=================

While rankings and ratings include a large body of research, many other trial types are possible.
