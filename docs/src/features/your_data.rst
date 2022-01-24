###################
Using your own data
###################

:Author: Brett D. Roads

To use your own data, you should place your data in an appropriate subclass of
:py:class:`psiz.trials.Observations`. Once the :py:class:`Observations` object
has been created, you can save it to disk by calling its :py:meth:`save`
method. It can be loaded later using the function
:py:func:`psiz.trials.load_trials`.

Consider the following example that creates some fake rank observations:

.. code-block:: python

    import numpy as np
    import psiz

    # Let's assume that we have 10 unique stimuli.
    stimuli_list = np.arange(0, 10, dtype=int)

    # Define a filepath for our observations.
    fp_obs = 'path/to/obs.hdf5'

    # Let's create 100 trials, where each trial is composed of a query and
    # four references. We will also assume that participants selected two
    # references (in order of their similarity to the query.)
    n_trial = 100
    n_reference = 4
    stimulus_set = np.empty([n_trial, n_reference + 1], dtype=int)
    n_select = 2 * np.ones((n_trial), dtype=int)
    for i_trial in range(n_trial):
        # Randomly selected stimuli and randomly simulate behavior for each
        # trial (one query, four references).
        stimulus_set[i_trial, :] = np.random.choice(
            stimuli_list, n_reference + 1, replace=False
        )

    # Create the observations object and save it to disk.
    obs = psiz.trials.RankObservations(stimulus_set, n_select=n_select)
    obs.save(fp_obs)

    # Load the observations from disk.
    obs = psiz.trials.load_trials(fp_obs)

Note that the values in `stimulus_set` are assumed to be contiguous integers
[0, N[, where N is the number of unique stimuli. Their order is also important.
The query is listed in the first column, an agent's selected references are
listed second (in order of selection if there are more than two) and then any
remaining unselected references are listed (in any order).
