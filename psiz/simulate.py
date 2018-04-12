"""Module for simulating agent behavior.

Author: B D Roads
"""

from psiz.trials import JudgedTrials

class Agent(object):
    """Agent that simulates similarity judgments.
    """

    def __init__(self, embedding, group_idx=0):
        """Initialize.

        Args:
            embedding:
            group_id (optional):
        """

        self.embedding = embedding
        self.group_idx = group_idx
    
    def simulate(self, displays):
        """Simulate similarity judgments for provided displays.

        Args:
            displays: UnjudgedTrials object representing the
                to-be-judged displays. The order of the stimuli in the
                stimulus set is ignored for the simulations.
        
        Returns:
            JudgedTrials object representing the judged displays. 
                The order of the stimuli is now informative.
        """

    def _probability(self, displays):
            """
            """
