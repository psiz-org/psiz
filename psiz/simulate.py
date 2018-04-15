# -*- coding: utf-8 -*-
# Copyright 2018 The PsiZ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for simulating agent behavior.

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
