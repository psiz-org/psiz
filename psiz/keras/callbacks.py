# -*- coding: utf-8 -*-
# Copyright 2020 The PsiZ Authors. All Rights Reserved.
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

"""Module of custom TensorFlow callbacks.

Classes:
    EarlyStoppingRe:
    TensorBoardRe:

"""

import os

from tensorflow.keras import callbacks


class EarlyStoppingRe(callbacks.EarlyStopping):
    """Custom early stopping.

    Arguments:
        See tf.keras.callbacks.EarlyStopping.

    """

    # def on_train_begin(self, logs=None):
    #     """Overload."""
    #     super().on_train_begin(logs=logs)

    #     # Add initial set of weights to best weights.
    #     if self.restore_best_weights:
    #         self.best_weights = self.model.get_weights()

    def reset(self, restart=None):
        """Reset best weights.

        Arguments:
            restart (optional): An integer indicating the restart
                number. This argument is not currently used, but
                that may change in later versions.

        """
        self.best_weights = None


class TensorBoardRe(callbacks.TensorBoard):
    """Custom TensorBoard callback."""

    def __init__(self, **kwargs):
        """Initialize.

        Arguments:
            kwargs: See tf.keras.callbacks.Tensorboard.

        """
        super().__init__(**kwargs)
        self.log_dir_init = self.log_dir

    def reset(self, restart=None):
        """Reset callback.

        Arguments:
            restart (optional): An integer indicating the restart
                number. If provided, results for each restart will be
                saved separately to allow joint viewing on TensorBoard.

        """
        # if retart == 0:
        #     self.write_graph = True
        # else:
        #     self.write_graph = False

        if restart is not None:
            # Distinguish between restart by setting log_dir for TensorBoard.
            self.log_dir = os.path.join(self.log_dir_init, str(restart))
