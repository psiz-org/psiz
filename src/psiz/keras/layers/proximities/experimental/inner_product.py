# -*- coding: utf-8 -*-
# Copyright 2024 The PsiZ Authors. All Rights Reserved.
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
# ============================================================================
"""InnerProduct pairwise proximity layer.

Classes:
    InnerProduct: A TensorFlow layer for computing the (weighted) inner
        product between pairs of vectors.

"""


import keras
import numpy as np

from psiz.keras.layers.proximities.proximity import Proximity


@keras.saving.register_keras_serializable(
    package="psiz.keras.layers", name="InnerProduct"
)
class InnerProduct(Proximity):
    """Compute the inner product between pairs of vectors.

    Computes the inner product z_i^T W z_j, where z_i and z_j exist on
    R^n and W exists on R^n x R^n. The constraints placed on the matrix
    "W" determine the type of the inner product operation. If W is
    constrained to be the identity matrix, the inner product is a dot
    product. If W is constrained to be symmetric positive definite, you
    get a generic inner product.

    The layer is implemented using `w_tril`, the lower triangular
    Cholesky decomposition of W (W = w_tril @ w_tril^T). This is done
    because positive definiteness is easier to gaurentee using this
    parameterization. Internally, a mask is applied to `w_tril` to
    ensure the variable `w_tril` is a lower diagonal matrix.

    The last axis of the input tensors is consumed in order to compute
    an inner product (see `call` method). It is assumed that both
    input tensors have the same rank, are broadcast-compatible, and
    have the same size for the last axis.

    """

    def __init__(
        self,
        w_tril_initializer=None,
        w_tril_regularizer=None,
        w_tril_constraint=None,
        w_tril_trainable=True,
        **kwargs
    ):
        """Initialize.

        Args:
            w_tril_initializer (optional): Initializer for a lower
                triangular matrix variable `w_tril`. By default this is
                the identity matrix.
            w_tril_regularizer (optional): Regularizer applied to `w_tril`.
            w_tril_constraint (optional): Constraint applied to `w_tril`.
            w_tril_trainable (optional): Boolean indicating if `w_tril`
                is trainable.

        """
        super(InnerProduct, self).__init__(**kwargs)

        self.w_tril_trainable = self.trainable and w_tril_trainable
        if w_tril_initializer is None:
            w_tril_initializer = keras.initializers.Identity(gain=1.0)
        else:
            w_tril_initializer = keras.initializers.get(w_tril_initializer)
        self.w_tril_initializer = w_tril_initializer
        self.w_tril_regularizer = keras.regularizers.get(w_tril_regularizer)
        self.w_tril_constraint = keras.constraints.get(w_tril_constraint)

    def build(self, input_shape):
        """Build."""
        n_dim = input_shape[0][-1]
        # Create lower triangular mask.
        tril_mask = []
        for i in range(n_dim):
            tril_mask_row = []
            for j in range(n_dim):
                if i >= j:
                    tril_mask_row.append(1.0)
                else:
                    tril_mask_row.append(0.0)
            tril_mask.append(tril_mask_row)
        tril_mask = np.stack(tril_mask)

        with keras.name_scope(self.name):
            self._tril_mask = self.add_weight(
                shape=[n_dim, n_dim],
                initializer=keras.initializers.Constant(value=tril_mask),
                trainable=False,
                name="tril_mask",
                dtype=self.compute_dtype,
            )
            self._untransformed_w_tril = self.add_weight(
                shape=[n_dim, n_dim],
                initializer=self.w_tril_initializer,
                regularizer=self.w_tril_regularizer,
                trainable=self.w_tril_trainable,
                name="w_tril",
                constraint=self.w_tril_constraint,
            )

    @property
    def w_tril(self):
        """Return `w_tril` attribute."""
        return self._tril_mask * self._untransformed_w_tril

    @property
    def w(self):
        """Return `w` attribute."""
        w_tril = self._tril_mask * self._untransformed_w_tril
        return keras.ops.matmul(w_tril, keras.ops.transpose(w_tril))

    def call(self, inputs):
        """Call.

        Args:
            inputs: A list of two tensors denoting a the set of
                vectors to compute pairwise distances. Each tensor is
                assumed to have the same shape and be at least rank-2.
                Any additional tensors in the list are ignored.
                shape = (batch_size, [n, m, ...] n_dim)

        Returns:
            shape = (batch_size, [n, m, ...])

        """
        z_0 = inputs[0]
        z_1 = inputs[1]

        w_tril = self._tril_mask * self._untransformed_w_tril
        w = keras.ops.matmul(w_tril, keras.ops.transpose(w_tril))

        # Add dummy axis to achieve batch dot product.
        z_0 = keras.ops.expand_dims(z_0, -2)
        z_1 = keras.ops.expand_dims(
            z_1, -1
        )  # NOTE: `axis=-1` is intentional to acheive transpose
        d = keras.ops.matmul(keras.ops.matmul(z_0, w), z_1)
        # Remove dummy and vetigal axis.
        d = keras.ops.squeeze(d, [-2, -1])

        return self.activation(d)

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update(
            {
                "w_tril_initializer": keras.initializers.serialize(
                    self.w_tril_initializer
                ),
                "w_tril_regularizer": keras.regularizers.serialize(
                    self.w_tril_regularizer
                ),
                "w_tril_constraint": keras.constraints.serialize(
                    self.w_tril_constraint
                ),
                "w_tril_trainable": self.w_tril_trainable,
            }
        )
        return config
