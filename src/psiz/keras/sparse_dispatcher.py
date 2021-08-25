# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
#
# Modified by The PsiZ Authors (2021). Updated to use TensorFlow 2.x modules
# (e.g., tf.math). Removed dependency on external module (common_layers).
#
# Copyright 2021 The PsiZ Authors. All Rights Reserved.
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
"""Module for TensorFlow dispatching.

Classes:
    SparseDispatcher: A sparse dispatcer.

"""
import tensorflow as tf
# from tensorflow.python.framework import function


# TODO do we need this, if so why does it break during eager?
# @function.Defun(
#     python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
#     shape_func=lambda op: [op.inputs[0].get_shape()])
# def convert_gradient_to_tensor(x):
#     """Identity operation whose gradient is converted to a `Tensor`.
#     Currently, the gradient to `tf.concat` is particularly expensive to
#     compute if dy is an `IndexedSlices` (a lack of GPU implementation
#     forces the gradient operation onto CPU).  This situation occurs when
#     the output of the `tf.concat` is eventually passed to `tf.gather`.
#     It is sometimes faster to convert the gradient to a `Tensor`, so as
#     to get the cheaper gradient for `tf.concat`.  To do this, replace
#     `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

#     Args:
#         x: A `Tensor`.
#     Returns:
#         The input `Tensor`.
#     """
#     return x


class SparseDispatcher():
    """Helper for implementing a mixture of experts.

    TODO: update doc strings for arbitrary length inputs

    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.

    There are two functions:
        dispatch: take an input Tensor and create input Tensors for
            each expert.
        combine: take output Tensors from each expert and form a
            combined output Tensor. Outputs from different experts for
            the same batch element are summed together, weighted by the
            provided "gates".

    The class is initialized with a "gates" Tensor, which specifies
    which batch elements go to which experts, and the weights to use
    when combining the outputs.  Batch element b is sent to expert e
    iff gates[b, e] != 0. The inputs and outputs are all
    two-dimensional [batch, depth]. Caller is responsible for
    collapsing additional dimensions prior to calling this class and
    reshaping the output to the original shape.
    See common_layers.reshape_like().

    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.

        dispatcher = SparseDispatcher(num_experts, gates)
        expert_inputs = dispatcher.dispatch(inputs)
        expert_outputs = [
            experts[i](expert_inputs[i]) for i in range(num_experts)
        ]
        outputs = dispatcher.combine(expert_outputs)

    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))

    This class takes advantage of sparsity in the gate matrix by
    including in the `Tensor`s for expert i only the batch elements for
    which `gates[b, i] > 0`.

    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher.

        Args:
            num_experts: an integer.
            gates: a `Tensor` of shape `[batch_size, num_experts]`.

        Returns:
            a SparseDispatcher

        """
        self._gates = gates
        self._num_experts = num_experts

        # Determine locations of nonzero values of `gates`. Structure
        # as a 2D Tensor of matrix indices, (i.e., shape=(n_nonzero, 2)),
        # where the first column is the expert index and the second column
        # is an index in to the minibatch position.
        where = tf.cast(
            tf.where(tf.transpose(gates) > 0),
            tf.int32
        )

        # Split `where` Tensor into two separate columns.
        self._expert_index, self._batch_index = tf.unstack(
            where, num=2, axis=1
        )

        # Determine minibatch sizes for each expert.
        self._part_sizes_tensor = tf.reduce_sum(
            tf.cast(gates > 0, tf.int32), [0]
        )

        # Grab nonzero gate values (in the same order as `where`).
        self._nonzero_gates = tf.gather(
            tf.reshape(self._gates, [-1]),
            self._batch_index * num_experts + self._expert_index,
            axis=0
        )

    # TODO maybe dispatch_multi vs dispatch single
    # @add_name_scope()
    def dispatch_single(self, inputs):
        """Create one input Tensor for each expert.

        The `Tensor` for a expert `i` contains the slices of `inp`
        corresponding to the batch elements `b` where `gates[b, i] > 0`.

        Args:
            inputs: a `Tensor` of shape "[batch_size, <extra_input_dims>]`

        Returns:
            a list of `num_experts` `Tensor`s with shapes
                `[expert_batch_size_i, <extra_input_dims>]`.

        """
        inputs = tf.gather(inputs, self._batch_index, axis=0)
        return tf.split(inputs, self._part_sizes_tensor, 0)

    # @add_name_scope()
    def dispatch_multi(self, inputs):
        """Create one input Tensor for each expert.

        The `Tensor` for a expert `i` contains the slices of `inp`
        corresponding to the batch elements `b` where `gates[b, i] > 0`.

        Args:
            inputs: a `Tensor` of shape "[batch_size, <extra_input_dims>]`

        Returns:
            a list of `num_experts` `Tensor`s with shapes
                `[expert_batch_size_i, <extra_input_dims>]`.

        """
        # Initialize empty list for each expert.
        expert_list = [[] for _ in range(self._num_experts)]
        # Loop over inputs, creating expert-specific list of `inputs`.
        for inp in inputs:
            inp = tf.gather(inp, self._batch_index, axis=0)
            inp = tf.split(inp, self._part_sizes_tensor, 0)
            for i_expert in range(self._num_experts):
                expert_list[i_expert].append(inp[i_expert])

        return expert_list

    # @add_name_scope()
    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.

        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.

        Args:
            expert_out: a list of `num_experts` `Tensor`s, each with
                shape `[expert_batch_size_i, <extra_output_dims>]`.
        multiply_by_gates: a boolean

        Returns:
            a `Tensor` with shape `[batch_size, <extra_output_dims>]`.

        """
        # TODO potential issue, original file was concerned with tf.concat
        # efficiency. See tensor2tensor comments on
        # `common_layers.convert_gradient_to_tensor`.
        # stitched = common_layers.convert_gradient_to_tensor(
        #     tf.concat(expert_out, 0)
        # )

        # Combine expert outputs.
        stitched = tf.concat(expert_out, 0)

        # Optionally weight expert outputs by gate value.
        if multiply_by_gates:
            stitched *= tf.expand_dims(self._nonzero_gates, 1)

        # For a given batch item, combine expert outputs using sum.
        combined = tf.math.unsorted_segment_sum(
            stitched, self._batch_index, tf.shape(self._gates)[0]
        )
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert
            `Tensor`s.

        Returns:
            a list of `num_experts` one-dimensional `Tensor`s with type
                `tf.float32` and shapes `[expert_batch_size_i]`

        """
        return tf.split(self._nonzero_gates, self._part_sizes_tensor, 0)

    def expert_to_batch_indices(self):
        """Batch indices corresponding to the examples in the
            per-expert `Tensor`s.

        Returns:
            a list of `num_experts` one-dimensional `Tensor`s with type
                `tf.int64` and shapes `[expert_batch_size_i]`

        """
        return tf.split(self._batch_index, self._part_sizes_tensor, 0)

    @property
    def part_sizes(self):
        return self._part_sizes_tensor
