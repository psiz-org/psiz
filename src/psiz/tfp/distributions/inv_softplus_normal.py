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
# ============================================================================
"""InvSoftplusNormal distribution classes."""

import tensorflow as tf
from tensorflow_probability.python.bijectors import Softplus
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import assert_util


class InvSoftplusNormal(transformed_distribution.TransformedDistribution):
    """The inverse-softplus-normal distribution."""

    def __init__(
            self, loc, scale, hinge_softness=1., validate_args=False,
            allow_nan_stats=True, name='InverseSoftplusNormal'):
        """Construct an inverse-softplus-normal distribution.

        The InverseSoftplusNormal distribution models positive-valued
        random variables whose inverse-softplus is normally distributed
        with mean `loc` and standard deviation `scale`. It is
        constructed as the softplus transformation of a Normal
        distribution.

        Arguments:
            loc: Floating-point `Tensor`; the means of the underlying
                Normal distribution(s).
            scale: Floating-point `Tensor`; the stddevs of the
                underlying Normal distribution(s).
            hinge_softness: Floating-point `Tensor`; Governs the hinge
                of the softplus operation.
            validate_args: Python `bool`, default `False`. Whether to
                validate input with asserts. If `validate_args` is
                `False`, and the inputs are invalid, correct behavior
                is not guaranteed.
            allow_nan_stats: Python `bool`, default `True`. If `False`,
                raise an exception if a statistic (e.g. mean, mode,
                etc...) is undefined for any batch member. If `True`,
                batch members with valid parameters leading to
                undefined statistics will return NaN for this statistic.
            name: The name to give Ops created by the initializer.

        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super(InvSoftplusNormal, self).__init__(
                distribution=normal.Normal(loc=loc, scale=scale),
                bijector=Softplus(hinge_softness=hinge_softness),
                validate_args=validate_args,
                parameters=parameters,
                name=name
            )

    @classmethod
    def _params_event_ndims(cls):
        return dict(loc=0, scale=0)

    @property
    def loc(self):
        """Distribution parameter for the pre-transformed mean."""
        return self.distribution.loc

    @property
    def scale(self):
        """Distribution parameter for pre-transformed standard deviation."""
        return self.distribution.scale

    @property
    def hinge_softness(self):
        """Bijector parameter governing the hinge softness."""
        return self.bijector.hinge_softness

    # def _param_shapes(self):
    #     raise NotImplementedError

    def _median(self, **kwargs):
        return self.bijector(self.Distribution.mean())

    def _mean(self, **kwargs):
        raise NotImplementedError
        # TODO work out math
        # return tf.exp(
        #     self.distribution.mean() + 0.5 * self.distribution.variance()
        # )

    def _variance(self, **kwargs):
        raise NotImplementedError
        # TODO work out math
        # variance = self.distribution.variance()
        # return (
        #     tf.math.expm1(variance) *
        #     tf.exp(2. * self.distribution.mean() + variance)
        # )

    def _covariance(self, **kwargs):
        raise NotImplementedError

    def _mode(self, **kwargs):
        raise NotImplementedError
        # TODO work out math
        # return tf.exp(
        #     self.distribution.mean() - self.distribution.variance()
        # )

    def _entropy(self, **kwargs):
        raise NotImplementedError
        # TODO work out math
        # return (
        #     self.distribution.mean() + 0.5 +
        #     tf.math.log(self.distribution.stddev()) + 0.5 * np.log(2 * np.pi)
        # )

    def _sample_control_dependencies(self, value):
        assertions = []
        if not self.validate_args:
            return assertions
        assertions.append(assert_util.assert_non_negative(
            value, message='Sample must be non-negative.')
        )
        return assertions

    def _default_event_space_bijector(self):
        return Softplus(validate_args=self.validate_args)


@kullback_leibler.RegisterKL(InvSoftplusNormal, InvSoftplusNormal)
def _kl_lognormal_lognormal(a, b, name=None):

    """Calculate the batched KL divergence KL(a || b) with a and b LogNormal.

    This is the same as the KL divergence between the underlying Normal
    distributions.

    Arguments:
        a: instance of a LogNormal distribution object.
        b: instance of a LogNormal distribution object.
        name: Name to use for created operations.
            Default value: `None` (i.e.,
            `'kl_invsoftplusnormal_invsoftplusnormal'`).

    Returns:
        kl_div: Batchwise KL(a || b)

    """
    return kullback_leibler.kl_divergence(
        a.distribution,
        b.distribution,
        name=(name or 'kl_invsoftplusnormal_invsoftplusnormal')
    )
