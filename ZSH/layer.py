# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.util.tf_export import tf_export
from tensorflow.keras.layers import BatchNormalization as BatchNormalizationOri

class MBatchNormalization(BatchNormalizationOri):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Arguments:
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """
  def call(self, inputs, training=None):
    original_training_value = training
    if training is None:
      training = K.learning_phase()

    in_eager_mode = context.executing_eagerly()
    if self.virtual_batch_size is not None:
      # Virtual batches (aka ghost batches) can be simulated by reshaping the
      # Tensor and reusing the existing batch norm implementation
      original_shape = [-1] + inputs.shape.as_list()[1:]
      expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

      # Will cause errors if virtual_batch_size does not divide the batch size
      inputs = array_ops.reshape(inputs, expanded_shape)

      def undo_virtual_batching(outputs):
        outputs = array_ops.reshape(outputs, original_shape)
        return outputs

    if self.fused:
      outputs = self._fused_batch_norm(inputs, training=training)
      if self.virtual_batch_size is not None:
        # Currently never reaches here since fused_batch_norm does not support
        # virtual batching
        outputs = undo_virtual_batching(outputs)
      if not context.executing_eagerly() and original_training_value is None:
        outputs._uses_learning_phase = True  # pylint: disable=protected-access
      return outputs

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.get_shape()
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    if training_value is not False:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: array_ops.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: moving_mean)
      variance = tf_utils.smart_cond(training,
                                     lambda: variance,
                                     lambda: moving_variance)

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
        new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
      else:
        new_mean, new_variance = mean, variance

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        if in_eager_mode and not self.trainable:
          return

        return self._assign_moving_average(var, value, self.momentum)

      mean_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, new_mean),
          lambda: self.moving_mean)
      variance_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, new_variance),
          lambda: self.moving_variance)
      if not context.executing_eagerly():
        self.add_update(mean_update, inputs=True)
        self.add_update(variance_update, inputs=True)
    else:
      mean, variance = self.moving_mean, self.moving_variance
    mean, variance = self.moving_mean, self.moving_variance
    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    outputs = nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    if not context.executing_eagerly() and original_training_value is None:
      outputs._uses_learning_phase = True  # pylint: disable=protected-access
    return outputs

