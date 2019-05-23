from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from rvae.model import IAFLayer, CVAE1
from rvae.tf_utils.layers import conv2d, ar_multiconv2d, deconv2d


class LayerwiseCVAE:
    """Allows layerwise execution of a CVAE model."""

    def __init__(self, model: CVAE1):
        self.model = model
        self.iaf_layers = model.layers

        with arg_scope([conv2d, deconv2d], init=(self.model.mode == "init")):
            self.latent_layers = [LatentLayer(lower, upper)
                                  for lower, upper in
                                  zip(self.iaf_layers[:-1], self.iaf_layers[1:])]

            self.top_context_inputs = create_context_inputs(self.model.hps, prefix='top_')

            input = self.model.initial_input_down()
            h_det, posterior, prior, _ = self.iaf_layers[-1].down_split(input,
                                                                        *self.top_context_inputs)
            top_inputs = (h_det, input)
            self.top_posterior_params_and_inputs = (posterior.mean, posterior.std), top_inputs
            self.top_prior_params_and_inputs = (prior.mean, prior.std), top_inputs

            self.bottom_down_inputs = create_down_inputs(self.model.hps, prefix='bottom_')
            input = self.iaf_layers[0].down_merge(*self.bottom_down_inputs)
            self.outputs = self.model.upsample_and_postprocess(input), self.model.dec_log_stdv

    def run_reconstruction(self, sess, bottom_outputs, sample):
        return sess.run(self.outputs,
                        dict(zip(self.bottom_down_inputs, bottom_outputs + (sample,))))

    def run_top_prior(self, sess):
        return sess.run(self.top_prior_params_and_inputs,
                        dict(zip(self.top_context_inputs, prior_contexts(self.model.hps))))

    def run_top_posterior(self, sess, contexts):
        return sess.run(self.top_posterior_params_and_inputs,
                        dict(zip(self.top_context_inputs, contexts)))

    def run_all_contexts(self, sess, x):
        return sess.run([(layer.qz_mean, layer.qz_logsd, layer.up_context)
                         for layer in self.iaf_layers], feed_dict={self.model.x: x})

    def get_model_parts_as_numpy_functions(self, sess):
        return partial(self.run_all_contexts, sess), \
               partial(self.run_top_prior, sess), \
               tuple(partial(layer.run_down_prior, sess) for layer in self.latent_layers), \
               partial(self.run_top_posterior, sess), \
               tuple(partial(layer.run_down_posterior, sess) for layer in self.latent_layers), \
               partial(self.run_reconstruction, sess)


class LatentLayer:
    """Shifted reinterpretation of ResNet layers, so that latents are the output of a layer."""

    def __init__(self, upper_layer: IAFLayer, lower_layer: IAFLayer):
        with arg_scope([conv2d, ar_multiconv2d]):
            self.upper_layer = upper_layer
            self.lower_layer = lower_layer

            self.hps = self.upper_layer.hps

            self.down_inputs = create_down_inputs(self.hps)
            self.down_output = self.lower_layer.down_merge(*self.down_inputs)

            self.context_inputs = create_context_inputs(self.hps)
            self.h_det_out, posterior, prior, _ = self.upper_layer.down_split(self.down_output,
                                                                              *self.context_inputs)
            self.down_outputs = self.h_det_out, self.down_output
            self.prior_params = prior.mean, prior.std
            self.posterior_params = posterior.mean, posterior.std

    def run_down_prior(self, sess: tf.Session, outputs, sample):
        return sess.run((self.prior_params, self.down_outputs),
                        {**dict(zip(self.down_inputs, outputs + (sample,))),
                         **dict(zip(self.context_inputs, prior_contexts(self.hps)))})

    def run_down_posterior(self, sess: tf.Session, outputs, sample, up_contexts):
        return sess.run((self.posterior_params, self.down_outputs),
                        {**dict(zip(self.down_inputs, outputs + (sample,))),
                         **dict(zip(self.context_inputs, up_contexts))})


def hidden_shape(hps):
    return hps.batch_size * hps.k, hps.h_size, hps.image_size[0] // 2, hps.image_size[1] // 2


def latent_shape(hps):
    return latent_from_image_shape(hps)(image_shape(hps))


def image_shape(hps):
    return (hps.batch_size * hps.k, 3, *hps.image_size)


def latent_from_image_shape(hps):
    return lambda s: (s[0], hps.z_size, s[2] // 2, s[3] // 2)


def create_down_inputs(hps, prefix=''):
    h_det_input = tf.placeholder(tf.float32, hidden_shape(hps), prefix + 'h_det_in')
    input = tf.placeholder(tf.float32, hidden_shape(hps), prefix + 'down_in')
    z_input = tf.placeholder(tf.float32, latent_shape(hps), prefix + 'z_in')
    return h_det_input, input, z_input


def create_context_inputs(hps, prefix=''):
    qz_mean_input = tf.placeholder(tf.float32, latent_shape(hps), prefix + 'qz_mean_in')
    qz_logstd_input = tf.placeholder(tf.float32, latent_shape(hps), prefix + 'qz_logstd_in')
    up_context_input = tf.placeholder(tf.float32, hidden_shape(hps), prefix + 'up_context_in')
    return qz_mean_input, qz_logstd_input, up_context_input


def prior_contexts(hps):
    return np.zeros(latent_shape(hps)), np.zeros(latent_shape(hps)), np.zeros(hidden_shape(hps))
