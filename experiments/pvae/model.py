import tflib as lib
import tflib.train_loop_2
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.embedding
from tflib.ops.util import split

import numpy as np

import tensorflow as tf
import functools

from scipy.misc import imsave



def nonlinearity(x):
    return tf.nn.elu(x)


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ResidualBlock(name, input_dim, output_dim, inputs, filter_size, mask_type=None, resample=None,
                  he_init=True, bn_is_training=True, bn_stats_iter=0, bn_update_moving_stats=True):
    """
    resample: None, 'down', or 'up'
    """
    if mask_type != None and resample != None:
        raise Exception('Unsupported configuration')

    if resample == 'down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim,
                                   stride=2)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(SubpixelConv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim,
                                 filter_size=1, mask_type=mask_type, he_init=False, biases=True,
                                 inputs=inputs)

    output = inputs
    if mask_type == None:
        output = nonlinearity(output)
        output = conv_1(name + '.Conv1', filter_size=filter_size, mask_type=mask_type, inputs=output,
                        he_init=he_init, weightnorm=False)
        output = nonlinearity(output)
        output = conv_2(name + '.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output,
                        he_init=he_init, weightnorm=False, biases=False)
        output = lib.ops.batchnorm.Batchnorm(name + '.BN', [0, 2, 3], output, bn_is_training,
                                             bn_stats_iter, update_moving_stats=bn_update_moving_stats)
    else:
        output = nonlinearity(output)
        output_a = conv_1(name + '.Conv1A', filter_size=filter_size, mask_type=mask_type, inputs=output,
                          he_init=he_init)
        output_b = conv_1(name + '.Conv1B', filter_size=filter_size, mask_type=mask_type, inputs=output,
                          he_init=he_init)
        output = pixcnn_gated_nonlinearity(output_a, output_b)
        output = conv_2(name + '.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output,
                        he_init=he_init)

    return shortcut + output


class OneLayerModel:
    def __init__(self, settings, bn_is_training, bn_stats_iter, bn_update_moving_stats):
        """Only for 64px_big_onelevel and MNIST. Needs modification for others."""
        self.s = settings
        self.bn_is_training = bn_is_training
        self.bn_stats_iter = bn_stats_iter
        self.bn_update_moving_stats = bn_update_moving_stats

        self.resblock = functools.partial(ResidualBlock,
                                          bn_is_training=self.bn_is_training,
                                          bn_stats_iter=self.bn_stats_iter,
                                          bn_update_moving_stats=self.bn_update_moving_stats)

    def encode(self, images):
        output = images

        if self.s.WIDTH == 64:
            if self.s.EMBED_INPUTS:
                output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                               output_dim=self.s.DIM_0, filter_size=1, inputs=output, he_init=False)
            else:
                output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=self.s.N_CHANNELS, output_dim=self.s.DIM_0,
                                               filter_size=1, inputs=output, he_init=False)

            output = self.resblock('EncFull.Res1', input_dim=self.s.DIM_0, output_dim=self.s.DIM_0,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res2', input_dim=self.s.DIM_0, output_dim=self.s.DIM_1,
                                   filter_size=3, resample='down', inputs=output)
            output = self.resblock('EncFull.Res3', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res4', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res5', input_dim=self.s.DIM_1, output_dim=self.s.DIM_2,
                                   filter_size=3, resample='down', inputs=output)
            output = self.resblock('EncFull.Res6', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res7', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res8', input_dim=self.s.DIM_2, output_dim=self.s.DIM_3,
                                   filter_size=3, resample='down', inputs=output)
            output = self.resblock('EncFull.Res9', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res10', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res11', input_dim=self.s.DIM_3, output_dim=self.s.DIM_4,
                                   filter_size=3, resample='down', inputs=output)
            output = self.resblock('EncFull.Res12', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res13', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4,
                                   filter_size=3, resample=None, inputs=output)
            output = tf.reshape(output, [-1, 4 * 4 * self.s.DIM_4])
            output = lib.ops.linear.Linear('EncFull.Output', input_dim=4 * 4 * self.s.DIM_4,
                                           output_dim=2 * self.s.LATENT_DIM_2, initialization='glorot',
                                           inputs=output)
        else:
            if self.s.EMBED_INPUTS:
                output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                               output_dim=self.s.DIM_1, filter_size=1, inputs=output, he_init=False)
            else:
                output = lib.ops.conv2d.Conv2D('EncFull.Input', input_dim=self.s.N_CHANNELS, output_dim=self.s.DIM_1,
                                               filter_size=1, inputs=output, he_init=False)

            output = self.resblock('EncFull.Res1', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res2', input_dim=self.s.DIM_1, output_dim=self.s.DIM_2,
                                   filter_size=3, resample='down', inputs=output)
            output = self.resblock('EncFull.Res3', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res4', input_dim=self.s.DIM_2, output_dim=self.s.DIM_3,
                                   filter_size=3, resample='down', inputs=output)
            output = self.resblock('EncFull.Res5', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3,
                                   filter_size=3, resample=None, inputs=output)
            output = self.resblock('EncFull.Res6', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3,
                                   filter_size=3, resample=None, inputs=output)
            output = tf.reduce_mean(output, reduction_indices=[2, 3])
            output = lib.ops.linear.Linear('EncFull.Output', input_dim=self.s.DIM_3, output_dim=2 * self.s.LATENT_DIM_2,
                                           initialization='glorot', inputs=output)

        return output

    def decode_partial(self, latents):
        output = tf.clip_by_value(latents, -50., 50.)

        if self.s.WIDTH == 64:
            output = lib.ops.linear.Linear('DecFull.Input', input_dim=self.s.LATENT_DIM_2,
                                           output_dim=4 * 4 * self.s.DIM_4, initialization='glorot', inputs=output)
            output = tf.reshape(output, [-1, self.s.DIM_4, 4, 4])
            output = self.resblock('DecFull.Res2', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res3', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res4', input_dim=self.s.DIM_4, output_dim=self.s.DIM_3, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('DecFull.Res5', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res6', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res7', input_dim=self.s.DIM_3, output_dim=self.s.DIM_2, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('DecFull.Res8', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res9', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res10', input_dim=self.s.DIM_2, output_dim=self.s.DIM_1, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('DecFull.Res11', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res12', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res13', input_dim=self.s.DIM_1, output_dim=self.s.DIM_0, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('DecFull.Res14', input_dim=self.s.DIM_0, output_dim=self.s.DIM_0, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
        else:
            output = lib.ops.linear.Linear('DecFull.Input', input_dim=self.s.LATENT_DIM_2, output_dim=self.s.DIM_3,
                                           initialization='glorot', inputs=output)
            output = tf.reshape(tf.tile(tf.reshape(output, [-1, self.s.DIM_3, 1]), [1, 1, 49]),
                                [-1, self.s.DIM_3, 7, 7])
            output = self.resblock('DecFull.Res2', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res3', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res4', input_dim=self.s.DIM_3, output_dim=self.s.DIM_2, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('DecFull.Res5', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('DecFull.Res6', input_dim=self.s.DIM_2, output_dim=self.s.DIM_1, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('DecFull.Res7', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
        return output

    def decode_pixelcnn(self, output, images):
        if self.s.WIDTH == 64:
            dim = self.s.DIM_0
        else:
            dim = self.s.DIM_1

        if self.s.PIXEL_LEVEL_PIXCNN:

            if self.s.EMBED_INPUTS:
                masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                                      output_dim=dim, filter_size=5, inputs=images,
                                                      mask_type=('a', self.s.N_CHANNELS), he_init=False)
            else:
                masked_images = lib.ops.conv2d.Conv2D('DecFull.Pix1', input_dim=self.s.N_CHANNELS, output_dim=dim,
                                                      filter_size=5, inputs=images,
                                                      mask_type=('a', self.s.N_CHANNELS), he_init=False)

            # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
            output = tf.concat([masked_images, output], axis=1)

            output = self.resblock('DecFull.Pix2Res', input_dim=2 * dim, output_dim=self.s.DIM_PIX_1, filter_size=3,
                                   mask_type=('b', self.s.N_CHANNELS), inputs=output)
            output = self.resblock('DecFull.Pix3Res', input_dim=self.s.DIM_PIX_1, output_dim=self.s.DIM_PIX_1,
                                   filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)
            output = self.resblock('DecFull.Pix4Res', input_dim=self.s.DIM_PIX_1, output_dim=self.s.DIM_PIX_1,
                                   filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)
            if self.s.WIDTH != 64:
                output = self.resblock('DecFull.Pix5Res', input_dim=self.s.DIM_PIX_1, output_dim=self.s.DIM_PIX_1,
                                       filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)

            output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=self.s.DIM_PIX_1, output_dim=256 * self.s.N_CHANNELS,
                                           filter_size=1, mask_type=('b', self.s.N_CHANNELS), he_init=False,
                                           inputs=output)

        else:

            output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=dim, output_dim=256 * self.s.N_CHANNELS,
                                           filter_size=1, he_init=False, inputs=output)

        return tf.transpose(
            tf.reshape(output, [-1, 256, self.s.N_CHANNELS, self.s.HEIGHT, self.s.WIDTH]),
            [0, 2, 3, 4, 1]
        )

    def decode(self, latents, images):
        output = self.decode_partial(latents)
        return self.decode_pixelcnn(output, images)


class TwoLayerModel:
    def __init__(self, s, bn_is_training, bn_stats_iter, bn_update_moving_stats):
        self.s = s
        self.bn_is_training = bn_is_training
        self.bn_stats_iter = bn_stats_iter
        self.bn_update_moving_stats = bn_update_moving_stats

        self.resblock = functools.partial(ResidualBlock,
                                          bn_is_training=self.bn_is_training,
                                          bn_stats_iter=self.bn_stats_iter,
                                          bn_update_moving_stats=self.bn_update_moving_stats)

    def enc1(self, images):
        output = images

        if self.s.WIDTH == 64:
            if self.s.EMBED_INPUTS:
                output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                               output_dim=self.s.DIM_0, filter_size=1, inputs=output, he_init=False)

                output = self.resblock('Enc1.InputRes0', input_dim=self.s.DIM_0, output_dim=self.s.DIM_0, filter_size=3,
                                       resample=None, inputs=output)
                output = self.resblock('Enc1.InputRes', input_dim=self.s.DIM_0, output_dim=self.s.DIM_1, filter_size=3,
                                       resample='down', inputs=output)
            else:
                output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=self.s.N_CHANNELS, output_dim=self.s.DIM_1,
                                               filter_size=1, inputs=output, he_init=False)
                output = self.resblock('Enc1.InputRes', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                                       resample='down', inputs=output)
        else:
            if self.s.EMBED_INPUTS:
                output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                               output_dim=self.s.DIM_1, filter_size=1, inputs=output, he_init=False)
            else:
                output = lib.ops.conv2d.Conv2D('Enc1.Input', input_dim=self.s.N_CHANNELS, output_dim=self.s.DIM_1,
                                               filter_size=1, inputs=output, he_init=False)

        output = self.resblock('Enc1.Res1Pre', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                               resample=None, inputs=output)
        output = self.resblock('Enc1.Res1Pre2', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                               resample=None, inputs=output)
        output = self.resblock('Enc1.Res1', input_dim=self.s.DIM_1, output_dim=self.s.DIM_2, filter_size=3,
                               resample='down', inputs=output)
        if self.s.LATENTS1_WIDTH == 16:
            output = self.resblock('Enc1.Res4Pre', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Enc1.Res4', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Enc1.Res4Post', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            mu_and_sigma = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=self.s.DIM_2, output_dim=2 * self.s.LATENT_DIM_1,
                                                 filter_size=1, inputs=output, he_init=False)
        else:
            output = self.resblock('Enc1.Res2Pre', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Enc1.Res2Pre2', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Enc1.Res2', input_dim=self.s.DIM_2, output_dim=self.s.DIM_3, filter_size=3,
                                   resample='down', inputs=output)
            output = self.resblock('Enc1.Res3Pre', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Enc1.Res3Pre2', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Enc1.Res3Pre3', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, inputs=output)
            mu_and_sigma = lib.ops.conv2d.Conv2D('Enc1.Out', input_dim=self.s.DIM_3, output_dim=2 * self.s.LATENT_DIM_1,
                                                 filter_size=1, inputs=output, he_init=False)

        return mu_and_sigma, output

    def dec1_partial(self, latents):
        output = tf.clip_by_value(tf.cast(latents, tf.float32), -50., 50.)

        if self.s.LATENTS1_WIDTH == 16:
            output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=self.s.LATENT_DIM_1, output_dim=self.s.DIM_2,
                                           filter_size=1, inputs=output, he_init=False)
            output = self.resblock('Dec1.Res1A', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Dec1.Res1B', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Dec1.Res1C', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
        else:
            output = lib.ops.conv2d.Conv2D('Dec1.Input', input_dim=self.s.LATENT_DIM_1, output_dim=self.s.DIM_3,
                                           filter_size=1, inputs=output, he_init=False)
            output = self.resblock('Dec1.Res1', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Dec1.Res1Post', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Dec1.Res1Post2', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Dec1.Res2', input_dim=self.s.DIM_3, output_dim=self.s.DIM_2, filter_size=3,
                                   resample='up', inputs=output)
            output = self.resblock('Dec1.Res2Post', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)
            output = self.resblock('Dec1.Res2Post2', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, inputs=output)

        output = self.resblock('Dec1.Res3', input_dim=self.s.DIM_2, output_dim=self.s.DIM_1, filter_size=3,
                               resample='up',
                               inputs=output)
        output = self.resblock('Dec1.Res3Post', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                               resample=None, inputs=output)
        output = self.resblock('Dec1.Res3Post2', input_dim=self.s.DIM_1, output_dim=self.s.DIM_1, filter_size=3,
                               resample=None, inputs=output)

        if self.s.WIDTH == 64:
            output = self.resblock('Dec1.Res4', input_dim=self.s.DIM_1, output_dim=self.s.DIM_0, filter_size=3,
                                   resample='up', inputs=output)
            output = self.resblock('Dec1.Res4Post', input_dim=self.s.DIM_0, output_dim=self.s.DIM_0, filter_size=3,
                                   resample=None, inputs=output)
        return output

    def dec1_pixelcnn(self, output, images):
        if self.s.PIXEL_LEVEL_PIXCNN:

            if self.s.WIDTH == 64:
                if self.s.EMBED_INPUTS:
                    masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                                          output_dim=self.s.DIM_0, filter_size=5, inputs=images,
                                                          mask_type=('a', self.s.N_CHANNELS), he_init=False)
                else:
                    masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=self.s.N_CHANNELS,
                                                          output_dim=self.s.DIM_0, filter_size=5, inputs=images,
                                                          mask_type=('a', self.s.N_CHANNELS), he_init=False)
            else:
                if self.s.EMBED_INPUTS:
                    masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=self.s.N_CHANNELS * self.s.DIM_EMBED,
                                                          output_dim=self.s.DIM_1, filter_size=5, inputs=images,
                                                          mask_type=('a', self.s.N_CHANNELS), he_init=False)
                else:
                    masked_images = lib.ops.conv2d.Conv2D('Dec1.Pix1', input_dim=self.s.N_CHANNELS,
                                                          output_dim=self.s.DIM_1, filter_size=5, inputs=images,
                                                          mask_type=('a', self.s.N_CHANNELS), he_init=False)

            # Make the variance of output and masked_images (roughly) match
            output /= 2

            # Warning! Because of the masked convolutions it's very important that masked_images comes first in this concat
            output = tf.concat([masked_images, output], axis=1)

            if self.s.WIDTH == 64:
                output = self.resblock('Dec1.Pix2Res', input_dim=2 * self.s.DIM_0, output_dim=self.s.DIM_PIX_1,
                                       filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)
                output = self.resblock('Dec1.Pix3Res', input_dim=self.s.DIM_PIX_1, output_dim=self.s.DIM_PIX_1,
                                       filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)
                output = self.resblock('Dec1.Pix4Res', input_dim=self.s.DIM_PIX_1, output_dim=self.s.DIM_PIX_1,
                                       filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)
            else:
                output = self.resblock('Dec1.Pix2Res', input_dim=2 * self.s.DIM_1, output_dim=self.s.DIM_PIX_1,
                                       filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)
                output = self.resblock('Dec1.Pix3Res', input_dim=self.s.DIM_PIX_1, output_dim=self.s.DIM_PIX_1,
                                       filter_size=3, mask_type=('b', self.s.N_CHANNELS), inputs=output)

            output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=self.s.DIM_PIX_1, output_dim=256 * self.s.N_CHANNELS,
                                           filter_size=1, mask_type=('b', self.s.N_CHANNELS), he_init=False,
                                           inputs=output)

        else:

            if self.s.WIDTH == 64:
                output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=self.s.DIM_0, output_dim=256 * self.s.N_CHANNELS,
                                               filter_size=1, he_init=False, inputs=output)
            else:
                output = lib.ops.conv2d.Conv2D('Dec1.Out', input_dim=self.s.DIM_1, output_dim=256 * self.s.N_CHANNELS,
                                               filter_size=1, he_init=False, inputs=output)

        return tf.transpose(
            tf.reshape(output, [-1, 256, self.s.N_CHANNELS, self.s.HEIGHT, self.s.WIDTH]),
            [0, 2, 3, 4, 1]
        )

    def dec1(self, latents1, images):
        theta1 = self.dec1_partial(latents1)
        return self.dec1_pixelcnn(theta1, images)

    def enc2(self, h1):
        output = h1

        if self.s.LATENTS1_WIDTH == 16:
            output = self.resblock('Enc2.Res0', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('Enc2.Res1Pre', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('Enc2.Res1Pre2', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('Enc2.Res1', input_dim=self.s.DIM_2, output_dim=self.s.DIM_3, filter_size=3,
                                   resample='down', he_init=True, inputs=output)

        output = self.resblock('Enc2.Res2Pre', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Enc2.Res2Pre2', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Enc2.Res2Pre3', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Enc2.Res1A', input_dim=self.s.DIM_3, output_dim=self.s.DIM_4, filter_size=3,
                               resample='down', he_init=True, inputs=output)
        output = self.resblock('Enc2.Res2PreA', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Enc2.Res2', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                               resample=None,
                               he_init=True, inputs=output)
        output = self.resblock('Enc2.Res2Post', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                               resample=None, he_init=True, inputs=output)

        output = tf.reshape(output, [-1, 4 * 4 * self.s.DIM_4])
        output = lib.ops.linear.Linear('Enc2.Output', input_dim=4 * 4 * self.s.DIM_4,
                                       output_dim=2 * self.s.LATENT_DIM_2,
                                       inputs=output)

        return output

    def dec2(self, latents, targets):
        output = tf.clip_by_value(latents, -50., 50.)
        output = lib.ops.linear.Linear('Dec2.Input', input_dim=self.s.LATENT_DIM_2, output_dim=4 * 4 * self.s.DIM_4,
                                       inputs=output)

        output = tf.reshape(output, [-1, self.s.DIM_4, 4, 4])

        output = self.resblock('Dec2.Res1Pre', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Dec2.Res1', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                               resample=None,
                               he_init=True, inputs=output)
        output = self.resblock('Dec2.Res1Post', input_dim=self.s.DIM_4, output_dim=self.s.DIM_4, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Dec2.Res3', input_dim=self.s.DIM_4, output_dim=self.s.DIM_3, filter_size=3,
                               resample='up',
                               he_init=True, inputs=output)
        output = self.resblock('Dec2.Res3Post', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Dec2.Res3Post2', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                               resample=None, he_init=True, inputs=output)
        output = self.resblock('Dec2.Res3Post3', input_dim=self.s.DIM_3, output_dim=self.s.DIM_3, filter_size=3,
                               resample=None, he_init=True, inputs=output)

        if self.s.LATENTS1_WIDTH == 16:
            output = self.resblock('Dec2.Res3Post5', input_dim=self.s.DIM_3, output_dim=self.s.DIM_2, filter_size=3,
                                   resample='up', he_init=True, inputs=output)
            output = self.resblock('Dec2.Res3Post6', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('Dec2.Res3Post7', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)
            output = self.resblock('Dec2.Res3Post8', input_dim=self.s.DIM_2, output_dim=self.s.DIM_2, filter_size=3,
                                   resample=None, he_init=True, inputs=output)

        if self.s.HIGHER_LEVEL_PIXCNN:
            targets = tf.clip_by_value(targets, -100., 100.)
            if self.s.LATENTS1_WIDTH == 16:
                masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=self.s.LATENT_DIM_1,
                                                       output_dim=self.s.DIM_2, filter_size=5,
                                                       mask_type=('a', self.s.PIX_2_N_BLOCKS), he_init=False,
                                                       inputs=targets)
            else:
                masked_targets = lib.ops.conv2d.Conv2D('Dec2.Pix1', input_dim=self.s.LATENT_DIM_1,
                                                       output_dim=self.s.DIM_3, filter_size=5,
                                                       mask_type=('a', self.s.PIX_2_N_BLOCKS), he_init=False,
                                                       inputs=targets)

            # Make the variance of output and masked_targets roughly match
            output /= 2

            output = tf.concat([masked_targets, output], axis=1)

            if self.s.LATENTS1_WIDTH == 16:
                output = self.resblock('Dec2.Pix2Res', input_dim=2 * self.s.DIM_2, output_dim=self.s.DIM_PIX_2,
                                       filter_size=3, mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=True,
                                       inputs=output)
            else:
                output = self.resblock('Dec2.Pix2Res', input_dim=2 * self.s.DIM_3, output_dim=self.s.DIM_PIX_2,
                                       filter_size=3, mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=True,
                                       inputs=output)
            output = self.resblock('Dec2.Pix3Res', input_dim=self.s.DIM_PIX_2, output_dim=self.s.DIM_PIX_2,
                                   filter_size=3,
                                   mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=True, inputs=output)
            output = self.resblock('Dec2.Pix4Res', input_dim=self.s.DIM_PIX_2, output_dim=self.s.DIM_PIX_2,
                                   filter_size=1,
                                   mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=True, inputs=output)

            output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=self.s.DIM_PIX_2, output_dim=2 * self.s.LATENT_DIM_1,
                                           filter_size=1, mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=False,
                                           inputs=output)

        else:

            if self.s.LATENTS1_WIDTH == 16:
                output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=self.s.DIM_2, output_dim=2 * self.s.LATENT_DIM_1,
                                               filter_size=1, mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=False,
                                               inputs=output)
            else:
                output = lib.ops.conv2d.Conv2D('Dec2.Out', input_dim=self.s.DIM_3, output_dim=2 * self.s.LATENT_DIM_1,
                                               filter_size=1, mask_type=('b', self.s.PIX_2_N_BLOCKS), he_init=False,
                                               inputs=output)

        return output
