"""
PixelVAE: A Latent Variable Model for Natural Images
Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, Francesco Visin, David Vazquez, Aaron Courville
"""

import os, sys

import tflib as lib
import tflib.train_loop_2
import tflib.ops.kl_unit_gaussian
import tflib.ops.kl_gaussian_gaussian
import tflib.ops.conv2d
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.embedding
from tflib.ops.util import split, clamp_logsig_and_sig, DotDict

import tflib.lsun_bedrooms
import tflib.mnist_256
import tflib.imagenet32
import tflib.imagenet64

import model

import datetime
import json
import numpy as np
import tensorflow as tf
from scipy.misc import imsave
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('--load_path', type=str, default='', nargs='?')
parser.add_argument('--n_gpus', type=int, default=1, nargs='?')
parser.add_argument('--dataset', type=str, default='mnist_256', nargs='?')
parser.add_argument('--settings', type=str, default='mnist_256', nargs='?')
parser.add_argument('--test_every', type=int, default=1000, nargs='?')
parser.add_argument('--stop_after', type=int, default=1000, nargs='?')
parser.add_argument('--callback_every', type=int, default=1000, nargs='?')
parser.add_argument('--batch_size', type=int, default=8, nargs='?')
parser.add_argument('--chunked_data', action='store_true', default=False)

args = parser.parse_args()

DATASET_PATH = args.data_path
N_GPUS = args.n_gpus
DATASET = args.dataset  # mnist_256, lsun_32, lsun_64, imagenet_64
SETTINGS = args.settings  # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big
TEST_EVERY = args.test_every
STOP_AFTER = args.stop_after
CALLBACK_EVERY = args.callback_every

SCRIPT_PATH, _ = os.path.split(os.path.realpath(__file__))

now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%dT%H.%M.%S')

if args.load_path:
    SAVE_PATH = args.load_path
else:
    SAVE_PATH = os.path.join(SCRIPT_PATH, 'saved/{}'.format(timestamp))


sys.path.append(SCRIPT_PATH)

with open(os.path.join(SCRIPT_PATH, 'settings.json')) as f:
    settings_dict = json.load(f)

s = DotDict(settings_dict[SETTINGS])  # these are our settings, use DotDict for easy reference
s.BATCH_SIZE = args.batch_size


if DATASET == 'mnist_256':
    train_data, dev_data, test_data = lib.mnist_256.load(s.BATCH_SIZE, s.BATCH_SIZE, DATASET_PATH)
elif DATASET == 'lsun_32':
    train_data, dev_data = lib.lsun_bedrooms.load(s.BATCH_SIZE, downsample=True)
elif DATASET == 'lsun_64':
    train_data, dev_data = lib.lsun_bedrooms.load(s.BATCH_SIZE, downsample=False)
elif DATASET == 'imagenet_32':
    train_data, dev_data = lib.imagenet32.load(s.BATCH_SIZE, DATASET_PATH)
elif DATASET == 'imagenet_64':
    if args.chunked_data:
        train_data, dev_data = lib.imagenet64.load_chunked(s.BATCH_SIZE, DATASET_PATH)
    else:
        train_data, dev_data = lib.imagenet64.load(s.BATCH_SIZE, DATASET_PATH)


lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    all_images = tf.placeholder(tf.int32, shape=[None, s.N_CHANNELS, s.HEIGHT, s.WIDTH], name='all_images')
    all_latents1 = tf.placeholder(tf.float32, shape=[None, s.LATENT_DIM_1, s.LATENTS1_HEIGHT, s.LATENTS1_WIDTH],
                                  name='all_latents1')

    split_images = tf.split(all_images, len(DEVICES), axis=0)
    split_latents1 = tf.split(all_latents1, len(DEVICES), axis=0)

    tower_cost = []
    tower_outputs1_sample = []

    for device_index, (device, images, latents1_sample) in enumerate(zip(DEVICES, split_images, split_latents1)):
        with tf.device(device):
            if device_index == 0:
                bn_update_moving_stats = True
            else:
                bn_update_moving_stats = False

            scaled_images = (tf.cast(images, 'float32') - 128.) / 64.
            if s.EMBED_INPUTS:
                embedded_images = lib.ops.embedding.Embedding('Embedding', 256, s.DIM_EMBED, images)
                embedded_images = tf.transpose(embedded_images, [0, 4, 1, 2, 3])
                embedded_images = tf.reshape(embedded_images, [-1, s.DIM_EMBED * s.N_CHANNELS, s.HEIGHT, s.WIDTH])

            if s.MODE == 'one_level':
                one_level_model = model.OneLayerModel(s, bn_is_training, bn_stats_iter, bn_update_moving_stats)

                if s.EMBED_INPUTS:
                    mu_and_logsig1 = one_level_model.encode(embedded_images)
                else:
                    mu_and_logsig1 = one_level_model.encode(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                eps = tf.random_normal(tf.shape(mu1))
                latents1 = mu1 + (eps * sig1)

                if s.EMBED_INPUTS:
                    outputs1 = one_level_model.decode(latents1, embedded_images)
                else:
                    outputs1 = one_level_model.decode(latents1, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.reshape(outputs1, [-1, 256]),
                        labels=tf.reshape(images, [-1])
                    )
                )

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha = tf.minimum(1., tf.cast(total_iters + 1, 'float32') / s.ALPHA1_ITERS) * s.KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu1,
                        logsig1,
                        sig1
                    )
                )

                kl_cost_1 *= float(s.LATENT_DIM_2) / (s.N_CHANNELS * s.WIDTH * s.HEIGHT)

                cost = reconst_cost + (alpha * kl_cost_1)

            elif s.MODE == 'two_level':
                # Layer 1

                two_layer_model = model.TwoLayerModel(s, bn_is_training, bn_stats_iter, bn_update_moving_stats)

                if s.EMBED_INPUTS:
                    mu_and_logsig1, h1 = two_layer_model.enc1(embedded_images)
                else:
                    mu_and_logsig1, h1 = two_layer_model.enc1(scaled_images)
                mu1, logsig1, sig1 = split(mu_and_logsig1)

                if mu1.get_shape().as_list()[2] != s.LATENTS1_HEIGHT:
                    raise Exception("s.LATENTS1_HEIGHT doesn't match mu1 shape!")
                if mu1.get_shape().as_list()[3] != s.LATENTS1_WIDTH:
                    raise Exception("s.LATENTS1_WIDTH doesn't match mu1 shape!")

                eps = tf.random_normal(tf.shape(mu1))
                latents1 = mu1 + (eps * sig1)

                if s.EMBED_INPUTS:
                    outputs1 = two_layer_model.dec1(latents1, embedded_images)
                    outputs1_sample = two_layer_model.dec1(latents1_sample, embedded_images)
                else:
                    outputs1 = two_layer_model.dec1(latents1, scaled_images)
                    outputs1_sample = two_layer_model.dec1(latents1_sample, scaled_images)

                reconst_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.reshape(outputs1, [-1, 256]),
                        labels=tf.reshape(images, [-1])
                    )
                )

                # Layer 2

                mu_and_logsig2 = two_layer_model.enc2(h1)
                mu2, logsig2, sig2 = split(mu_and_logsig2)

                eps = tf.random_normal(tf.shape(mu2))
                latents2 = mu2 + (eps * sig2)

                outputs2 = two_layer_model.dec2(latents2, latents1)

                mu1_prior, logsig1_prior, sig1_prior = split(outputs2)
                logsig1_prior, sig1_prior = clamp_logsig_and_sig(logsig1_prior, sig1_prior,
                                                                 total_iters, s.BETA_ITERS)
                mu1_prior = 2. * tf.nn.softsign(mu1_prior / 2.)

                # Assembly

                # An alpha of exactly 0 can sometimes cause inf/nan values, so we're
                # careful to avoid it.
                alpha1 = tf.minimum(1., tf.cast(total_iters + 1, 'float32') / s.ALPHA1_ITERS) * s.KL_PENALTY
                alpha2 = tf.minimum(1., tf.cast(total_iters + 1, 'float32') / s.ALPHA2_ITERS) * alpha1  # * s.KL_PENALTY

                kl_cost_1 = tf.reduce_mean(
                    lib.ops.kl_gaussian_gaussian.kl_gaussian_gaussian(
                        mu1,
                        logsig1,
                        sig1,
                        mu1_prior,
                        logsig1_prior,
                        sig1_prior
                    )
                )

                kl_cost_2 = tf.reduce_mean(
                    lib.ops.kl_unit_gaussian.kl_unit_gaussian(
                        mu2,
                        logsig2,
                        sig2
                    )
                )

                kl_cost_1 *= float(s.LATENT_DIM_1 * s.LATENTS1_WIDTH * s.LATENTS1_HEIGHT) / (
                            s.N_CHANNELS * s.WIDTH * s.HEIGHT)
                kl_cost_2 *= float(s.LATENT_DIM_2) / (s.N_CHANNELS * s.WIDTH * s.HEIGHT)

                cost = reconst_cost + (alpha1 * kl_cost_1) + (alpha2 * kl_cost_2)
                cost = cost / np.log(2)

            tower_cost.append(cost)
            if s.MODE == 'two_level':
                tower_outputs1_sample.append(outputs1_sample)

    full_cost = tf.reduce_mean(
        tf.concat([tf.expand_dims(x, 0) for x in tower_cost], axis=0), 0
    )

    if s.MODE == 'two_level':
        full_outputs1_sample = tf.concat(tower_outputs1_sample, axis=0)

    # Sampling

    if s.MODE == 'one_level':

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.stack([0, ch_sym, y_sym, x_sym, 0]), tf.stack([-1, 1, 1, 1, -1])),
                            [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]


        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out,
                               feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x,
                                          total_iters: 99999, bn_is_training: False, bn_stats_iter: 0})

        sample_fn_latents1 = np.random.normal(size=(8, s.LATENT_DIM_2)).astype('float32')


        def generate_and_save_samples(tag):
            def color_grid_vis(X, nh, nw, save_path):
                # from github.com/Newmu
                X = X.transpose(0, 2, 3, 1)
                h, w = X[0].shape[:2]
                img = np.zeros((h * nh, w * nw, 3))
                for n, x in enumerate(X):
                    j = int(n / nw)
                    i = n % nw
                    img[j * h:j * h + h, i * w:i * w + w, :] = x
                imsave(save_path, img)

            latents1_copied = np.zeros((64, s.LATENT_DIM_2), dtype='float32')
            for i in range(8):
                latents1_copied[i::8] = sample_fn_latents1

            samples = np.zeros(
                (64, s.N_CHANNELS, s.HEIGHT, s.WIDTH),
                dtype='int32'
            )

            print("Generating samples")
            for y in range(s.HEIGHT):
                for x in range(s.WIDTH):
                    for ch in range(s.N_CHANNELS):
                        next_sample = dec1_fn(latents1_copied, samples, ch, y, x)
                        samples[:, ch, y, x] = next_sample

            print("Saving samples")
            color_grid_vis(
                samples,
                8,
                8,
                os.path.join(SCRIPT_PATH, 'saved/samples_{}.png').format(tag)
            )

    elif s.MODE == 'two_level':

        def dec2_fn(_latents, _targets):
            return session.run([mu1_prior, logsig1_prior],
                               feed_dict={latents2: _latents, latents1: _targets, total_iters: 99999,
                                          bn_is_training: False, bn_stats_iter: 0})


        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits_sym = tf.reshape(
            tf.slice(full_outputs1_sample, tf.stack([0, ch_sym, y_sym, x_sym, 0]), tf.stack([-1, 1, 1, 1, -1])),
            [-1, 256])


        def dec1_logits_fn(_latents, _targets, _ch, _y, _x):
            return session.run(logits_sym,
                               feed_dict={all_latents1: _latents,
                                          all_images: _targets,
                                          ch_sym: _ch,
                                          y_sym: _y,
                                          x_sym: _x,
                                          total_iters: 99999,
                                          bn_is_training: False,
                                          bn_stats_iter: 0})


        N_SAMPLES = s.BATCH_SIZE
        if N_SAMPLES % N_GPUS != 0:
            raise Exception("N_SAMPLES must be divisible by N_GPUS")
        HOLD_Z2_CONSTANT = False
        HOLD_EPSILON_1_CONSTANT = False
        HOLD_EPSILON_PIXELS_CONSTANT = False

        # Draw z2 from N(0,I)
        z2 = np.random.normal(size=(N_SAMPLES, s.LATENT_DIM_2)).astype('float32')
        if HOLD_Z2_CONSTANT:
            z2[:] = z2[0][None]

        # Draw epsilon_1 from N(0,I)
        epsilon_1 = np.random.normal(size=(N_SAMPLES, s.LATENT_DIM_1, s.LATENTS1_HEIGHT, s.LATENTS1_WIDTH)).astype(
            'float32')
        if HOLD_EPSILON_1_CONSTANT:
            epsilon_1[:] = epsilon_1[0][None]

        # Draw epsilon_pixels from U[0,1]
        epsilon_pixels = np.random.uniform(size=(N_SAMPLES, s.N_CHANNELS, s.HEIGHT, s.WIDTH))
        if HOLD_EPSILON_PIXELS_CONSTANT:
            epsilon_pixels[:] = epsilon_pixels[0][None]


        def generate_and_save_samples(tag):
            # Draw z1 autoregressively using z2 and epsilon1
            print("Generating z1")
            z1 = np.zeros((N_SAMPLES, s.LATENT_DIM_1, s.LATENTS1_HEIGHT, s.LATENTS1_WIDTH), dtype='float32')
            for y in range(s.LATENTS1_HEIGHT):
                for x in range(s.LATENTS1_WIDTH):
                    z1_prior_mu, z1_prior_logsig = dec2_fn(z2, z1)
                    z1[:, :, y, x] = z1_prior_mu[:, :, y, x] + np.exp(z1_prior_logsig[:, :, y, x]) * epsilon_1[:, :, y,
                                                                                                     x]

            # Draw pixels (the images) autoregressively using z1 and epsilon_x
            print("Generating pixels")
            pixels = np.zeros((N_SAMPLES, s.N_CHANNELS, s.HEIGHT, s.WIDTH)).astype('int32')
            for y in range(s.HEIGHT):
                for x in range(s.WIDTH):
                    for ch in range(s.N_CHANNELS):
                        # start_time = time.time()
                        logits = dec1_logits_fn(z1, pixels, ch, y, x)
                        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                        probs = probs / np.sum(probs, axis=-1, keepdims=True)
                        cdf = np.cumsum(probs, axis=-1)
                        pixels[:, ch, y, x] = np.argmax(cdf >= epsilon_pixels[:, ch, y, x, None], axis=-1)
                        # print time.time() - start_time

            # Save them
            def color_grid_vis(X, nh, nw, save_path):
                # from github.com/Newmu
                X = X.transpose(0, 2, 3, 1)
                h, w = X[0].shape[:2]
                img = np.zeros((h * nh, w * nw, 3))
                for n, x in enumerate(X):
                    j = int(n / nw)
                    i = int(n % nw)
                    img[j * h:j * h + h, i * w:i * w + w, :] = x
                imsave(save_path, img)

            print("Saving")
            rows = int(np.sqrt(N_SAMPLES))
            while N_SAMPLES % rows != 0:
                rows -= 1
            color_grid_vis(
                pixels, rows, int(N_SAMPLES / rows),
                os.path.join(SCRIPT_PATH, 'saved/samples_{}.png').format(tag)
            )

    # Train!

    if s.MODE == 'one_level':
        prints = [
            ('alpha', alpha),
            ('reconst', reconst_cost),
            ('kl1', kl_cost_1)
        ]
    elif s.MODE == 'two_level':
        prints = [
            ('alpha1', alpha1),
            ('alpha2', alpha2),
            ('reconst', reconst_cost),
            ('kl1', kl_cost_1),
            ('kl2', kl_cost_2),
        ]

    decayed_lr = tf.train.exponential_decay(
        s.LR,
        total_iters,
        s.LR_DECAY_AFTER,
        s.LR_DECAY_FACTOR,
        staircase=True
    )

    lib.train_loop_2.train_loop(
        session=session,
        inputs=[total_iters, all_images],
        inject_iteration=True,
        bn_vars=(bn_is_training, bn_stats_iter),
        cost=full_cost,
        stop_after=STOP_AFTER,
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_and_save_samples,
        callback_every=CALLBACK_EVERY,
        test_every=TEST_EVERY,
        save_checkpoints=True,
        save_root=SAVE_PATH
    )
