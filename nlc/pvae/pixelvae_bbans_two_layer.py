import os
import argparse
import json
import sys
from functools import partial
from itertools import product

from scipy.stats import norm

import tensorflow as tf
import numpy as np
from autograd.builtins import tuple as ag_tuple
import craystack.vectorans as vrans
import craystack as cs
import craystack.bb_ans as bb_ans
import craystack.codecs as codecs
import time

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

from two_layer_pvae_codec import TwoLayerVAE, AutoRegressive_return_params


def softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    return np.exp(x - max_x) / np.sum(np.exp(x - max_x), axis=axis, keepdims=True)


rng = np.random.RandomState(0)

prior_precision = 8
obs_precision = 16
q_precision = 14

np.seterr(divide='raise')

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('--load_path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--settings', type=str)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_batches', type=int, default=1)
parser.add_argument('--chunked_data', action='store_true', default=False)
parser.add_argument('--chunk_idx', type=int, default=None)
args = parser.parse_args()

PARAMS_FILE = args.load_path
DATASET_PATH = args.data_dir
DATASET = args.dataset  # mnist_256, lsun_32, lsun_64, imagenet_64
SETTINGS = args.settings  # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big
batch_size = args.batch_size
num_batches = args.num_batches

if args.chunk_idx is not None:
    print('Chunk: {}'.format(args.chunk_idx))

SCRIPT_PATH, _ = os.path.split(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)

with open(os.path.join(SCRIPT_PATH, 'settings.json')) as f:
    settings_dict = json.load(f)

s = DotDict(settings_dict[SETTINGS])

latent1_shape = (batch_size, s.LATENT_DIM_1, s.LATENTS1_HEIGHT, s.LATENTS1_WIDTH)
latent2_shape = (batch_size, s.LATENT_DIM_2)
latent1_size = np.prod(latent1_shape)
latent2_size = np.prod(latent2_shape)

num_dims = num_batches * batch_size * s.N_CHANNELS * s.WIDTH * s.HEIGHT

if DATASET == 'mnist_256':
    train_data, dev_data, test_data = lib.mnist_256.load(batch_size, batch_size,
                                                         DATASET_PATH, rs=rng)
elif DATASET == 'lsun_32':
    train_data, dev_data = lib.lsun_bedrooms.load(batch_size, DATASET_PATH, downsample=True)
elif DATASET == 'lsun_64':
    train_data, dev_data = lib.lsun_bedrooms.load(batch_size, DATASET_PATH, downsample=False)
elif DATASET == 'imagenet_32':
    train_data, dev_data = lib.imagenet32.load(batch_size, DATASET_PATH, rs=rng)
elif DATASET == 'imagenet_64':
    if args.chunked_data:
        train_data, dev_data = lib.imagenet64.load_chunked(batch_size, DATASET_PATH, chunk=args.chunk_idx)
    else:
        train_data, dev_data = lib.imagenet64.load(batch_size, DATASET_PATH)
else:
    raise NotImplementedError("dataset not implemented")

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

lib.print_model_settings(locals().copy())

# Note: using GPU will not work because of GPU non-determinism
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')
    images_tf = tf.placeholder(tf.int32, shape=[None, s.N_CHANNELS, s.HEIGHT, s.WIDTH], name='all_images')

    latents1 = tf.placeholder(tf.float32, shape=[None, s.LATENT_DIM_1, s.LATENTS1_HEIGHT, s.LATENTS1_WIDTH],
                              name='all_latents1')
    latents2 = tf.placeholder(tf.float32, (batch_size, s.LATENT_DIM_2))

    embedded_images = lib.ops.embedding.Embedding('Embedding', 256, s.DIM_EMBED, images_tf)
    embedded_images = tf.transpose(embedded_images, [0, 4, 1, 2, 3])
    embedded_images = tf.reshape(embedded_images, [-1, s.DIM_EMBED * s.N_CHANNELS, s.HEIGHT, s.WIDTH])

    bn_update_moving_stats = False

    two_layer_model = model.TwoLayerModel(s, bn_is_training, bn_stats_iter, bn_update_moving_stats)

    # first layer
    mu_and_logsig1, h1 = two_layer_model.enc1(embedded_images)
    mu1, logsig1, sig1 = split(mu_and_logsig1)
    theta = two_layer_model.dec1_partial(latents1)
    logits = two_layer_model.dec1_pixelcnn(theta, embedded_images)

    # second layer
    mu_and_logsig2 = two_layer_model.enc2(h1)
    mu2, logsig2, sig2 = split(mu_and_logsig2)

    outputs2 = two_layer_model.dec2(latents2, latents1)
    mu1_prior, logsig1_prior, sig1_prior = split(outputs2)  # params on p(z2|z1)
    mu1_prior = 2. * tf.nn.softsign(mu1_prior / 2.)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    saver.restore(session, PARAMS_FILE)
    print('Restored model from file')


    def rec_net1(x):
        _mu, _sig, h = session.run([mu1, sig1, h1],
                                   feed_dict={images_tf: x,
                                              bn_is_training: False,
                                              bn_stats_iter: 0,
                                              total_iters: 99999})
        return _mu, _sig, h


    def rec_net2(h):
        return session.run([mu2, sig2],
                           feed_dict={h1: h,
                                      bn_is_training: False,
                                      bn_stats_iter: 0,
                                      total_iters: 99999})


    def gen_net1(z1_partial, z2):
        return session.run([mu1_prior, sig1_prior],
                           feed_dict={latents1: z1_partial,
                                      latents2: z2,
                                      bn_is_training: False,
                                      bn_stats_iter: 0,
                                      total_iters: 99999})


    def gen_net2_partial(z1):
        return session.run(theta,
                           feed_dict={latents1: z1,
                                      bn_is_training: False,
                                      bn_stats_iter: 0,
                                      total_iters: 99999})


    def gen_net2_pixelcnn(x_partial, theta_):
        logits_ = session.run(logits,
                              feed_dict={theta: theta_,
                                         images_tf: x_partial,
                                         bn_is_training: False,
                                         bn_stats_iter: 0,
                                         total_iters: 99999})
        return softmax(logits_)


    data_gen = dev_data()
    # images = np.concatenate([next(data_gen) for _ in range(num_batches)]).astype(np.uint64)
    images = np.concatenate(list(data_gen)[:num_batches]).astype(np.uint64)

    # create the codec for the posterior on z1
    post1_elem_idxs = [(slice(None), slice(None), y, x) for y, x in product(range(s.LATENTS1_HEIGHT),
                                                                            range(s.LATENTS1_WIDTH))]

    # the post1 elem codec needs to read and write from the correct subhead
    # which is the part of the head corresponding to that (x,y) coord
    def post1_elem_codec(params, idx):
        return cs.substack(codecs.DiagGaussianLatent(params[..., 0], params[..., 1],
                                                     params[..., 2], params[..., 3],
                                                     q_precision, prior_precision),
                           lambda head: head[idx])


    def post1_elem_param_fn(z2, mu1_post, sig1_post):
        def g(z1_idxs, params=None, idx=None):
            _, _, mu1_prior, sig1_prior = np.moveaxis(params, -1, 0)
            z1 = mu1_prior + sig1_prior * bb_ans.std_gaussian_centres(prior_precision)[z1_idxs]
            mu1_prior, sig1_prior = gen_net1(z1, z2)
            return np.stack((mu1_post, sig1_post, mu1_prior, sig1_prior), axis=-1)
        return g


    post1_codec = lambda z2, mu1_post, sig1_post: \
        AutoRegressive_return_params(post1_elem_param_fn(z2, mu1_post, sig1_post),
                                     latent1_shape,
                                     latent1_shape + (4,),
                                     post1_elem_idxs,
                                     post1_elem_codec)

    def get_theta1(eps1_vals, z2_vals):
        z1 = np.zeros(latent1_shape, dtype='float32')
        for idx in post1_elem_idxs:
            mu, sig = gen_net1(z1, z2_vals)
            z1_temp = mu + sig * eps1_vals
            z1[idx] = z1_temp[idx]
        return np.stack((np.zeros_like(mu), np.zeros_like(sig),
                         mu, sig), axis=-1)  # TODO: fix

    # define the order of autoregression
    obs_elem_idxs = [(slice(None), c, y, x) for y, x, c in product(range(s.HEIGHT),
                                                                   range(s.WIDTH),
                                                                   range(s.N_CHANNELS))]

    def obs_elem_param_fn(theta):
        def g(x, params=None, idx=None):
            return gen_net2_pixelcnn(x, theta)
        return g

    obs_elem_codec = lambda p, idx: codecs.Categorical(p, obs_precision)
    obs_codec = lambda theta: codecs.AutoRegressive(obs_elem_param_fn(theta),
                                                    np.shape(images[0]),
                                                    np.shape(images[0]) + (256,),
                                                    obs_elem_idxs,
                                                    obs_elem_codec)

    # Setup codecs
    def vae_view(head):
        return ag_tuple((np.reshape(head[:latent1_size], latent1_shape),
                         np.reshape(head[latent1_size:latent1_size + latent2_size], latent2_shape),
                         np.reshape(head[latent1_size + latent2_size:], (batch_size,))))


    vae_append, vae_pop = cs.repeat(cs.substack(
        TwoLayerVAE(gen_net2_partial,
                    rec_net1, rec_net2,
                    post1_codec, obs_codec,
                    prior_precision, q_precision,
                    get_theta1),
        vae_view), num_batches)

    other_bits_count = 1000000
    init_message = codecs.random_stack(other_bits_count,
                                       batch_size + latent1_size + latent2_size,
                                       rng)

    init_len = 32 * other_bits_count

    encode_t0 = time.time()
    message = vae_append(init_message, images.astype('uint64'))

    flat_message = codecs.flatten_benford(message)
    encode_t = time.time() - encode_t0

    print("All encoded in {:.2f}s".format(encode_t))

    message_len = 32 * len(flat_message)
    print("Used {} bits.".format(message_len))
    print("This is {:.2f} bits per dim.".format(message_len / num_dims))
    print('Extra bits: {}'.format(message_len - init_len))
    print('Extra per dim: {:.2f}'.format((message_len - init_len) / num_dims))

    ## Decode
    message = codecs.unflatten_benford(flat_message, batch_size + latent1_size + latent2_size)

    decode_t0 = time.time()
    message, images_ = vae_pop(message)
    decode_t = time.time() - decode_t0

    print('All decoded in {:.2f}s'.format(decode_t))

    np.testing.assert_equal(images, images_)

    init_head, init_tail = init_message
    head, tail = message
    assert np.all(init_head == head)

    # use this, or get into recursion issues
    while init_tail:
        el, init_tail = init_tail
        el_, tail = tail
        assert el == el_

