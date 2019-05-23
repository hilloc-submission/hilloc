import argparse
import json
from functools import partial
from itertools import product

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
from scipy.stats import norm
from tflib.ops.util import split, clamp_logsig_and_sig, DotDict

import tflib.lsun_bedrooms
import tflib.mnist_256
import tflib.imagenet32
import tflib.imagenet64

import model


def softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    return np.exp(x - max_x) / np.sum(np.exp(x - max_x), axis=axis, keepdims=True)


rng = np.random.RandomState(0)

prior_precision = 8
obs_precision = 16
q_precision = 14

np.seterr(divide='raise')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--load_path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--settings', type=str)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_batches', type=int, default=5)
args = parser.parse_args()

PARAMS_FILE = args.load_path
DATASET_PATH = args.data_dir
DATASET = args.dataset  # mnist_256, lsun_32, lsun_64, imagenet_64
SETTINGS = args.settings  # mnist_256, 32px_small, 32px_big, 64px_small, 64px_big
batch_size = args.batch_size
num_batches = args.num_batches

with open('settings.json') as f:
    settings_dict = json.load(f)

s = DotDict(settings_dict[SETTINGS])

latent_dim = s.LATENT_DIM_2
latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
num_dims = num_batches * batch_size * s.N_CHANNELS * s.WIDTH * s.HEIGHT

if DATASET == 'mnist_256':
    train_data, dev_data, test_data = lib.mnist_256.load(batch_size, batch_size,
                                                         DATASET_PATH, rs=rng)
elif DATASET == 'lsun_32':
    train_data, dev_data = lib.lsun_bedrooms.load(batch_size, DATASET_PATH, downsample=True)
elif DATASET == 'lsun_64':
    train_data, dev_data = lib.lsun_bedrooms.load(batch_size, DATASET_PATH, downsample=False)
elif DATASET == 'imagenet_32':
    train_data, dev_data = lib.imagenet32.load(batch_size, DATASET_PATH)
elif DATASET == 'imagenet_64':
    train_data, dev_data = lib.imagenet64.load(batch_size, DATASET_PATH)
else:
    raise NotImplementedError("dataset not implemented")

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

lib.print_model_settings(locals().copy())


# Note: using GPU will not work because of GPU non-determinism
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})) as session:
    bn_is_training = tf.placeholder(tf.bool, shape=None, name='bn_is_training')
    bn_stats_iter = tf.placeholder(tf.int32, shape=None, name='bn_stats_iter')
    total_iters = tf.placeholder(tf.int32, shape=None, name='total_iters')

    images_tf = tf.placeholder(tf.int32, shape=[None, s.N_CHANNELS, s.HEIGHT, s.WIDTH], name='all_images')
    latents = tf.placeholder(tf.float32, (batch_size, s.LATENT_DIM_2))

    embedded_images = lib.ops.embedding.Embedding('Embedding', 256, s.DIM_EMBED, images_tf)
    embedded_images = tf.transpose(embedded_images, [0, 4, 1, 2, 3])
    embedded_images = tf.reshape(embedded_images, [-1, s.DIM_EMBED * s.N_CHANNELS, s.HEIGHT, s.WIDTH])

    bn_update_moving_stats = False
    one_level_model = model.OneLayerModel(s, bn_is_training, bn_stats_iter, bn_update_moving_stats)

    mu_and_logsig = one_level_model.encode(embedded_images)
    mu, log_sig, sig = split(mu_and_logsig)

    theta = one_level_model.decode_partial(latents)
    logits = one_level_model.decode_pixelcnn(theta, embedded_images)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    saver.restore(session, PARAMS_FILE)

    def rec_net(x):
        _mu, _sig = session.run([mu, sig],
                                feed_dict={images_tf: x,
                                           bn_is_training: False,
                                           bn_stats_iter: 0,
                                           total_iters: 99999})
        return _mu, _sig


    def gen_net(z):
        return session.run(theta,
                           feed_dict={latents: z,
                                      bn_is_training: False,
                                      bn_stats_iter: 0,
                                      total_iters: 99999})


    def pixelcnn(_theta, x_partial, params=None, idx=None):
        logits_ = session.run(logits, feed_dict={theta: _theta,
                                              images_tf: x_partial,
                                              bn_is_training: False,
                                              bn_stats_iter: 0,
                                              total_iters: 99999})
        return softmax(logits_)


    train_gen = train_data()
    images = np.concatenate([next(train_gen) for _ in range(num_batches)]).astype(np.uint64)

    # estimate entropy of q(z|x)
    mu_post, sig_post = rec_net(images[0])

    q_entropy = np.sum(norm.entropy(loc=mu_post, scale=sig_post))

    print('Popping z should require: {:.2f} bits: '.format(q_entropy * np.log2(np.e)))

    # define the order of autoregression
    obs_elem_idxs = [(slice(None), c, y, x) for y, x, c in product(range(s.HEIGHT),
                                                                   range(s.WIDTH),
                                                                   range(s.N_CHANNELS))]

    obs_elem_codec = lambda p, idx: codecs.Categorical(p, obs_precision)

    def obs_codec(theta):
        append, pop = codecs.AutoRegressive(partial(pixelcnn, theta),
                                            np.shape(images[0]),
                                            np.shape(images[0]) + (256,),
                                            obs_elem_idxs,
                                            obs_elem_codec)
        def pop_(msg):
            msg, (data, _) = pop(msg)
            return msg, data
        return append, pop_

    # Setup codecs
    def vae_view(head):
        return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                         np.reshape(head[latent_size:], (batch_size,))))


    vae_append, vae_pop = cs.repeat(cs.substack(
        bb_ans.VAE(gen_net, rec_net, obs_codec, prior_precision, q_precision),
        vae_view), num_batches)

    # Codec for adding extra bits to the start of the chain (necessary for bits
    # back).
    p = prior_precision
    other_bits_depth = 10
    other_bits_append, _ = cs.substack(cs.repeat(codecs.Uniform(p), other_bits_depth),
                                       lambda h: vae_view(h)[0])

    ## Encode
    # Initialize message with some 'extra' bits
    encode_t0 = time.time()
    init_message = vrans.x_init(batch_size + latent_size)

    other_bits = rng.randint(1 << p, size=(other_bits_depth,) + latent_shape, dtype=np.uint64)
    init_message = other_bits_append(init_message, other_bits)

    init_len = 32 * len(codecs.flatten_benford(init_message))

    # Encode the mnist images
    message = vae_append(init_message, images)

    flat_message = codecs.flatten_benford(message)
    encode_t = time.time() - encode_t0

    print("All encoded in {:.2f}s".format(encode_t))

    message_len = 32 * len(flat_message)
    print("Used {} bits.".format(message_len))
    print("This is {:.2f} bits per dim.".format(message_len / num_dims))
    print('Extra bits per dim: {:.2f}'.format((message_len - init_len) / num_dims))
    print('Extra bits: {:.2f}'.format(message_len - init_len))

    ## Decode
    message = codecs.unflatten_benford(flat_message, batch_size + latent_size)

    decode_t0 = time.time()
    message, images_ = vae_pop(message)
    decode_t = time.time() - decode_t0

    print('All decoded in {:.2f}s'.format(decode_t))

    np.testing.assert_equal(images, images_)
    np.testing.assert_equal(message, init_message)
