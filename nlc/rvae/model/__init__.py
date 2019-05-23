import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from rvae.tf_utils.adamax import AdamaxOptimizer
from rvae.tf_utils.common import split, assign_to_gpu, average_grads
from rvae.tf_utils.distributions import DiagonalGaussian, discretized_logistic, compute_lowerbound, \
    repeat
from rvae.tf_utils.layers import conv2d, resize_nearest_neighbor, ar_multiconv2d, deconv2d

# settings
flags = tf.flags
flags.DEFINE_string("logdir", "./log", "Logging directory.")
flags.DEFINE_string("hpconfig",
                    "depth=1,num_blocks=4,kl_min=0.1,learning_rate=0.002,batch_size=32,enable_iaf=False,bidirectional=True,dataset=test_images",
                    "Overrides default hyper-parameters.")
flags.DEFINE_string("mode", "bbans", "Whether to run 'train' or 'eval' model.")
flags.DEFINE_string("evalmodel", "20190130-182348_depth=1,num_blocks=4,kl_min=0.1,learning_rate=0.002,batch_size=32,enable_iaf=False",
                    "File path to the eval model checkpoint, relative to <logdir>/train."
                    "By default, evaluation loads TF0.9 model from '<logdir>/train/model.ckpt-1250528', "
                    "otherwise TF1.10+ model from specified relative path. "
                    "If the specified path is a directory, will load the latest checkpoint.")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs used.")
FLAGS = flags.FLAGS


class IAFLayer(object):
    def __init__(self, hps, mode, scope, downsample=False):
        self.hps = hps
        self.mode = mode
        self.downsample = downsample
        self.scope = scope

    def up(self, input, **_):
        with arg_scope([conv2d]):
            self.qz_mean, self.qz_logsd, self.up_context, h = self.up_split(input)
            return self.up_merge(h, input)

    def up_split(self, input):
        with self.scope:
            x = tf.nn.elu(input)
            x = conv2d("up_conv1", x, 2 * self.hps.z_size + 2 * self.hps.h_size, stride=(
                [2, 2] if self.downsample else [1, 1]))
            return split(x, 1, [(self.hps.z_size), (self.hps.z_size),
                                (self.hps.h_size), (self.hps.h_size)])

    def up_merge(self, h, input):
        with self.scope:
            h = tf.nn.elu(h)
            h = conv2d("up_conv3", h, self.hps.h_size)
            if self.downsample:
                input = resize_nearest_neighbor(input, 0.5)
            return input + 0.1 * h

    def down(self, input):
        with arg_scope([conv2d, ar_multiconv2d]):
            h_det, posterior, prior, ar_context = self.down_split(
                input, self.qz_mean, self.qz_logsd, self.up_context)

            with self.scope:
                if self.mode in ["init", "sample"]:
                    z = prior.sample
                else:
                    z = posterior.sample

                hps = self.hps
                if self.mode == "sample":
                    kl_cost = kl_obj = tf.zeros([hps.batch_size * hps.k])
                else:
                    logqs = posterior.logps(z)
                    if hps.enable_iaf:
                        x = ar_multiconv2d("ar_multiconv2d", z, ar_context, [(hps.h_size),
                                                                             (hps.h_size)],
                                           [hps.z_size, hps.z_size])
                        arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                        z = (z - arw_mean) / tf.exp(arw_logsd)
                        logqs += arw_logsd

                    logps = prior.logps(z)

                    kl_cost = logqs - logps

                    if hps.kl_min > 0:
                        # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                        kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [2, 3]), [0], keepdims=True)
                        kl_ave = tf.maximum(kl_ave, hps.kl_min)
                        kl_ave = tf.tile(kl_ave, [hps.batch_size * hps.k, 1])
                        kl_obj = tf.reduce_sum(kl_ave, [1])
                    else:
                        kl_obj = tf.reduce_sum(kl_cost, [1, 2, 3])
                    kl_cost = tf.reduce_sum(kl_cost, [1, 2, 3])

            return self.down_merge(h_det, input, z), kl_obj, kl_cost

    def down_split(self, input, qz_mean, qz_logsd, up_context):
        with self.scope:
            x = tf.nn.elu(input)
            x = conv2d("down_conv1", x, 4 * self.hps.z_size + self.hps.h_size * 2)
            pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det = split(
                x, 1, [self.hps.z_size] * 4 + [self.hps.h_size] * 2)
            prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
            posterior = DiagonalGaussian(
                qz_mean + (rz_mean if self.hps.bidirectional else 0),
                2 * (qz_logsd + (rz_logsd if self.hps.bidirectional else 0)))
            return h_det, posterior, prior, up_context + down_context

    def down_merge(self, h_det, input, z):
        with self.scope:
            h = tf.concat([z, h_det], 1)
            h = tf.nn.elu(h)
            if self.downsample:
                input = resize_nearest_neighbor(input, 2)
                h = deconv2d("down_deconv2", h, self.hps.h_size)
            else:
                h = conv2d("down_conv2", h, self.hps.h_size)
            return input + 0.1 * h


class CVAE1(object):
    def __init__(self, hps, mode, x):
        self.hps = hps.copy()
        self.image_size = hps.image_size
        self.mode = mode
        self.x = x
        self.m_trunc = []
        self.dec_log_stdv = tf.get_variable("dec_log_stdv", initializer=tf.constant(0.0))

        assert self.hps.depth == 1

        # Input images are repeated k times on the input.
        # This is used for Importance Sampling loss (k is number of samples).
        self.data_size = hps.batch_size * hps.k

        losses = []
        grads = []
        xs = tf.split(self.x, hps.num_gpus)
        opt = AdamaxOptimizer(hps.learning_rate)

        num_pixels = 3 * np.prod(hps.image_size)
        for i in range(hps.num_gpus):
            with tf.device(assign_to_gpu(i)):
                m, obj, loss = self._forward(xs[i], i)
                losses += [loss]
                self.m_trunc += [m]

                # obj /= (np.log(2.) * num_pixels * hps.batch_size)
                if mode == "train":
                    grads += [opt.compute_gradients(obj)]

        self.global_step = tf.get_variable("global_step", [], tf.int32,
                                           initializer=tf.zeros_initializer,
                                           trainable=False)
        self.bits_per_dim = tf.add_n(losses) / (
                np.log(2.) * num_pixels * hps.batch_size * hps.num_gpus)

        if mode == "train":
            # add gradients together and get training updates
            grad = average_grads(grads)
            self.train_op = opt.apply_gradients(grad, global_step=self.global_step)
            tf.summary.scalar("model/bits_per_dim", self.bits_per_dim)
            tf.summary.scalar("model/dec_log_stdv", self.dec_log_stdv)
            self.summary_op = tf.summary.merge_all()
        else:
            self.train_op = tf.no_op()

        if mode in ["train", "eval"]:
            with tf.name_scope(None):  # This is needed due to EMA implementation silliness.
                # keep track of moving average
                ema = tf.train.ExponentialMovingAverage(decay=0.999)
                self.train_op = tf.group(*[self.train_op, ema.apply(tf.trainable_variables())])

                vars = ema.variables_to_restore()
                self.avg_dict = dict((k.replace('model/model/', 'model/', 1), v)
                                     for k, v in vars.items()) \
                    if is_eval_model_in_original_format() else vars

    def _forward(self, x, gpu):
        hps = self.hps

        x = self.preprocess(x)

        with arg_scope([conv2d, deconv2d], init=(self.mode == "init")):
            self.layers = self.create_layers()

            input = self.downsample(x)
            for layer in self.layers:
                input = layer.up(input)

            input = self.initial_input_down()
            kl_cost = kl_obj = 0.0

            for j, layer in reversed(list(enumerate(self.layers))):
                input, cur_obj, cur_cost = layer.down(input)
                kl_obj += cur_obj
                kl_cost += cur_cost

                if self.mode == "train" and gpu == hps.num_gpus - 1:
                    tf.summary.scalar("model/kl_obj_%02d_%02d" % (0, j),
                                      tf.reduce_mean(cur_obj))
                    tf.summary.scalar("model/kl_cost_%02d_%02d" % (0, j),
                                      tf.reduce_mean(cur_cost))

            x_out = self.upsample_and_postprocess(input)

        log_pxz = discretized_logistic(x_out, self.dec_log_stdv, sample=x)
        obj = tf.reduce_sum(kl_obj - log_pxz)

        if self.mode == "train" and gpu == hps.num_gpus - 1:
            tf.summary.scalar("model/log_pxz", -tf.reduce_mean(log_pxz))
            tf.summary.scalar("model/kl_obj", tf.reduce_mean(kl_obj))
            tf.summary.scalar("model/kl_cost", tf.reduce_mean(kl_cost))

        loss = tf.reduce_sum(compute_lowerbound(log_pxz, kl_cost, hps.k))
        return x_out, obj, loss

    def create_layers(self):
        return [IAFLayer(self.hps, self.mode, scope=tf.variable_scope("IAF_0_%d" % j))
                for j in range(self.hps.num_blocks)]

    def initial_input_down(self):
        self.h_top = tf.get_variable("h_top", [self.hps.h_size], initializer=tf.zeros_initializer)
        return tf.tile(tf.reshape(self.h_top, [1, -1, 1, 1]),
                       [self.data_size, 1, self.image_size[0] // 2,
                        self.image_size[1] // 2])

    def upsample_and_postprocess(self, input):
        x = tf.nn.elu(input)
        x = deconv2d("x_dec", x, 3, [5, 5])
        x = tf.clip_by_value(x, -0.5 + 1 / 512., 0.5 - 1 / 512.)
        return x

    def downsample(self, x):
        h = conv2d("x_enc", x, self.hps.h_size, [5, 5], [2, 2])  # -> [16, 16]
        return h

    def preprocess(self, x):
        x = tf.to_float(x)
        x = tf.clip_by_value((x + 0.5) / 256.0, 0.0, 1.0) - 0.5
        x = repeat(x, self.hps.k)
        return x


def is_eval_model_in_original_format():
    return FLAGS.evalmodel is None
