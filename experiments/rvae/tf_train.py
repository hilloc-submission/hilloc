import time
from functools import lru_cache
from itertools import repeat
from operator import itemgetter
from pathlib import Path
from time import strftime

import numpy as np
import observations
import tensorflow as tf
import tqdm
from craystack import codecs
from tensorflow.python.training.supervisor import Supervisor

from rvae.datasets import sampling_testimage, test_image, sampling_testimages
from rvae.model import CVAE1, is_eval_model_in_original_format, FLAGS
from rvae.model.layerwise import LayerwiseCVAE, latent_shape, latent_from_image_shape, image_shape
from rvae.tf_utils.common import img_stretch, img_tile
from rvae.tf_utils.hparams import HParams

def relative_eval_model_path():
    return 'model.ckpt-1250528' if is_eval_model_in_original_format() else FLAGS.evalmodel


def restore_path():
    p = Path(FLAGS.logdir) / "train" / relative_eval_model_path()
    if p.is_dir():
        p = tf.train.latest_checkpoint(str(p))
    return str(p)


def get_default_hparams():
    return HParams(
        batch_size=16,  # Batch size on one GPU.
        eval_batch_size=100,  # Batch size for evaluation.
        num_gpus=8,  # Number of GPUs (effectively increases batch size).
        learning_rate=0.01,  # Learning rate.
        z_size=32,  # Size of z variables.
        h_size=160,  # Size of resnet block.
        kl_min=0.25,  # Number of "free bits/nats".
        depth=1,  # Number of downsampling blocks.
        num_blocks=2,  # Number of resnet blocks for each downsampling layer.
        k=1,  # Number of samples for IS objective.
        dataset="cifar10",  # Dataset name.
        image_size=None,  # Image size, automatically determined (input is ignored).
        enable_iaf=True,  # True for IAF, False for Gaussian posterior
        bidirectional=True,  # True for bidirectional, False for bottom-up inference
        path=".", # Dataset path
        compression_always_variable=False,
        compression_exclude_sizes=False
    )


def images(hps):
    cifar10 = lambda: map(itemgetter(0), observations.cifar10(hps.path))
    imagenet32 = lambda: map(lambda x: np.transpose(x, (0, 3, 1, 2)),
                             observations.small32_imagenet(hps.path))
    imagenet64 = lambda: map(lambda x: np.transpose(x, (0, 3, 1, 2)),
                             observations.small64_imagenet(hps.path))

    def quartered(dataset, times=1):
        return lambda: map(lambda x: quarter_images_repeatedly(x, times=times), dataset())

    def quarter_images_repeatedly(x, times):
        for _ in range(times):
            x = quarter_images(x)

        return x

    def quarter_images(x):
        size = x.shape[2]

        quarters = [
            x[:, :, :size // 2, :size // 2],
            x[:, :, size // 2:, :size // 2],
            x[:, :, :size // 2, size // 2:],
            x[:, :, size // 2:, size // 2:]
        ]

        return np.concatenate(quarters)

    datasets = {
        "cifar10": cifar10,
        "cifar10to16": quartered(cifar10),
        "cifar10to8": quartered(cifar10, times=2),
        "small32_imagenet": imagenet32,
        "small32to16_imagenet": quartered(imagenet32),
        "small32to8_imagenet": quartered(imagenet32, times=2),
        "small64_imagenet": imagenet64,
        "small64to32_imagenet": quartered(imagenet64),
        "small64to16_imagenet": quartered(imagenet64, times=2),
        "small64to8_imagenet": quartered(imagenet64, times=3),
    }

    test_images_name = "test_images"
    if hps.dataset.startswith(test_images_name):
        indices =  range(14) if hps.dataset == test_images_name else \
            [int(i) for i in hps.dataset[len(test_images_name):].split(';')]

        hps.eval_batch_size = 1
        hps.batch_size = 1
        return None, [test_image(index) for index in indices]

    if hps.dataset.startswith("test_image"):
        index = int(hps.dataset[len("test_image"):])

        hps.eval_batch_size = 1
        hps.batch_size = 1
        return None, test_image(index)

    if hps.dataset.startswith("sampling_test_images"):
        resolution = int(hps.dataset[len("sampling_test_images"):])

        hps.eval_batch_size = 1
        hps.batch_size = 1
        return None, sampling_testimages(resolution=resolution)

    if hps.dataset.startswith("sampling_test_image"):
        index = int(hps.dataset[len("sampling_test_image"):])

        hps.eval_batch_size = 1
        hps.batch_size = 1
        return None, sampling_testimage(index)

    if hps.dataset.startswith("half_sampling_test_image"):
        index = int(hps.dataset[len("half_sampling_test_image"):])

        hps.eval_batch_size = 1
        hps.batch_size = 1
        return None, sampling_testimage(index, resolution=1200)

    return datasets[hps.dataset]()


def run(hps):
    train_images, _ = images(hps)
    hps.image_size = validate_and_get_image_size(train_images)

    # To avoid error due to GraphDef being over 2GB
    # (https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays):
    images_placeholder = tf.placeholder(train_images.dtype, train_images.shape)

    iterator = tf.data.Dataset.from_tensor_slices(images_placeholder). \
        shuffle(10000, reshuffle_each_iteration=True).repeat(). \
        batch(batch_size=hps.batch_size).make_initializable_iterator()

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        x = tf.reshape(iterator.get_next(), (-1, 3, *hps.image_size))

        # Data-dependent initialization causes freeze during cycle detection (TF bug?):
        # hps.num_gpus = 1
        # init_x = x[:hps.batch_size, :, :, :]
        # init_model = CVAE1(hps, "init", init_x)

        # vs.reuse_variables()
        hps.num_gpus = FLAGS.num_gpus
        model = CVAE1(hps, "train", x)

    saver = tf.train.Saver(max_to_keep=2)

    total_size = 0
    for v in tf.trainable_variables():
        total_size += np.prod([int(s) for s in v.get_shape()])
    print("Num trainable variables: %d" % total_size)

    init_op = tf.global_variables_initializer()

    def init_fn(ses):
        print("Initializing parameters.")
        ses.run(iterator.initializer, feed_dict={images_placeholder: train_images})
        # XXX(rafal): TensorFlow bug?? Default initializer should handle things well..
        # ses.run(init_model.h_top.initializer)
        ses.run(init_op)
        print("Initialized!")

    sv = Supervisor(is_chief=True,
                    logdir=FLAGS.logdir + "/train/{}_{}".format(strftime('%Y%m%d-%H%M%S'), FLAGS.hpconfig),
                    summary_op=None,  # Automatic summaries don"t work with placeholders.
                    saver=saver,
                    global_step=model.global_step,
                    save_summaries_secs=120,
                    save_model_secs=0,
                    init_op=None,
                    init_fn=init_fn)

    print("starting training")
    local_step = 0
    begin = time.time()

    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config) as sess:
        print("Running first iteration!")
        while not sv.should_stop():
            fetches = [model.bits_per_dim, model.global_step, model.dec_log_stdv, model.train_op]

            should_compute_summary = (local_step % 20 == 19)
            if should_compute_summary:
                fetches += [model.summary_op]

            fetched = sess.run(fetches)

            if should_compute_summary:
                sv.summary_computed(sess, fetched[-1])

            if local_step < 10 or should_compute_summary:
                print("Iteration %d, time = %.2fs, train bits_per_dim = %.4f, dec_log_stdv = %.4f"
                      % (fetched[1], time.time() - begin, fetched[0], fetched[2]))
                begin = time.time()
            if np.isnan(fetched[0]):
                print("NAN detected!")
                break
            if local_step % 3000 == 0:
                saver.save(sess, sv.save_path, global_step=sv.global_step, write_meta_graph=False)

            local_step += 1
        sv.stop()


def run_eval(hps):
    _, datasets = images(hps)

    all_channel_counts = []
    all_average_bits = []

    for test_images in datasets if isinstance(datasets, list) else [datasets]:
        tf.reset_default_graph()
        hps.num_gpus = 1
        hps.batch_size = hps.eval_batch_size
        hps.image_size = validate_and_get_image_size(test_images)

        # To avoid error due to GraphDef being over 2GB
        # (https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays):
        images_placeholder = tf.placeholder(test_images.dtype, test_images.shape)
        images_iterator = tf.data.Dataset.from_tensor_slices(images_placeholder).repeat(). \
            batch(batch_size=hps.batch_size).make_initializable_iterator()
        x = tf.reshape(images_iterator.get_next(), (-1, 3) + hps.image_size)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            model = CVAE1(hps, "eval", x)
            sample_model = CVAE1(hps, "sample", x)

        saver = tf.train.Saver(model.avg_dict)
        # Use only 4 threads for the evaluation.
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=4)
        sess = tf.Session(config=config)
        sess.run(images_iterator.initializer, feed_dict={images_placeholder: test_images})

        sw = tf.summary.FileWriter(
            str(Path(FLAGS.logdir) / FLAGS.mode / relative_eval_model_path()),
            sess.graph)
        # ckpt_loader = CheckpointLoader(saver, model.global_step, FLAGS.logdir + "/train")

        with sess.as_default():
            # assert dataset.n % hps.batch_size == 0
            epoch_size = int(len(test_images) / hps.batch_size)
            saver.restore(sess, restore_path())

            global_step = 0  # ckpt_loader.last_global_step

            summary = tf.Summary()
            all_bits_per_dim = []
            for _ in tqdm.trange(epoch_size):
                all_bits_per_dim += [sess.run(model.bits_per_dim)]

            all_channel_counts.append(hps.batch_size * np.prod(hps.image_size) * epoch_size)
            average_bits = float(np.mean(all_bits_per_dim))
            all_average_bits.append(average_bits)
            print("Step: %d Score: %.3f" % (global_step, average_bits))
            summary.value.add(tag='eval_bits_per_dim', simple_value=average_bits)

            if False:  # hps.k == 1:
                # show reconstructions from the model
                total_samples = 36
                num_examples = 0
                imgs_inputs = np.zeros([total_samples // 2, *hps.image_size, 3],
                                       np.float32)
                imgs_recs = np.zeros([total_samples // 2, *hps.image_size, 3],
                                     np.float32)
                while num_examples < total_samples // 2:
                    sample_x, batch = sess.run([model.m_trunc[0], model.x])
                    batch_bhwc = np.transpose(batch, (0, 2, 3, 1))
                    img_bhwc = np.transpose(sample_x, (0, 2, 3, 1))

                    if num_examples + hps.batch_size > total_samples // 2:
                        cur_examples = total_samples // 2 - num_examples
                    else:
                        cur_examples = hps.batch_size

                    imgs_inputs[num_examples:num_examples + cur_examples, ...] = img_stretch(
                        batch_bhwc[:cur_examples, ...])
                    imgs_recs[num_examples:num_examples + cur_examples, ...] = img_stretch(
                        img_bhwc[:cur_examples, ...])
                    num_examples += cur_examples

                imgs_to_plot = np.zeros([total_samples, *hps.image_size, 3], np.float32)
                imgs_to_plot[::2, ...] = imgs_inputs
                imgs_to_plot[1::2, ...] = imgs_recs
                imgs = img_tile(imgs_to_plot, aspect_ratio=1.0, border=0).astype(np.float32)
                imgs = np.expand_dims(imgs, 0)
                im_summary = tf.summary.image("reconstructions", imgs, 1)
                summary.MergeFromString(sess.run(im_summary))

                # generate samples from the model
                num_examples = 0
                imgs_to_plot = np.zeros([total_samples, *hps.image_size, 3], np.float32)
                while num_examples < total_samples:
                    sample_x = sess.run(sample_model.m_trunc[0])
                    img_bhwc = img_stretch(np.transpose(sample_x, (0, 2, 3, 1)))

                    if num_examples + hps.batch_size > total_samples:
                        cur_examples = total_samples - num_examples
                    else:
                        cur_examples = hps.batch_size

                    imgs_to_plot[num_examples:num_examples + cur_examples, ...] = img_stretch(
                        img_bhwc[:cur_examples, ...])
                    num_examples += cur_examples

                imgs = img_tile(imgs_to_plot, aspect_ratio=1.0, border=0).astype(np.float32)
                imgs = np.expand_dims(imgs, 0)
                im_summary = tf.summary.image("samples", imgs, 1)
                summary.MergeFromString(sess.run(im_summary))

            sw.add_summary(summary, global_step)
            sw.flush()

    print('Overall average: ' + str(np.average(all_average_bits, weights=all_channel_counts)))


def validate_and_get_image_size(images):
    assert images.shape[1] == 3
    image_dimensions = (images.shape[2], images.shape[3])
    assert image_dimensions[0] % 2 == 0 and image_dimensions[1] % 2 == 0
    return image_dimensions


def rvae_variable_size_codec(codec_from_shape, latent_from_image_shape, image_count,
                             dimensions=4, dimension_bits=16):
    size_append, size_pop = codecs.repeat(codecs.Uniform(dimension_bits), dimensions)

    def append(message, symbol):
        """append sizes and array in alternating order"""
        assert len(symbol.shape) == dimensions

        symbol_append, _ = codec_from_shape(symbol.shape)
        head_size = np.prod(latent_from_image_shape(symbol.shape)) + np.prod(symbol.shape)
        message = codecs.reshape_head(message, (head_size,))
        message = symbol_append(message, symbol)
        message = codecs.reshape_head(message, (1, ))
        message = size_append(message, np.array(symbol.shape))
        return message

    def pop(message):
        message, size = size_pop(message)
        # TODO make codec 0 dimensional:
        size = size[:, 0]
        assert size.shape == (dimensions, )
        size = size.astype(np.int)
        head_size = np.prod(latent_from_image_shape(size)) + np.prod(size)
        _, symbol_pop = codec_from_shape(tuple(size))

        message = codecs.reshape_head(message, (head_size,))
        message, symbol = symbol_pop(message)
        message = codecs.reshape_head(message, (1,))

        return message, symbol

    return codecs.serial([(append, pop)] * image_count)


def serial_with_shapes(symbol_codecs, shapes):
    """
    Applies given codecs in series on a sequence of symbols requiring various ANS stack head shapes.
    The head shape required for each symbol is given through shapes.
    """

    def reshape_append(append, shape, message, symbol):
        message = codecs.reshape_head(message, shape)
        message = append(message, symbol)
        return message

    def reshape_pop(pop, shape, message):
        message = codecs.reshape_head(message, shape)
        message, symbol = pop(message)
        return message, symbol

    create_reshape_append = lambda append, shape: lambda m, s: reshape_append(append, shape, m, s)
    create_reshape_pop = lambda pop, shape: lambda m: reshape_pop(pop, shape, m)

    return codecs.serial([
        (create_reshape_append(append, shape), create_reshape_pop(pop, shape))
        for (append, pop), shape in zip(symbol_codecs, shapes)])


def rvae_variable_known_size_codec(codec_from_image_shape, latent_from_image_shape, image_shapes):
    image_codecs = [codec_from_image_shape(s) for s in image_shapes]
    head_shapes = [(np.prod(latent_from_image_shape(s)) + np.prod(s),) for s in image_shapes]

    return serial_with_shapes(image_codecs, head_shapes)


def run_bbans(hps):
    from autograd.builtins import tuple as ag_tuple
    from rvae.resnet_codec import ResNetVAE

    hps.num_gpus = 1
    batch_size = 1
    hps.batch_size = batch_size
    hps.eval_batch_size = batch_size

    _, datasets = images(hps)
    datasets = datasets if isinstance(datasets, list) else [datasets]
    test_images = [np.array([image]).astype('uint64')
                   for dataset in datasets for image in dataset]

    num_dims = np.sum([batch.size for batch in test_images])

    prior_precision = 10
    obs_precision = 24
    q_precision = 20

    @lru_cache()
    def codec_from_shape(shape):
        print("Creating codec for shape " + str(shape))

        hps.image_size = (shape[2], shape[3])

        z_shape = latent_shape(hps)
        z_size = np.prod(z_shape)

        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                x = tf.placeholder(tf.float32, shape, 'x')
                model = CVAE1(hps, "eval", x)
                stepwise_model = LayerwiseCVAE(model)

        saver = tf.train.Saver(model.avg_dict)
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=4,
                                device_count={'GPU': 0})
        sess = tf.Session(config=config, graph=graph)
        saver.restore(sess, restore_path())

        run_all_contexts, run_top_prior, runs_down_prior, run_top_posterior, runs_down_posterior, \
        run_reconstruction = stepwise_model.get_model_parts_as_numpy_functions(sess)

        # Setup codecs
        def vae_view(head):
            return ag_tuple((np.reshape(head[:z_size], z_shape),
                             np.reshape(head[z_size:], shape)))

        obs_codec = lambda h, z1: codecs.Logistic(*run_reconstruction(h, z1),
                                                  obs_precision, bin_prec=8)

        return codecs.substack(
            ResNetVAE(run_all_contexts,
                      run_top_posterior, runs_down_posterior,
                      run_top_prior, runs_down_prior,
                      obs_codec, prior_precision, q_precision),
            vae_view)

    is_fixed = not hps.compression_always_variable and \
               (len(set([dataset[0].shape[-2:] for dataset in datasets])) == 1)
    fixed_size_codec = lambda: codecs.repeat(codec_from_shape(test_images[0].shape), len(test_images))
    variable_codec_including_sizes = lambda: rvae_variable_size_codec(codec_from_shape,
                                        latent_from_image_shape=latent_from_image_shape(hps),
                                        image_count=len(test_images))
    variable_known_sizes_codec = lambda: rvae_variable_known_size_codec(
        codec_from_image_shape=codec_from_shape,
        latent_from_image_shape=latent_from_image_shape(hps),
        image_shapes=[i.shape for i in test_images])
    variable_size_codec = \
        variable_known_sizes_codec if hps.compression_exclude_sizes else variable_codec_including_sizes
    codec = fixed_size_codec if is_fixed else variable_size_codec
    vae_append, vae_pop = codec()

    rng = np.random.RandomState(0)

    np.seterr(divide='raise')

    # Codec for adding extra bits to the start of the chain (necessary for bits back).
    other_bits_count = 100000000

    encode_t0 = time.time()
    init_head_shape = (np.prod(image_shape(hps)) + np.prod(latent_shape(hps)) if is_fixed else 1,)
    init_message = codecs.random_stack(other_bits_count, init_head_shape, rng)
    init_len = 32 * other_bits_count

    print("Encoding...")
    message = vae_append(init_message, test_images)
    flat_message = codecs.flatten_benford(message)
    encode_t = time.time() - encode_t0

    print("All encoded in {:.2f}s".format(encode_t))

    message_len = 32 * len(flat_message)
    print("Used {} bits.".format(message_len))
    print("This is {:.2f} bits per dim.".format(message_len / num_dims))
    print('Extra bits per dim: {:.2f}'.format((message_len - init_len) / num_dims))
    print('Extra bits: {:.2f}'.format(message_len - init_len))

    ## Decode
    decode_t0 = time.time()
    message = codecs.unflatten_benford(flat_message, init_head_shape)
    message, images_ = vae_pop(message)
    decode_t = time.time() - decode_t0

    print('All decoded in {:.2f}s'.format(decode_t))

    assert len(test_images) == len(images_), (len(test_images), len(images_))
    for test_image, image in zip(test_images, images_):
        np.testing.assert_equal(test_image, image)
    init_head, init_tail = init_message

    message = codecs.reshape_head(message, init_head_shape)
    head, tail = message
    assert init_head.shape == head.shape, (init_head.shape, head.shape)
    assert np.all(init_head == head)

    # use this, or get into recursion issues
    while init_tail:
        el, init_tail = init_tail
        el_, tail = tail
        assert el == el_

def main(_):
    hps = get_default_hparams().parse(FLAGS.hpconfig)
    print(hps)

    fun = {"train": run, "eval": run_eval, "bbans": run_bbans}

    fun[FLAGS.mode](hps)


if __name__ == "__main__":
    tf.app.run()
