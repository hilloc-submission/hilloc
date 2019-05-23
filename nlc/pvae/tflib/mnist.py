import numpy

import os
from urllib import request
import gzip
import pickle

def mnist_generator(data, batch_size, n_labelled, rs=numpy.random):
    images, targets = data

    images = images.astype('float32')
    targets = targets.astype('int32')
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = rs.get_state()
        rs.shuffle(images)
        rs.set_state(rng_state)
        rs.shuffle(targets)

        if n_labelled is not None:
            rs.set_state(rng_state)
            rs.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, datapath, n_labelled=None, rs=numpy.random):
    mnist_dir = os.path.join(datapath, 'mnist')
    if not os.path.isdir(mnist_dir):
        os.mkdir(mnist_dir)

    filename = 'mnist.pkl.gz'
    filepath = os.path.join(mnist_dir, filename)
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in datapath, downloading...")
        request.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f, encoding='bytes')

    return (
        mnist_generator(train_data, batch_size, n_labelled, rs=rs), 
        mnist_generator(dev_data, test_batch_size, n_labelled, rs=rs), 
        mnist_generator(test_data, test_batch_size, n_labelled, rs=rs)
    )
