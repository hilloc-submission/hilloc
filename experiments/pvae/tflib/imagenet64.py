import numpy as np
import scipy.misc
import time
import os

def make_generator(path, n_files, batch_size, random_state):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = list(range(n_files))
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(len(str(n_files)))))
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, datapath, rs=np.random):
    return (
        make_generator(os.path.join(datapath, 'imagenet64/train_64x64'), 1281149, batch_size, rs),
        make_generator(os.path.join(datapath, 'imagenet64/valid_64x64'), 49999, batch_size, rs)
    )

def make_generator_from_chunked(path, n_files, batch_size, random_state, chunk):
    if chunk is not None:
        files = ['images_{}.npy'.format(chunk)]
    else:
        files = os.listdir(path)

    def get_epoch():
        for file in files:
            chunk = np.load(os.path.join(path, file))
            n_batchs = chunk.shape[0] // batch_size
            for n in range(n_batchs):
                yield (chunk[n*batch_size:(n+1)*batch_size],)
    return get_epoch

def load_chunked(batch_size, datapath, rs=np.random, chunk=None):
    return (
        make_generator_from_chunked(os.path.join(datapath, 'imagenet64/train_64_chunks'), 1281149, batch_size, rs, chunk),
        make_generator_from_chunked(os.path.join(datapath, 'imagenet64/valid_64_chunks'), 49999, batch_size, rs, chunk)
    )


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()