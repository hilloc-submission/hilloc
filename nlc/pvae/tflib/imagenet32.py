import numpy as np
import scipy.misc
import time
import os

def make_generator(path, n_files, batch_size, random_state):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 32, 32), dtype='int32')
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
        make_generator(os.path.join(datapath, 'imagenet32/train_32x32'), 1281149, batch_size, rs),
        make_generator(os.path.join(datapath, 'imagenet32/valid_32x32'), 49999, batch_size, rs)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()