import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from observations import maybe_download_and_extract

default_max_pixels = 1260 * 840


def image_files_to_array(image_files, max_pixels=default_max_pixels):
    import cv2

    # https://stackoverflow.com/a/15074748/1692437
    return np.array([
        np.swapaxes(scale_down_and_round_even(cv2.imread(str(f)), max_pixels)[:, :, ::-1], 0, 2)
        for f in image_files])


def sampling_testimage_paths(dir=Path('.'), size=2400):
    link = 'https://sourceforge.net/projects/testimages/files/' \
           'SAMPLING/8BIT/RGB/SAMPLING_8BIT_RGB_{}x{}.tar.bz2/download'.format(size, size)
    tar_file = dir / 'sampling_test_images_{}.tar.bz2'.format(size)
    image_dir = dir / 'SAMPLING/8BIT/RGB/{}x{}/C00C00'.format(size, size)

    if not image_dir.exists():
        if not tar_file.exists():
            subprocess.call(['wget', link, '-O' + str(tar_file)])

        with tarfile.open(str(tar_file)) as f:
            f.extractall(dir)

        tar_file.unlink()

        for dir in image_dir.parent.iterdir():
            if dir != image_dir:
                shutil.rmtree(str(dir))

    return sorted(filter(lambda p: p.name.endswith('.png'), image_dir.iterdir()))


def sampling_testimages(dir=Path('.'), resolution=2400):
    image_files = sampling_testimage_paths(dir, size=resolution)
    return image_files_to_array(image_files)


def sampling_testimage(index, dir=Path('.'), resolution=2400):
    image_files = sampling_testimage_paths(dir, size=resolution)
    return image_files_to_array([image_files[index]])


def test_image(index, path=Path('.')):
    image_files = test_image_files(path)
    # assert len(image_files) == 14

    image_file = image_files[index]
    print('Loading ' + str(image_file))

    return image_files_to_array([image_file])


def test_image_files(path=Path('.')):
    path = path / 'test_images'
    maybe_download_and_extract(str(path), 'http://imagecompression.info/test_images/rgb8bit.zip')
    image_files = sorted([p for p in path.iterdir() if p.name.endswith('.ppm')])
    return image_files


def round_to_even(x):
    return int(round(x / 2) * 2)


def scale_down_and_round_even(image, max_pixels=default_max_pixels):
    if max_pixels is None:
        return image

    import cv2

    orig_pixels = np.prod(image.shape[:-1])
    scale = np.sqrt(max_pixels / orig_pixels)
    if scale >= 1:
        return image

    new_x = round_to_even(image.shape[0] * scale)
    new_y = round_to_even(image.shape[1] * scale)

    return cv2.resize(image, (new_y, new_x), interpolation=cv2.INTER_AREA)
