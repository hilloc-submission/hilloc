import os
from collections import defaultdict

import numpy as np

from rvae.datasets import scale_down_and_round_even


def benchmark(image_paths, formats=('ppm', 'png', 'jp2', 'webp'), max_pixels=None):
    def print_summary(format, channels, size):
        bitsperchannel = size * 8 / channels
        print(f'{format}, {size}, {bitsperchannel:.2f}, {bitsperchannel * 100 / 8:.1f}%')

    import cv2

    print(f'format, size/bytes, bits/channel, compression rate')

    all_sizes = defaultdict(lambda: [])
    all_channels = defaultdict(lambda: [])
    for image_file in image_paths:
        image = scale_down_and_round_even(cv2.imread(str(image_file)), max_pixels=max_pixels)

        resolution = image.shape[:-1]
        print(f'{image_file.name} ({resolution[0]} x {resolution[1]})')
        for format in formats:
            dir = image_file.parent / "out"
            dir.mkdir(exist_ok=True)
            target = str(dir / f'{image_file.stem}.{format}')
            cv2.imwrite(target, image)

            # assert np.array_equal(image, cv2.imread(target)), f'{format} is not lossless.'
            channels = np.prod(image.shape)
            all_channels[format].append(channels)
            size = os.path.getsize(target)
            all_sizes[format].append(size)
            print_summary(format, channels, size)

    print('Overall')
    for format in formats:
        channels = np.sum(all_channels[format])
        size = np.sum(all_sizes[format])
        print_summary(format, channels, size)
