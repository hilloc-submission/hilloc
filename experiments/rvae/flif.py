import subprocess

import cv2
import numpy as np

from rvae.datasets import test_image
import craystack as cs


encode_command = f'flif -e - - --effort=100 --no-metadata --no-color-profile --no-crc'
decode_command = f'flif -d - -'
codec = cs.Uniform(8)
len_codec = cs.Uniform(31)


def pop_varint(bytes):
    leading_bit = 1
    i = 0
    content = 0
    mask = (1 << 7) - 1
    while leading_bit:
        byte = bytes[i]
        i += 1
        leading_bit = byte >> 7
        byte_content = mask & byte
        content = (content << 7) + byte_content
    content += 1  # ?!
    return content, bytes[i:]


def im_transform(image):
    return np.swapaxes(image, 0, 2)[:, :, ::-1]


def inverse_im_transform(image):
    return np.swapaxes(image[:, :, ::-1], 0, 2)


def FLIF():
    def push(message, image):
        """expects image to be chw"""
        image = im_transform(image).astype(np.uint8)
        success, im_buffer = cv2.imencode(".ppm", image)
        process = subprocess.run(encode_command.split(),
                                 input=im_buffer.tobytes(),
                                 capture_output=True)
        if process.returncode != 0:
            raise Exception(f"flif encode failed: {process.stderr}")
        compressed_bytes = process.stdout

        # take off the 'FLIF' magic header
        compressed_bytes = compressed_bytes[4:]
        # can also remove RGB interlaced byte and bytes per chan (next two bytes)
        # then there are 3 varints for width, height and number of frames
        # https://flif.info/spec.html for details

        compressed_bits = list(compressed_bytes)  # list of uint8s
        n_compressed_bits = len(compressed_bits)
        cbits_codec = cs.repeat(codec, n_compressed_bits)
        message = cbits_codec.push(message, compressed_bits)
        message = len_codec.push(message, np.uint64(n_compressed_bits))
        return message

    def pop(message):
        message, n_compressed_bits = len_codec.pop(message)
        cbits_codec = cs.repeat(codec, n_compressed_bits[0])
        message, compressed_bits = cbits_codec.pop(message)
        compressed_bits = np.squeeze(compressed_bits).astype(np.uint8)
        bytes_buffer = b'FLIF' + bytes(compressed_bits)
        process = subprocess.run(decode_command.split(),
                                 input=bytes_buffer,
                                 capture_output=True)
        if process.returncode != 0:
            raise Exception(f"flif decode failed: {process.stderr}")
        im_buffer = np.frombuffer(process.stdout, dtype=np.uint8)
        image = cv2.imdecode(im_buffer, flags=1)  # this gives in hwc
        return message, inverse_im_transform(image)
    return cs.Codec(push, pop)

FLIF = FLIF()


if __name__ == '__main__':
    image = test_image(0)[0]
    message = cs.empty_message((1,))
    message = FLIF.push(message, image)
    message, decoded_image = FLIF.pop(message)
    np.testing.assert_equal(image, decoded_image)
