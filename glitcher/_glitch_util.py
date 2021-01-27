"""
Boring utility functions that help with glitcher.py
"""
import numpy as np

def _flip_bit_of_byte(byte, bit):
    mask = 1 << bit
    return byte ^ mask

def _get_even_slices(start, end, chunks):
    chunk_size = (end - start) // (chunks)
    return [start + i * chunk_size for i in range(chunks + 1)]

def _cellular_automata(row, row_len, rule):
    """
    Given a rule of universal cellular automata and a row, as a one-dimensional
    numpy array, return the
    next row
    """
    input = (row << 1) + (np.append(row[1:],row[0])) + (np.append(row[-1],row[:-1]) << 2)
    mask = 1 << input
    return (rule & mask) >> input

def _find_start_and_end(jpeg_image, im_size):
    """
    jpeg image is a list or numpy array of the bytes of the jpeg image.
    We use FFDA as the start code and FFD9 as the end code, and we return
    the first byte after FFDA, and the first byte of FFD9
    """
    found = False
    for i in range(im_size - 1):
        if jpeg_image[i] == 0xFF and jpeg_image[i + 1] == 0xDA:
            start_of_image = i + 2
            found = True
            break
    if not found:
        raise ValueError('Image is not formatted as a correct jpeg.')
    found = False
    for i in range(im_size - 1, start_of_image, -1):
        if jpeg_image[i] == 0xFF and jpeg_image[i + 1] == 0xD9:
            end_of_image = i - 1
            found = True
            break
    if not found:
        raise ValueError('Image is not formatted as a correct jpeg.')
    return start_of_image, end_of_image