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
    numpy array, return the next row.
    """
    input = (row << 1) + \
            (np.append(row[1:],row[0])) + \
            (np.append(row[-1],row[:-1]) << 2)
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

def _pad_with_val(arr, size, val):
    """
    arr: one dimensional array to be padded
    size: new length of the array
    val: what number the array should be padded with. negative means: pad with
    the image repeated over and over again.
    """
    l = len(arr)
    if l > size:
        return arr[:size]
    elif val < 0:
        repeats = (size // l) + 1
        a =  np.concatenate([arr for i in range(repeats)]).flatten()
        return a[:size].astype("uint8")
    else:
        a = np.concatenate((arr, np.ones(size - l) * val))
        return a.flatten().astype("uint8")

def _valid_automata_(rulebook):
    """
    This could all probably be done easier with regular expressions...
    """
    result = True
    if not type(rulebook) is list:
        result = False
    for rule in rulebook:
        if not (type(rule) is str and (
                len(rule) == 5 or (len(rule) == 6 and rule[0] == '-'))):
            result = False
        for char in rule[-5:]:
            if char not in "*01":
                result = False
    return result

def _get_2d_automata_num(rulebook):
    """
    Rule format should be a list of strings, either 5 characters long made of
    1, 0, and *, or 6 characters long, like above, with a - at the start.
    The strings without a minus are the patterns that should map to 1, with * as
    a wildcard. Strings with a minus are the patterns that should map to 0.
    Later strings in the list override earlier ones. Returns -1 if rule is not
    formatted correctly.

    This is an unbelievably bad algorithm, but it's all I have the energy for
    right now.
    """
    if not _valid_automata_(rulebook):
        return -1
    strs = rulebook.copy()
    i = 0
    while i < len(strs):
        if '*' in strs[i]:
            ind = strs[i].index('*')
            strs.append(strs[i][:ind] + '0' + strs[i][ind + 1:])
            strs.append(strs[i][:ind] + '1' + strs[i][ind + 1:])
            strs[i] = ''
        i += 1
    strs_result = [i for i in strs if i]
    print(strs_result)
    result = 0
    for s in strs_result:
        prefix = s[0]
        s = int(s[-5:], 2) # base 2
        if prefix != '-':
            result |= 1 << s
    for s in strs_result:
        prefix = s[0]
        s = int(s[-5:], 2) # base 2
        if prefix == '-':
            result &= ~(1 << (s))
    return result
