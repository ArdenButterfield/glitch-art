"""
Glitcher

Here it is... the code that makes photos into fun glitch art.

-- Arden Butterfield 2021
"""

# Other peoples' stuff
import numpy as np # For general processing.
import logging
from scipy.io import wavfile # For reading and writing to .wav
import random # for shuffle
from PIL import ImageFilter, ImageEnhance
import IPython # For inline display

# My stuff
from image_storage import ImageStorage
from _glitch_util import \
    _flip_bit_of_byte, \
    _find_start_and_end, \
    _get_even_slices, \
    _cellular_automata, \
    _pad_with_val,\
    _get_2d_automata_num
from bayer import Bayer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

SAMPLERATE = 44100 # For exporting audio


class Glitcher:
    def __init__(self, max_checkpoints=-1):

        self._image = ImageStorage()
        self._max_checkpoints = max_checkpoints
        self._checkpoints = []
        self._num_checkpoints = 0

        self._bayer = Bayer()

    ############################################################################
    #
    # Import/Export methods
    #
    ############################################################################

    def load_image(self, image_file):
        """
        image_file: file name of image to load
        """
        self._image.load_file(image_file)

    def copy(self):
        """
        Create a copy of a Glitcher object.
        """
        copied = Glitcher()
        copied._image = self._image.copy()
        return copied

    def save_image(self, file_name, disp=True):
        """
        Save the image to [file_name]

        disp:
        Should we display the image when we save it?
        """

        image = self._image.as_pil_array()
        image.save(file_name)
        if disp:
            image.show()

    def load_binary(self,
                    filename,
                    width,
                    height,
                    grayscale=False):
        with open(filename, 'rb') as file:
            if grayscale:
                size = width * height
            else:
                size = width * height * 3
            raw_data = file.read(size)
        self._image.load_binary(raw_data, width, height, grayscale)

    def display(self):
        """
        Show the image in a window, using the PIL Image.show() method.
        """
        image = self._image.as_pil_array()
        image.show()

    def display_inline(self):
        """
        Useful for Jupyter Notebooks.
        """
        image = self._image.as_pil_array()
        IPython.display.display(image)

    ############################################################################
    #
    # Checkpoint stuff
    #
    ############################################################################

    def set_checkpoint(self, name=""):
        """
        Set a checkpoint with the current state of the Glitcher object. A name
        can optionally be provided, which makes it easier to jump back to a
        specific checkpoint. If the number of checkpoints set reaches the max
        number of checkpoints, setting another checkpoint will delete the oldest
        checkpoint.
        """

        if self._num_checkpoints == self._max_checkpoints:
            self._checkpoints.pop(0)
            self._num_checkpoints -= 1

        self._checkpoints.append((self.copy(), name))
        self._num_checkpoints += 1

    def undo(self):
        """
        Undo the glitcher object to the most recent checkpoint.
        """

        self.revert_to_checkpoint()

    def revert_to_checkpoint(self, name=""):
        """
        Revert to the most recent checkpoint with name. If name is not provided,
        or is an empty string, revert to the most recent checkpoint.
        """

        if self._num_checkpoints == 0:
            logging.warning("No checkpoints saved, unable to revert.")
            return

        if name:
            index = self._find_checkpoint_index(name)
            self._checkpoints = self._checkpoints[:index + 1]
            self._num_checkpoints = index + 1

        last = self._checkpoints.pop()
        last_object = last[0]
        self._image = last_object.image

        self._num_checkpoints -= 1

    def _find_checkpoint_index(self, name):
        """
        name: checkpoint name to search for.
        returns the most recent index of that checkpoint name.
        """

        for i in range(self._num_checkpoints - 1, -1, -1):
            if self._checkpoints[i][1] == name:
                return i
        return -1

    ############################################################################
    #
    # Wav file methods
    #
    ############################################################################

    def save_wav(self,
                 file_name,
                 mode):
        """
        Save the image to a wav file, or list of wav files. There are three
        modes:
        0: all channels on one file, one after another. [RR...RRGG...GGBB...BB]
        1: all channels on one file, interlaced. [RGBRGB...RGBRGB]
        2: each channel to its own file. [RR...RR], [GG...GG], [BB...BB]

        If mode=2, file should be a list of three filename strings, otherwise it
        should be a single string.

        returns the dimensions of the image as a (width, height) tuple, which
        are needed for reading a wav file.
        """

        im = self._image.as_numpy_array()
        width = len(im[0])
        height = len(im)

        if mode == 0:
            if type(file_name) != str:
                raise TypeError("Filename must be a string for mode 0")
            channels = np.concatenate([im[:, :, channel].flatten()
                                       for channel in range(3)])
            wavfile.write(file_name, SAMPLERATE, channels)

        elif mode == 1:
            if type(file_name) != str:
                raise TypeError("Filename must be a string for mode 1")
            channels = im.flatten()
            wavfile.write(file_name,SAMPLERATE, channels)

        elif mode == 2:
            if type(file_name) != list or \
                    len(file_name) != 3 or \
                    not max([type(file_name[i]) is str for i in range(3)]):
                raise TypeError(
                    "For mode 2, file must be a list of three strings")

            for i in range(3):
                data = im[:, :, i].flatten()
                wavfile.write(file_name[i], SAMPLERATE, data)
        else:
            raise ValueError("Unrecongnized mode")
        return width, height

    def read_wav(self,
                 file_name,
                 mode,
                 dimensions):
        """
        Read the image from a wav file, or list of wav files. There are three
        modes:
        0: all channels on one file, one after another. [RR...RRGG...GGBB...BB]
        1: all channels on one file, interlaced. [RGBRGB...RGBRGB]
        2: each channel to its own file. [RR...RR], [GG...GG], [BB...BB]

        If mode=2, file should be a list of three filename strings, otherwise it
        should be a single string.

        dimensions are the dimensions of the image, represented as a
        (width, height) tuple, which are returned by the save_wav method.
        """

        self._image.as_numpy_array()
        width, height = dimensions

        if mode == 0:
            if type(file_name) != str:
                raise TypeError("Filename must be a string for mode 0")
            data = np.copy(wavfile.read(file_name)[1])

            if data.dtype == "int16":
                data //= 256
                data += 128
            data = data.astype("uint8")
            data = data[:width * height * 3]
            im = np.swapaxes(data.reshape((3 , -1)), 0, 1)\
                .reshape((height, width, 3))

        elif mode == 1:
            if type(file_name) != str:
                raise TypeError("Filename must be a string for mode 1")
            data = np.copy(wavfile.read(file_name)[1])

            if data.dtype == "int16":
                data //= 256
                data += 128
            data = data.astype("uint8")
            data = data[:width * height * 3]
            im = data.reshape((height, width, 3))

        elif mode == 2:
            if type(file_name) != list or \
                    len(file_name) != 3 or \
                    not max([type(file_name[i]) is str for i in range(3)]):
                raise TypeError(
                    "For mode 2, file must be a list of three strings")

            data = np.concatenate(
                [wavfile.read(f)[1][:width * height] for f in file_name]
            ).flatten()
            # [1] since wavfile.read() returns fs, data

            im = np.swapaxes(data.reshape((3, -1)), 0, 1)\
                .reshape((height, width, 3))

        else:
            raise ValueError("Invalid mode.")

        self._image.im_representation = im

    ############################################################################
    #
    # Image file type methods
    #
    ############################################################################

    def get_bmp_dims(self):
        """
        Returns the dimensions dictated by the header of the image's BMP
        representation, as a (width, height) tuple.
        """

        im = self._image.as_bmp()
        im_array = np.array(list(im.getvalue()))

        width = im_array[0x12] + \
                im_array[0x13] * 0x100 + \
                im_array[0x14] * 0x10000 + \
                im_array[0x15] * 0x1000000
        height = im_array[0x16] + \
                im_array[0x17] * 0x100 + \
                im_array[0x18] * 0x10000 + \
                im_array[0x19] * 0x1000000

        return width, height

    def rescale_bmp_dims(self,
                         newdims,
                         filename=""):
        """
        Rescale a bmp image by changing the width and height values in the
        header.

        newdims:
        either the new dimensions that the image should be OR, if the
        either of them are less than zero, the change to the dimensions.

        filename:
        If this is provided, it will write directly to that file name.
        This is useful, since if you continue working with the image, PIL might
        complain that the image is not formatted correctly.
        """

        width, height = newdims

        if width < 0 or height < 0:
            w, h = self.get_bmp_dims()
            width += w
            height += h

        im = self._image.as_bmp()

        im_array = np.array(list(im.getvalue()))

        im_array[0x12] = width & 0xFF
        im_array[0x13] = (width & 0xFF00) >> 0x8
        im_array[0x14] = (width & 0xFF0000) >> 0x10
        im_array[0x15] = (width & 0xFF000000) >> 0x18

        im_array[0x16] = height & 0xFF
        im_array[0x17] = (height & 0xFF00) >> 0x8
        im_array[0x18] = (height & 0xFF0000) >> 0x10
        im_array[0x19] = (height & 0xFF000000) >> 0x18

        if filename:
            with open(filename, 'wb') as f:
                f.write(bytes(list(im_array)))
        else:
            self._image.im_representation.write(bytes(list(im_array)))

    def jpeg_noise(self, quality):
        """
        Saves to a jpeg of quality 0 to 100.
        """

        self._image.as_jpeg(quality)

    def jpeg_bit_flip(self,
                      num_bits,
                      change_bytes=False,
                      filename=""):
        """
        Let's flip some bits in our pretty little jpeg!

        num_bits:
        number of bits we will flip.

        change_bytes:
        if True, we select a byte, rather than a bit,
        and randomize its values. From a visual standpoint, changing this
        doesn't really do much.

        filename:
        If this is provided, it will write directly to that file name.
        This is useful, since if you continue working with the image, PIL might
        complain that the image is not formatted correctly.
        """

        im = self._image.as_jpeg()
        im_array = list(im.getvalue())
        im_size = len(im_array)
        start, end = _find_start_and_end(im_array, im_size)
        bytes_to_flip = np.random.randint(start, end, num_bits)

        for i in bytes_to_flip:
            if change_bytes:
                im_array[i] = np.random.randint(0, 254)
            else:
                im_array[i] = _flip_bit_of_byte(im_array[i],
                                                     np.random.randint(0,8))
        if filename:
            with open(filename, 'wb') as f:
                f.write(bytes(list(im_array)))
        else:
            self._image.im_representation.write(bytes(list(im_array)))

    def shuffle(self,
                format=0,
                random_order=True,
                even_slices=False,
                chunks=2,
                entire_image=True,
                filename=""):
        """
        Cuts the bytes like a deck of cards... well sort of.
        Take a section from somewhere in the middle of the array and put it
        somewhere else.

        format:
        0 -- Numpy array
        1 -- BMP
        2 -- JPEG

        random_order:
        If random_order is true, the ordering of the sliced components will be
        random. Otherwise, they will be put back in the reverse order to how
        they started out.

        even_slices:
        If even_slices is false, the slices will be placed randomly within the
        image. Otherwise, they will be evenly spaced.

        slices:
        This is the number of slices in the shuffle.

        entire_image:
        If true, the entire image will be shuffled. Otherwise, just a randomly
        chosen segment from the middle of the image will be.
        Recommended: use entire_image=False for jpeg images.

        filename:
        If this is provided, it will write directly to that file name.
        This is useful, since if you continue working with the image, PIL might
        complain that the image is not formatted correctly.
        """

        if format == 0: # NUMPY
            im = self._image.as_numpy_array()
            height = len(im)
            width = len(im[0])
            channels = len(im[0][0])
            im_size = width * height
            im_array = np.reshape(im, (im_size, channels))
            start_of_image = 0
            end_of_image = im_size

        elif format == 1: # BMP
            im = self._image.as_bmp()
            im_array = np.array(list(im.getvalue()))

            # Offset to the start of pixel data is stored little endian
            # (at least on my machine) starting at the 10th byte.
            start_of_image = im_array[10] + \
                             im_array[11] * 0x100 + \
                             im_array[12] * 0x10000 + \
                             im_array[13] * 0x1000000

            # if I understand correctly, there is no end of image code.
            end_of_image = len(im_array) - 1

        elif format == 2: # JPEG
            im = self._image.as_jpeg()
            im_array = np.array(list(im.getvalue()))
            im_size = len(im_array)

            # Based on https://docs.fileformat.com/image/jpeg/ and
            # https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format#File_format_structure
            start_of_image, end_of_image = _find_start_and_end(im_array, im_size)
        else:
            raise ValueError("Incorrect mode. Only 0 and 2 are implemented.")

        if not entire_image:
            points = np.random.randint(start_of_image, end_of_image, 2)
            points.sort()
            start_of_image = points[0]
            end_of_image = points[1]

        if even_slices:
            slice_points = _get_even_slices(
                start_of_image, end_of_image, chunks)
        else:
            slice_points = list(np.random.randint(
                start_of_image, end_of_image, chunks - 1))
            slice_points.sort()
            slice_points = [start_of_image] + slice_points + [end_of_image]

        chunk_list =  [im_array[slice_points[i]:slice_points[i+1]] for
                      i in range(chunks)]

        if random_order:
            random.shuffle(chunk_list)
        else:
            chunk_list.reverse()
        chunk_list.insert(0, im_array[:slice_points[0]])
        chunk_list.append(im_array[slice_points[-1]:])
        im_array = np.concatenate(chunk_list, axis=0)

        if format == 0:
            im = np.reshape(im_array, (height, width, channels))
            self._image.im_representation = im
            self.save_image(filename)

        elif format in [1, 2]: # BMP or JPEG
            if filename:
                with open(filename, 'wb') as f:
                    f.write(bytes(list(im_array)))
            else:
                self._image.im_representation.write(bytes(list(im_array)))

    ############################################################################
    #
    # Misc. Glitch methods
    #
    ############################################################################

    def smear(self, distance):
        """
        Moving from the top of the image downwards, if the channel value at a
        particular pixel is less than [distance] away from the channel value at
        the pixel immediately above it, we replace it with the value of the
        channel in the pixel above it.
        """

        im = self._image.as_numpy_array()

        for row in range(1, len(im)):
            mask = abs(im[row] - im[row - 1]) < distance
            im[row][mask] = im[row - 1][mask]

        self._image.im_representation = im

    def srgb_to_rgb(self):
        """
        Using the algorithm in https://surma.dev/things/ditherpunk/, convert
        an srgb image to rgb.
        """

        convert = lambda b: b / 12.92 if \
            (b <= (255 * 0.04045)) else \
            ((((b / 255) + 0.055) / 1.055) ** 2.4) * 255

        vconvert = np.vectorize(convert)
        im = self._image.as_numpy_array()

        self._image.im_representation = vconvert(im)

    def rgb_to_srgb(self):
        """
        Using the algorithm in https://surma.dev/things/ditherpunk/, convert
        an srgb image to rgb.
        """

        convert = lambda b: 12.95 * b if \
            (b <= (0.0031308 * 255)) else \
            (1.055 * (b / 255)**(1/2.4) - 0.055) * 255

        vconvert = np.vectorize(convert)
        im = self._image.as_numpy_array()

        self._image.im_representation = vconvert(im)

    def dither(self,n,initial=-1):
        """
        Using a Bayer algorithm, dither an image.
        n: The number of times we apply the recursive matrix generator rule to
        the initial matrix
        initial: The initial matrix. Defaults to [[0,2],[3,1]]
        """

        if type(initial) is list and type(initial[0]) is list:
            self._bayer = Bayer(initial=initial)

        bayer_matrix = self._bayer.get_scaled_matrix(n, 255)
        im = self._image.as_numpy_array()
        h = len(im)
        w = len(im[0])

        bayer_matrix = (np.tile(bayer_matrix,
                               (int(w / len(bayer_matrix[0])) + 1)))
        bayer_matrix = np.concatenate(
            [bayer_matrix for _ in range(int(h / len(bayer_matrix)) + 1)])

        bayer_matrix = np.repeat(bayer_matrix[:h,:w], 3).reshape((h, w, 3))

        im[im >= bayer_matrix] = 255
        im[im < bayer_matrix] = 0

        self._image.im_representation = im

    def elementary_automata(self, rule=154, cutoff=230):
        """
        Suggested rule: 154
        Does a cellular automata effect to the image. The rule should be an
        integer between 0 and 255.
        See here for more info:
        https://en.wikipedia.org/wiki/Elementary_cellular_automaton
        """

        infection_condition = lambda x: \
            np.all(x > cutoff, axis=1).astype(np.int0)
        symptom = lambda x: 255 - x

        im = self._image.as_numpy_array()
        row_len = len(im[0])
        row_array = np.zeros(row_len, dtype=np.int0)

        for row in range(len(im)):
            row_array = row_array | infection_condition(im[row])

            mask = row_array.astype(bool)
            im[row][mask] = symptom(im[row][mask])

            row_array = _cellular_automata(row_array, row_len, rule)
        self._image.im_representation = im

    def cellular_2d_automata(self,
                             rule,
                             high_cutoff=100):
        """
        Rule can be either a number, or a list that will be converted to a
        number.
        If it's a list: Rule format should be a list of strings, either 5
        characters long made of 1, 0, and *, or 6 characters long, like above,
        with a - at the start. The strings without a minus are the patterns that
        should map to 1, with * as a wildcard. Strings with a minus are the
        patterns that should map to 0. Later strings in the list override
        earlier ones.
        """

        if type(rule) is list:
            rule = _get_2d_automata_num(rule)
        if type(rule) is not int or rule < 0:
            raise ValueError("Automata rule is incorrectly formatted.")
        infection_condition = lambda x: (x > high_cutoff).astype(int)

        im = self._image.as_numpy_array()

        cols = len(im[0])

        center = infection_condition(im)

        up = np.concatenate((center[1:], [center[0]]))
        down = np.concatenate(([center[-1]], center[:-1]))

        right_order = [cols - 1] + [i for i in range(cols - 1)]
        left_order = [i for i in range(1, cols)] + [0]
        left_arr = np.empty_like(left_order)
        right_arr = np.empty_like(right_order)
        left_arr[left_order] = np.arange(cols)
        right_arr[right_order] = np.arange(cols)
        left = center[:, left_order]
        right = center[:, right_order]

        input = (up << 4) + (left << 3) + (center << 2) + (right << 1) + down
        mask = 1 << input

        self._image.im_representation = ((rule & mask) >> input) * 255

    def flatten_reshape(self,
                        newdims,
                        fillwith=0):
        """
        newdims: a width height tuple.
        fillwith: if the image is not large enough for the new dimensions, what
        do we fill the remainder with?
        0 – black
        1 - white
        2 - the image repeated again.
        """
        width, height = newdims
        im = self._image.as_numpy_array()
        im = im.flatten()

        new_len = width * height * 3
        if fillwith == 0:
            im = _pad_with_val(im, new_len, 0)
        elif fillwith == 1:
            im = _pad_with_val(im, new_len, 255)
        elif fillwith == 2:
            im = _pad_with_val(im, new_len, -1)

        self._image.im_representation = im.reshape((width, height, 3))

    def fly_eye(self,
                box_dims,
                scale,
                x_backwards=False,
                y_backwards=False,
                in_place=False):
        """
        A grid, a bit like a fly's eye, or some sort of funky lens.

        box_dims:
        The size of each grid box, as a (width, height) tuple

        scale:
        How much area should be inside the box. I.e. scale=2 means that
        an image twice the size of the box is shrunken down to fit in the box.

        x_backwards:
        loop backwards horizontally.

        y_backwards:
        loop backwards vertically.

        in_place:
        Modifies the image in place, instead of working with a copy.
        Note that if in_place=False, x_backwards and y_backwards will have no
        effect.
        """

        im = self._image.as_numpy_array()
        if in_place:
            src = im
            dst = im
        else:
            new_im = im.copy()
            src = im
            dst = new_im
        box_w, box_h = box_dims
        y_max = len(im)
        x_max = len(im[0])

        if y_backwards:
            y_0 = y_max - 1
            y_step = -1
            y_stop = -1
        else:
            y_0 = 0
            y_step = 1
            y_stop = y_max

        if x_backwards:
            x_0 = x_max - 1
            x_step = -1
            x_stop = -1
        else:
            x_0 = 0
            x_step = 1
            x_stop = x_max

        for y in range(y_0, y_stop, y_step):
            if not y % 10:
                print(f"{y} of {y_max}")

            y_t = (y // box_h) * box_h
            y_ind = min(int(y_t + scale * (y - y_t)),y_max - 1)

            for x in range(x_0, x_stop, x_step):
                x_t = (x // box_w) * box_w
                x_ind = min(int(x_t + scale * (x - x_t)),x_max - 1)
                dst[y][x] = src[y_ind][x_ind]

        self._image.im_representation = dst

    def pixel_extreme(self, cutoff=-1):
        """
        Send all pixel channels below the cutoff to 0 and all pixel channels
        above the cutoff to 255. If cutoff is negative, or not provided, use the
        mean pixel value as the cutoff.
        """

        im = self._image.as_numpy_array()
        if cutoff < 0:
            cutoff = np.mean(im)
        im[im >= cutoff] = 255
        im[im < cutoff] = 0
        # I love numpy
        self._image.im_representation = im

    def edge_detect(self,
                    kernel_dims=(3, 3),
                    kernel=(-1, -1, -1, -1, 8, -1, -1, -1, -1)):
        """
        ... Or whatever you want it to be. Note that the maximum kernel
        dimensions are (5,5) because of PIL.
        source:
        https://www.geeksforgeeks.org/python-edge-detection-using-pillow/
        """

        im = self._image.as_pil_array()
        im = im.convert("L")
        im = im.filter(ImageFilter.Kernel(kernel_dims, kernel, 1, 0))
        im = im.convert("RGB")
        self._image.im_representation = im

    ############################################################################
    #
    # Utility image methods
    #
    ############################################################################

    def get_dimensions(self):
        """
        Returns width, height
        """

        im = self._image.as_numpy_array()
        return len(im[0]),len(im)

    def make_grayscale(self):
        """
        Convert the image to grayscale.
        """

        im = self._image.as_pil_array()
        im = im.convert('LA').convert('RGB')
        self._image.im_representation = im

    def invert_colors(self):
        """
        Invert the colors to their opposite.
        """

        im = self._image.as_numpy_array()
        self._image.im_representation = 255 - im

    def rotate(self, turns):
        """
        Rotate clockwise by the given number of turns.
        """

        im = self._image.as_numpy_array()
        self._image.im_representation = np.rot90(im, turns, (1,0))

    def horizontal_flip(self):
        """
        Flip the image horizontally.
        """

        im = self._image.as_numpy_array()
        self._image.im_representation = np.fliplr(im)

    def vertical_flip(self):
        """
        Flip the image vertically.
        """

        im = self._image.as_numpy_array()
        self._image.im_representation = np.flipud(im)

    def contrast(self, value):
        self.enhance("contrast", value)

    def color(self, value):
        self.enhance("color", value)

    def brightness(self, value):
        self.enhance("brightness", value)

    def sharpness(self, value):
        self.enhance("sharpness", value)

    def enhance(self, style, value):
        """
        Some basic image enhancing, using our trusty friend PIL. Options for
        style are: color, contrast, brightness, and sharpness.
        """
        
        im = self._image.as_pil_array()
        if style == "color":
            enhancer = ImageEnhance.Color(im)
        elif style == "contrast":
            enhancer = ImageEnhance.Contrast(im)
        elif style == "brightness":
            enhancer = ImageEnhance.Brightness(im)
        elif style == "sharpness":
            enhancer = ImageEnhance.Sharpness(im)
        else:
            raise ValueError("Unrecognized style value in enhance method.")

        self._image.im_representation = enhancer.enhance(value)
