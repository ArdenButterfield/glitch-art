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
import json # For the logging structure
import copy  # for deepcopy method
from PIL import ImageFilter


# My stuff
from image_storage import ImageStorage
from _glitch_util import \
    _flip_bit_of_byte, \
    _find_start_and_end, \
    _get_even_slices, \
    _cellular_automata, \
    _pad_with_val
import bug_eater
from bayer import Bayer

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

SAMPLERATE = 44100 # For exporting audio

# TODO: test on more diverse images! specifically the pride flags lol.
class Glitcher:
    def __init__(self, log_file="", max_checkpoints=-1):

        self._dispatch = {

            'load_image': self.load_image,
            'save_wav': self.save_wav,
            'read_wav': self.read_wav,
            'set_checkpoint': self.set_checkpoint,
            'revert_to_checkpoint': self.revert_to_checkpoint,
            'invert_colors': self.invert_colors,
            'rotate': self.rotate,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'save_image': self.save_image

        }
        self.image = ImageStorage()
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.num_checkpoints = 0

        self.bayer = Bayer()

        if not log_file:
            # No log file provided, so we are starting from scratch.
            self.logging = True
            self.log = []
        else:
            self._reconstruct_from_log(log_file)

    ############################################################################
    #
    # Import/Export methods
    #
    ############################################################################

    def _reconstruct_from_log(self, log_file):
        self.logging = False
        with open(log_file) as f:
            self.log = json.load(f)
        for line in self.log:
            # TODO: there should only be one func on the line.
            #  Right? Why do I have it this way?
            for func_name in line:
                if func_name in self._dispatch:
                    func = self._dispatch[func_name]
                    args = line[func_name]
                    func(*args)
                elif func_name == 'save_image':
                    # To avoid overwriting or cluttering things, we won't write
                    # to image files while reconstructing.
                    pass
                else:
                    logging.warning(
                        "Unrecognized function name when reconstructing log: "
                        + func_name)
                    logging.warning("Skipping it.")

        # Now that we are done reconstructing the log, we want to be adding to
        # it, going forward.
        self.logging = True

    def load_image(self, image_file:str):

        if self.logging:
            self.log.append({"load_image":[image_file]})

        self.image.load_file(image_file)

    def copy(self):
        """
        Create a copy of a Glitcher object.
        Note that we do not log these copies. TODO: should we?
        """
        copied = Glitcher()
        copied.image = self.image.copy()
        copied.log = copy.deepcopy(self.log)
        return copied

    def save_image(self, file_name:str):
        """
        Save the image to [file_name], and saves the log to [file_name].txt
        TODO: should we be logging saving images?
        """

        if self.logging:
            self.log.append({"save_image": [file_name]})

        image = self.image.as_pil_array()
        image.show()
        logging.info("Saving image")
        image.save(file_name)

        if self.logging:
            logging.info("Writing log")
            with open(f"{file_name}.json", 'w') as log_file:
                log_file.write(json.dumps(self.log,indent=4))

    def load_binary(self, filename, width, height, grayscale=False):
        with open(filename, 'rb') as file:
            if grayscale:
                size = width * height
            else:
                size = width * height * 3
            raw_data = file.read(size)
        self.image.load_binary(raw_data, width, height, grayscale)

    def display(self):
        """
        Show the image in a window, using the PIL Image.show() method.
        Don't log display. TODO: should we?
        """
        image = self.image.as_pil_array()
        image.show()

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

        if self.logging:
            self.log.append({"set_checkpoint":[name]})

        if self.num_checkpoints == self.max_checkpoints:
            self.checkpoints.pop(0)
            self.num_checkpoints -= 1

        self.checkpoints.append((self.copy(), name))
        self.num_checkpoints += 1

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

        if self.logging:
            self.log.append({"revert_to_checkpoint":[name]})

        if self.num_checkpoints == 0:
            logging.warning("No checkpoints saved, unable to revert.")
            return

        if name:
            index = self._find_checkpoint_index(name)
            self.checkpoints = self.checkpoints[:index + 1]
            self.num_checkpoints = index + 1

        last = self.checkpoints.pop()
        last_object = last[0]
        self.image = last_object.image
        # TODO: This is kinda clunky,
        #  and I still need to figure out the logging.

        self.num_checkpoints -= 1
        return

    def _find_checkpoint_index(self, name):
        """
        name: checkpoint name to search for.
        returns the most recent index of that checkpoint name.
        """
        for i in range(self.num_checkpoints - 1, -1, -1):
            if self.checkpoints[i][1] == name:
                return i
        return -1

    ############################################################################
    #
    # Wav file methods
    #
    ############################################################################

    def save_wav(self, file, mode):
        """
        Save the image to a wav file, or list of wav files. There are three
        modes:
        0: all channels on one file, one after another. [RR...RRGG...GGBB...BB]
        1: all channels on one file, interlaced. [RGBRGB...RGBRGB]
        2: each channel to its own file. [RR...RR], [GG...GG], [BB...BB]

        If mode=2, file should be a list of three filename strings, otherwise it
        should be a single string.

        returns the dimensions of the image, which are needed for reading a wav
        file.
        """
        logging.info("Saving to wav file")
        if self.logging:
            self.log.append({"save_wav":[file, mode]})
        im = self.image.as_numpy_array()
        width = len(im[0])
        height = len(im)

        if mode == 0:
            if type(file) != str:
                raise TypeError("Filename must be a string for mode 0")
            channels = np.concatenate([im[:, :, channel].flatten()
                                       for channel in range(3)])
            wavfile.write(file, SAMPLERATE, channels)

        elif mode == 1:
            if type(file) != str:
                raise TypeError("Filename must be a string for mode 1")
            channels = im.flatten()
            wavfile.write(file,SAMPLERATE, channels)

        elif mode == 2:
            if type(file) != list or \
                    len(file) != 3 or \
                    not max([type(file[i]) is str for i in range(3)]):
                raise TypeError(
                    "For mode 2, file must be a list of three strings")

            for i in range(3):
                data = im[:, :, i].flatten()
                wavfile.write(file[i], SAMPLERATE, data)
        else:
            raise ValueError("Unrecongnized mode")
        return width, height

    def read_wav(self, file, mode, dimensions):
        """
        Read the image from a wav file, or list of wav files. There are three
        modes:
        0: all channels on one file, one after another. [RR...RRGG...GGBB...BB]
        1: all channels on one file, interlaced. [RGBRGB...RGBRGB]
        2: each channel to its own file. [RR...RR], [GG...GG], [BB...BB]

        If mode=2, file should be a list of three filename strings, otherwise it
        should be a single string.

        dimensions are the dimensions of the image, which are returned by
        the save_wav method.
        """
        logging.info("Reading wav file")
        if self.logging:
            self.log.append({'read_wav':[file, mode, dimensions]})
        self.image.as_numpy_array()
        width, height = dimensions

        if mode == 0:
            if type(file) != str:
                raise TypeError("Filename must be a string for mode 0")
            data = wavfile.read(file)[1]

            if data.dtype == "int16":
                data //= 256
                data += 128
            data = data.astype("uint8")
            data = data[:width * height * 3]
            im = np.swapaxes(data.reshape((3 , -1)), 0, 1)\
                .reshape((height, width, 3))

        elif mode == 1:
            if type(file) != str:
                raise TypeError("Filename must be a string for mode 1")
            data = wavfile.read(file)[1]

            if data.dtype == "int16":
                data //= 256
                data += 128
            data = data.astype("uint8")
            data = data[:width * height * 3]
            im = data.reshape((height, width, 3))

        elif mode == 2:
            if type(file) != list or \
                    len(file) != 3 or \
                    not max([type(file[i]) is str for i in range(3)]):
                raise TypeError(
                    "For mode 2, file must be a list of three strings")

            data = np.concatenate(
                [wavfile.read(f)[1][:width * height] for f in file]
            ).flatten()
            # [1] since wavfile.read() returns fs, data

            im = np.swapaxes(data.reshape((3, -1)), 0, 1)\
                .reshape((height, width, 3))
        else:
            raise ValueError("Invalid mode.")
        self.image.im_representation = im

    ############################################################################
    #
    # Jpeggish methods
    #
    ############################################################################

    def jpeg_noise(self, quality):
        """
        Saves to a jpeg of quality 0 to 100.
        TODO: Maybe also include subsampling? https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg
        """
        if logging:
            self.log.append({'jpeg_noise':[quality]})
        self.image.as_jpeg(quality)

    def jpeg_bit_flip(self, num_bits, change_bytes=False):
        """
        Let's flip some bits in our pretty little jpeg!
        num_bits: number of bits we will flip.
        change_bytes: if True, we select a byte, rather than a bit,
        and randomize its values. From a visual standpoint, changing this
        doesn't really do much.

        """
        im = self.image.as_jpeg()
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
        self.image.im_representation.write(bytes(list(im_array)))

    def shuffle(self, format=0, random_order=True, even_slices=False, chunks=2,
                entire_image=True):
        """
        Cuts the bytes like a deck of cards... well sort of.
        Take a section from somewhere in the middle of the array and put it
        somewhere else.

        format:
        0 -- Numpy array
        1 -- PNG
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
        Recommended: use entire_image=False for jpeg images. TODO: why?
        """
        if logging:
            self.log.append({'shuffle':[format, random_order, even_slices,
                                        chunks, entire_image]})
        if format == 0:
            im = self.image.as_numpy_array()
            height = len(im)
            width = len(im[0])
            channels = len(im[0][0])
            im_size = width * height
            im_array = np.reshape(im, (im_size, channels))
            start_of_image = 0
            end_of_image = im_size

        elif format == 2:
            im = self.image.as_jpeg()
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
            self.image.im_representation = im
        elif format == 2:
            self.image.im_representation.write(bytes(list(im_array)))

    ############################################################################
    #
    # Misc. Glitch methods
    #
    ############################################################################

    def bayer_filter(self,n):
        """
        TODO: This is inefficient. Make it better.
        """
        bayer_matrix = self.bayer.get_scaled_matrix(n, 255)
        m_dim = len(bayer_matrix)
        im = self.image.as_numpy_array()
        a = len(im)
        for row in range(len(im)):
            print(f"{row} of {a}")
            for col in range(len(im[0])):
                for channel in range(3):
                    if im[row][col][channel] > bayer_matrix[row % m_dim][col % m_dim]:
                        im[row][col][channel] = 255
                    else:
                        im[row][col][channel] = 0
        self.image.im_representation = im

    def elementary_automata(self, rule=154):
        """
        Suggested rule: 154
        Does a cellular automata effect to the image. The rule should be an
        integer between 0 and 255.
        See here for more info:
        https://en.wikipedia.org/wiki/Elementary_cellular_automaton
        """
        infection_condition = lambda x: \
            np.all(x > 230, axis=1).astype(np.int0)
        symptom = lambda x: 255 - x

        im = self.image.as_numpy_array()
        row_len = len(im[0])
        row_array = np.zeros(row_len, dtype=np.int0)

        for row in range(len(im)):
            row_array = row_array | infection_condition(im[row])

            mask = row_array.astype(bool)
            im[row][mask] = symptom(im[row][mask])

            row_array = _cellular_automata(row_array, row_len, rule)
        self.image.im_representation = im

    def flatten_reshape(self, newdims, fillwith=0):
        """
        newdims: a width height tuple.
        fillwith: if the image is not large enough for the new dimensions, what
        do we fill the remainder with?
        0 – black
        1 - white
        2 - the image repeated again.
        """
        width, height = newdims
        im = self.image.as_numpy_array()
        im = im.flatten()

        new_len = width * height * 3
        if fillwith == 0:
            im = _pad_with_val(im, new_len, 0)
        elif fillwith == 1:
            im = _pad_with_val(im, new_len, 255)
        elif fillwith == 2:
            im = _pad_with_val(im, new_len, -1)

        self.image.im_representation = im.reshape((width, height, 3))

    def fly_eye(self,
                box_dims,
                scale,
                x_backwards=False,
                y_backwards=False,
                in_place=False):
        """
        A grid, a bit like a fly's eye, or some sort of funky lens.

        box_dims: A width, height tuple
        scale: How much area should be inside the box. I.e. scale=2 means that
        an image twice the size of the box is shrunken down to fit in the box.
        x_backwards: loop backwards horizontally.
        y_backwards: loop backwards vertically.
        in_place: Modifies the image in place, instead of working with a copy.
        Note that if in_place=False, x_backwards and y_backwards will have no
        effect.
        """
        im = self.image.as_numpy_array()
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

        self.image.im_representation = dst

    def pixel_extreme(self, cutoff):
        """
        Send all pixel channels below the cutoff to 0 and all pixel channels
        above the cutoff to 255. If cutoff is negative, use an auto cutoff.
        (median)
        """
        im = self.image.as_numpy_array()
        if cutoff < 0:
            cutoff = np.mean(im)
        im[im >= cutoff] = 255
        im[im < cutoff] = 0
        # I love numpy
        self.image.im_representation = im

    def edge_detect(self, kernel_dims=(3,3), kernel=(-1, -1, -1, -1, 8, -1, -1, -1, -1)):
        """
        ... Or whatever you want it to be.
        source:
        https://www.geeksforgeeks.org/python-edge-detection-using-pillow/
        """
        im = self.image.as_pil_array()
        im = im.convert("L")
        im = im.filter(ImageFilter.Kernel(kernel_dims, kernel, 1, 0))
        im = im.convert("RGB")
        self.image.im_representation = im

    def to_bug_eater(self, mode):
        """
        mode:
        0: Bugs eat from black
        1: Bugs eat from white

        Create a board object of the image for the bug eater fun!
        """
        im = self.image.as_numpy_array()
        board = bug_eater.Board(im, mode)
        return board

    ############################################################################
    #
    # Utility image methods
    #
    ############################################################################

    def get_dimensions(self):
        """
        Returns width, height
        """
        im = self.image.as_numpy_array()
        return len(im[0]),len(im)

    def make_grayscale(self):
        """
        Convert the image to grayscale.
        """
        im = self.image.as_pil_array()
        im = im.convert('LA').convert('RGB')
        self.image.im_representation = im

    def invert_colors(self):
        """
        Invert the colors to their opposite.
        """
        if self.logging:
            self.log.append({"invert_colors":[]})

        logging.info("Inverting colors")
        im = self.image.as_numpy_array()
        self.image.im_representation = 255 - im

    def rotate(self,turns:int):
        """
        Rotate clockwise by the given number of turns.
        """

        if self.logging:
            self.log.append({"rotate": [turns]})

        logging.info(f"Rotating {turns}")
        im = self.image.as_numpy_array()
        self.image.im_representation = np.rot90(im, turns, (1,0))

    def horizontal_flip(self):
        """
        Flip the image horizontally.
        """

        if self.logging:
            self.log.append({"horizontal_flip": []})

        logging.info(f"Flipping horizontally")
        im = self.image.as_numpy_array()
        self.image.im_representation = np.fliplr(im)

    def vertical_flip(self):
        """
        Flip the image vertically.
        """

        if self.logging:
            self.log.append({"vertical_flip": []})

        logging.info(f"Flipping vertically")
        im = self.image.as_numpy_array()
        self.image.im_representation = np.flipud(im)
