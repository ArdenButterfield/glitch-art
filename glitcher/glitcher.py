import numpy as np # For general processing.
import sys # For exiting with error messages.
from PIL import Image # For reading and writing images
import logging
from scipy.io import wavfile # For reading and writing to .wav
import io
from filetypes import *
from image_storage import ImageStorage
import json # For the logging structure
import copy  # for deepcopy method


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

SAMPLERATE = 44100 # For exporting audio


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

        if not log_file:
            # No log file provided, so we are starting from scratch.
            self.logging = True
            self.log = []
        else:
            self._reconstruct_from_log(log_file)

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
                assert TypeError("Filename must be a string for mode 0")
            channels = np.concatenate([im[:, :, channel].flatten()
                                       for channel in range(3)])
            wavfile.write(file, SAMPLERATE, channels)

        elif mode == 1:
            if type(file) != str:
                assert TypeError("Filename must be a string for mode 1")
            channels = im.flatten()
            wavfile.write(file,SAMPLERATE, channels)

        elif mode == 2:
            if type(file) != list or \
                    len(file) != 3 or \
                    not max([type(file[i]) is str for i in range(3)]):
                assert TypeError(
                    "For mode 2, file must be a list of three strings")

            for i in range(3):
                data = im[:, :, i].flatten()
                wavfile.write(file[i], SAMPLERATE, data)
        else:
            assert ValueError("Unrecongnized mode")
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
                assert TypeError("Filename must be a string for mode 0")
            data = wavfile.read(file)[1]
            if data.dtype == "int16":
                data //= 256
                data += 128
            data = data.astype("uint8")

            # TODO: this is very baaaad. You should use a numpy meeeethod...
            i = 0
            im = np.swapaxes(data.reshape((3,-1)),0,1).reshape(height, width, 3)



        elif mode == 1:
            if type(file) != str:
                assert TypeError("Filename must be a string for mode 1")
            data = wavfile.read(file)[1]
            if data.dtype == "int16":
                data //= 256
                data += 128
            data = data.astype("uint8")
            im = data.reshape((height, width, 3))

        elif mode == 2:
            if type(file) != list or \
                    len(file) != 3 or \
                    not max([type(file[i]) is str for i in range(3)]):
                assert TypeError(
                    "For mode 2, file must be a list of three strings")

            data = np.concatenate([wavfile.read(f)[1] for f in file]).flatten()
            # [1] since wavfile.read() returns fs, data
            im = np.swapaxes(data.reshape((3,-1)),0,1).reshape(height, width, 3)
        self.image.im_representation = im

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
        start, end = self._find_start_and_end(im_array, im_size)
        bytes_to_flip = np.random.randint(start, end, num_bits)
        for i in bytes_to_flip:
            if change_bytes:
                im_array[i] = np.random.randint(0, 254)
            else:
                im_array[i] = self._flip_bit_of_byte(im_array[i],
                                                     np.random.randint(0,8))
        self.image.im_representation.write(bytes(list(im_array)))

    def _flip_bit_of_byte(self, byte, bit):
        mask = 1 << bit
        return byte ^ mask


    def _find_start_and_end(self, jpeg_image, im_size):
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

        TODO: well the numpy array (the boring one) works like a charm... but
        the jpeg file is getting corrupted and angry. !?
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
            start_of_image, end_of_image = self._find_start_and_end(im_array, im_size)

        if not entire_image:
            points = np.random.randint(start_of_image, end_of_image, 2)
            points.sort()
            start_of_image = points[0]
            end_of_image = points[1]

        if even_slices:
            slice_points = self._get_even_slices(
                start_of_image, end_of_image, chunks)
        else:
            slice_points = np.random.randint(
                start_of_image, end_of_image, chunks - 1)
            slice_points.sort()
            slice_points = np.concatenate([
                [start_of_image], slice_points, [end_of_image]])

        chunk_list =  [im_array[slice_points[i]:slice_points[i+1]] for
                      i in range(chunks)]

        if random_order:
            np.random.shuffle(chunk_list)
        else:
            chunk_list = np.flip(chunk_list)
        chunk_list = [i for i in chunk_list] # TODO: booo... janky alert... and doesn't even work!!
        chunk_list.insert(0, im_array[:slice_points[0]])
        chunk_list.append(im_array[slice_points[-1]:])
        im_array = np.concatenate(chunk_list, axis=0)

        if format == 0:
            im = np.reshape(im_array, (height, width, channels))
            self.image.im_representation = im
        elif format == 2:
            self.image.im_representation.write(bytes(list(im_array)))

    def _get_even_slices(self, start, end, chunks):
        chunk_size = (end - start) // (chunks)
        return np.array([start + i * chunk_size for i in range(chunks + 1)])

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

    def copy(self):
        """
        Create a copy of a Glitcher object.
        Note that we do not log these copies. TODO: should we?
        """
        copied = Glitcher()
        copied.image = self.image.copy()
        copied.log = copy.deepcopy(self.log)
        return copied

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

    def display(self):
        """
        Show the image in a window, using the PIL Image.show() method.
        Don't log display. TODO: should we?
        """
        image = self.image.as_pil_array()
        image.show()
