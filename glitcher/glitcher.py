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
        if not log_file:
            self.logging = True
            self.log = []
        self.image = ImageStorage()
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.num_checkpoints = 0

    def load_image(self, image_file:str):

        if self.logging:
            self.log.append({"load_image":[image_file]})

        self.image.load_file(image_file)

    def save_wav(self, files):
        sys.exit("Don't use these methods for now, they're broken") # TODO
        """Export as a wav file."""

        if self.logging:
            self.log.append({"save_wav":[files]})

        if len(files) != 3:
            logging.error("There must be 3 wav files")
        for f in files:
            if not f.endswith(".wav"):
                logging.error("To save as a wav, file must end with .wav")
        channels = [self.image[:, :, channel].flatten() for channel in range(3)]
        width = len(self.image[0])
        height = len(self.image)
        for i in range(3):
            wavfile.write(files[i], SAMPLERATE, channels[i])
        # TODO: figure out a more elegant way to do all of this.
        return width, height

    def read_wav(self, files, dimensions):
        sys.exit("Don't use these methods for now, they're broken") # TODO

        if self.logging:
            self.log.append({'read_wav':[files, dimensions]})

        if len(files) != 3:
            logging.error("There must be 3 wav files")
        data = [wavfile.read(file)[1] for file in files]
        # [1] since wavfile.read() returns fs, data
        for i in data:
            if i.dtype == "int16":
                i //= 256
                i += 128
                i = i.astype("uint8")

        width, height = dimensions
        self.image = np.zeros((height, width, 3), dtype="int8")
        i = 0
        for row in self.image:
            for col in row:
                for channel in range(3):
                    col[channel] = data[channel][i]
                i += 1

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
        self.log += "invert_colors\n"

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
        """

        if self.logging:
            self.log.append({"save_image": [file_name]})

        image = self.image.as_pil_array()
        image.show()
        logging.info("Saving image")
        image.save(file_name)

        if self.logging:
            logging.info("Writing log")
            with open(f"{file_name}.txt", 'w') as log_file:
                log_file.write(json.dumps(self.log,indent=4))

    def display(self):
        """
        Show the image in a window, using the PIL Image.show() method.
        Don't log display. TODO: should we?
        """
        image = self.image.as_pil_array()
        image.show()
