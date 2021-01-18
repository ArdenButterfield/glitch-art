import numpy as np # For general processing.
from PIL import Image # For reading and writing images
import logging
from scipy.io import wavfile # For reading and writing to .wav
import io
from filetypes import *
from image_storage import ImageStorage

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

SAMPLERATE = 44100 # For exporting audio


class Glitcher:
    def __init__(self, log_file="", max_checkpoints=-1):
        if not log_file:
            self.logging = True
            self.log = ""
        self.image = ImageStorage()
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.num_checkpoints = 0

    def load_image(self, image_file:str):
        self.input_file = image_file
        self.log += image_file + "\n"
        self.image.load_file(image_file)

    def save_wav(self, files):
        """Export as a wav file."""
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
        if self.num_checkpoints == 0:
            logging.warning("No checkpoints saved, unable to revert.")
            return

        if name:
            index = self._find_checkpoint_index(name)
            self.checkpoints = self.checkpoints[:index + 1]
            self.num_checkpoints = index + 1

        self = self.checkpoints.pop()[0]
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
        Create a copy of a Glitcher object
        """
        copied = Glitcher()
        copied.image = self.image.copy()
        copied.log = self.log
        copied.input_file = self.input_file
        return copied

    def invert_colors(self):
        """
        Invert the colors to their opposite.
        """
        logging.info("Inverting colors")
        im = self.image.as_numpy_array()
        self.image.im_representation = 255 - im
        self.log += "invert_colors\n"

    def rotate(self,turns:int):
        """
        Rotate clockwise by the given number of turns.
        """
        logging.info(f"Rotating {turns}")
        self.log += f"rotate,{turns}\n"
        im = self.image.as_numpy_array()
        self.image.im_representation = np.rot90(im, turns, (1,0))

    def horizontal_flip(self):
        """
        Flip the image horizontally.
        """
        logging.info(f"Flipping horizontally")
        self.log += "horizontal_flip"
        im = self.image.as_numpy_array()
        self.image.im_representation = np.fliplr(im)

    def vertical_flip(self):
        """
        Flip the image vertically.
        """
        logging.info(f"Flipping vertically")
        self.log += "vertical_flip"
        im = self.image.as_numpy_array()
        self.image.im_representation = np.flipud(im)

    def save_image(self, file_name:str):
        """
        Save the image to [file_name], and saves the log to [file_name].txt
        """
        image = self.image.as_pil_array()
        image.show()
        logging.info("Saving image")
        self.log += f"save_to,{file_name}\n"
        image.save(file_name)

        logging.info("Writing log")
        with open(f"{file_name}.txt", 'w') as log_file:
            log_file.write(self.log)

    def display(self):
        """
        Show the image in a window, using the PIL Image.show() method.
        """
        image = self.image.as_pil_array()
        image.show()
