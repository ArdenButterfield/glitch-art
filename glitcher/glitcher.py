import numpy as np # For general processing.
from PIL import Image # For reading and writing images
import logging
from scipy.io import wavfile # For reading and writing to .wav

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

SAMPLERATE = 44100 # For exporting audio


class Glitcher:
    def __init__(self, log_file=""):
        if not log_file:
            self.logging = True
            self.log = ""
        self.input_file = None
        self.image = None

    def load_image(self, image_file:str):
        self.input_file = image_file
        try:
            image = Image.open(image_file)
        except FileNotFoundError:
            logging.error(f"Usage: cannot find image {image_file}\n")
            return

        image.load()
        self.log += image_file + "\n"
        self.image = np.array(image)

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

    def copy(self):
        """Create a copy of a Glitcher object"""
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
        self.image = 255 - self.image
        self.log += "invert_colors\n"

    def rotate(self,turns:int):
        """
        Rotate clockwise by the given number of turns.
        """
        logging.info(f"Rotating {turns}")
        self.log += f"rotate,{turns}\n"
        self.image = np.rot90(self.image, turns, (1,0))

    def horizontal_flip(self):
        """
        Flip the image horizontally.
        """
        logging.info(f"Flipping horizontally")
        self.log += "horizontal_flip"
        self.image = np.fliplr(self.image)

    def vertical_flip(self):
        """
        Flip the image vertically.
        """
        logging.info(f"Flipping vertically")
        self.log += "vertical_flip"
        self.image = np.flipud(self.image)

    def save_image(self, file_name:str):
        """
        Save the image to [file_name], and saves the log to [file_name].txt
        """
        image = Image.fromarray(np.uint8(self.image))
        image.show()
        logging.info("Saving image")
        self.log += f"save_to,{file_name}\n"
        image.save(file_name)

        logging.info("Writing log")
        with open(f"{file_name}.txt", 'w') as log_file:
            log_file.write(self.log)

    def display(self):
        image = Image.fromarray(np.uint8(self.image))
        image.show()

