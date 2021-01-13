import numpy as np
from PIL import Image
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class Glitcher:
    def __init__(self, image_file:str, log_file=""):
        self.input_file = image_file
        if not log_file:
            self.logging = True
            self.log = ""
        try:
            image = Image.open(image_file)
        except FileNotFoundError:
            logging.error(f"Usage: cannot find image {image_file}\n")
            return

        image.load()
        self.log += image_file + "\n"
        self.image = np.array(image, dtype="int32")

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

    def save_to(self, file_name:str):
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


a = Glitcher("../images/raw/branches.jpg")
a.invert_colors()
a.vertical_flip()
a.save_to("../images/glitched/test2.png")