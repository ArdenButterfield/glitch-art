"""
ImageStorage class

This class facilitates conversion between image formats, Python Image Library
images, and numpy arrays. I will likely implement more image file formats soon,
and I may implement storing/loading files to wav, though that is a more
unconventional way to store images, so it may not make sense to handle them in
this class.

-- Arden Butterfield, 2021
"""

from filetypes import *
import numpy as np
import sys
from PIL import Image, ImageFile
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True  # otherwise we get more errors when
                                        # messing up jpeg images.

class ImageStorage:
    def __init__(self, im_type=None, im_rep=None):
        self.im_type = im_type
        self.im_representation = im_rep

    def load_file(self, filename):
        lower_name = filename.lower()
        if lower_name.endswith('.png'):
            self.im_type = PNG
            with open(filename, 'rb') as f:
                self.im_representation = io.BytesIO(f.read())
        elif lower_name.endswith('.jpg' or '.jpeg'):
            self.im_type = JPEG
            with open(filename, 'rb') as f:
                self.im_representation = io.BytesIO(f.read())
        else:
            sys.exit(f"File {filename} is not an accepted type.")

    def _as_bytes(self, format):
        assert self.im_type == PIL_ARRAY
        byte_array = io.BytesIO()
        if format == PNG:
            self.im_representation.save(byte_array, format='PNG')
        elif format == JPEG:
            self.im_representation.save(byte_array, format='JPEG')
        return byte_array

    def as_numpy_array(self):
        if self.im_type == NP_ARRAY:
            pass
        elif self.im_type == JPEG or self.im_type == PNG:
            self.as_pil_array()
            self.im_representation = np.array(self.im_representation)
        elif self.im_type == PIL_ARRAY:
            self.im_representation = np.array(self.im_representation)

        self.im_type = NP_ARRAY
        return self.im_representation

    def as_pil_array(self):
        if self.im_type == NP_ARRAY:
            arr = Image.fromarray(np.uint8(self.im_representation))
            print(arr)
            self.im_representation = arr
        elif self.im_type == PIL_ARRAY:
            pass
        elif self.im_type == JPEG or self.im_type == PNG:
            self.im_representation = Image.open(self.im_representation)

        self.im_type = PIL_ARRAY
        return self.im_representation

    def as_jpeg(self, quality=-1):
        if type(quality) not in [int, float]:
            sys.exit("Invalid quality parameter in as_jpg. Must be a number")
        if quality < 0:
            if self.im_type in [NP_ARRAY, PNG]:
                self.as_pil_array()
                self.im_representation = self._as_bytes(JPEG)
            elif self.im_type == PIL_ARRAY:
                self.im_representation = self._as_bytes(JPEG)
        else:
            self.as_pil_array()
            byte_array = io.BytesIO()
            self.im_representation.save(byte_array,
                                        format='JPEG',
                                        quality=int(min(quality , 100)))
            self.im_representation = byte_array
        self.im_type = JPEG
        return self.im_representation

    def as_png(self):
        if self.im_type in [NP_ARRAY, JPEG]:
            self.as_pil_array()
            self.im_representation = self._as_bytes(PNG)
        elif self.im_type == PIL_ARRAY:
            self.im_representation = self._as_bytes(PNG)

        self.im_type = JPEG
        return self.im_representation

    def copy(self):
        if self.im_type in [NP_ARRAY, PIL_ARRAY]:
            rep = self.im_representation.copy()

        elif self.im_type in [PNG, JPEG]:
            rep = io.BytesIO(self.im_representation.getvalue())
        else:
            sys.exit("TODO: copy other types of data?")

        return ImageStorage(im_type=self.im_type, im_rep=rep)