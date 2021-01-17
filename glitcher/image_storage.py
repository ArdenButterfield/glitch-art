from filetypes import *
import numpy as np
import sys
from PIL import Image
import io

class ImageStorage:
    def __init__(self, filename):
        lower_name = filename.lower()
        if lower_name.endswith('.wav'):
            # TODO: deal with it.
            pass
        elif lower_name.endswith('.png'):
            self.im_type = PNG
            with open(filename, 'rb') as f:
                self.im_representation = f.read()
        elif lower_name.endswith('.jpg' or '.jpeg'):
            self.im_type = JPEG
            with open(filename, 'rb') as f:
                self.im_representation = f.read()
        else:
            sys.exit(f"File {filename} is not an accepted type.")

    def _as_bytes(self, format):
        assert self.im_type == PIL_ARRAY
        print(self.im_representation)
        byte_array = io.BytesIO()
        if format == PNG:
            self.im_representation.save(byte_array, format='PNG')
        elif format == JPEG:
            self.im_representation.save(byte_array, format='JPEG')
        return byte_array.getvalue()

    def as_numpy_array(self):
        if self.im_type == NP_ARRAY:
            pass
        elif self.im_type == JPEG or self.im_type == PNG:
            self.as_pil_array()
            self.im_representation = np.array(self.im_representation)
        elif self.im_type == WAV:
            # TODO: do something...
            pass
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
        elif self.im_type == WAV:
            # TODO: figure it out
            pass
        elif self.im_type == JPEG or self.im_type == PNG:
            self.im_representation = Image.open(io.BytesIO(self.im_representation))

        self.im_type = PIL_ARRAY
        return self.im_representation

    def as_jpeg(self):
        if self.im_type in [NP_ARRAY, WAV, PNG]:
            self.as_pil_array()
            self.im_representation = self._as_bytes(JPEG)
        elif self.im_type == PIL_ARRAY:
            self.im_representation = self._as_bytes(JPEG)

        self.im_type = JPEG
        return self.im_representation

    def as_png(self):
        if self.im_type in [NP_ARRAY, WAV, JPEG]:
            self.as_pil_array()
            self.im_representation = self._as_bytes(PNG)
        elif self.im_type == PIL_ARRAY:
            self.im_representation = self._as_bytes(PNG)

        self.im_type = JPEG
        return self.im_representation

a = ImageStorage('wb.png')
for i in range(3):
    a.as_numpy_array()
    a.as_png()
    a.as_pil_array()
    a.as_jpeg()
    a.as_png()
    a.as_numpy_array()
    a.as_jpeg()
