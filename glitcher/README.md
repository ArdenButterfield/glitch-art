# Glitcher Documentation

### Glitcher(max_checkpoints=-1)

Initialization of Glitcher object. If max_checkpoints is a positive integer, then that is the maximum number of checkpoints that will be stored. Otherwise, there is no limit to the number of checkpoints.

## Import/Export Methods

### Glitcher.load_image(filename)

**image_file**: file name of image to load, as a string.

### Glitcher.copy()

Returns a copy of a Glitcher object. Checkpoints are not copied.

### Glitcher.save_image(filename, disp=True)

Save the image to **file_name** (string)

**disp**: Should we display the image when we save it?

### Glitcher.load_binary(filename, width, height, grayscale=False)

Load contents of **filename** as binary data into an image of **width** and **height**, both integers. If grayscale=False, the image is in color, otherwise black and white.

### Glitcher.display()

Show the image in a window, using the PIL Image.show() method.

### Glitcher.display_inline()

Display the image inline. Useful for Jupyter Notebooks.

## Checkpoint Methods

### Glitcher.set_checkpoint(name="")

Set a checkpoint with the current state of the Glitcher object. A name can optionally be provided, which makes it easier to jump back to a specific checkpoint. If the number of checkpoints set reaches the max number of checkpoints, setting another checkpoint will delete the oldest checkpoint.

### Glitcher.undo()

Undo the glitcher object to the most recent checkpoint.

### Glitcher.revert_to_checkpoint(name="")

Revert to the most recent checkpoint with name. If name is not provided, or is an empty string, revert to the most recent checkpoint.

## Audio Methods

### Glitcher.save_wav(filename, mode)

Save the image to a wav file, or list of wav files. There are three **mode**s:

0: all channels on one file, one after another. \[RR...RRGG...GGBB...BB\]

1: all channels on one file, interlaced. \[RGBRGB...RGBRGB\]

2: each channel to its own file. \[RR...RR\], \[GG...GG\], \[BB...BB\]

If mode=2, file should be a list of three filename strings, otherwise it should be a single string.

**Returns** the dimensions of the image as a (width, height) tuple, which are needed for reading a wav file.

### Glitcher.load_wav(filename, mode, dimensions):

Read the image from a wav file, or list of wav files. There are three **mode**s:

0: all channels on one file, one after another. [RR...RRGG...GGBB...BB]

1: all channels on one file, interlaced. [RGBRGB...RGBRGB]

2: each channel to its own file. [RR...RR], [GG...GG], [BB...BB]

If mode=2, file should be a list of three filename strings, otherwise it should be a single string.

**dimensions** are the dimensions of the image, represented as a (width, height) tuple, which are returned by the save_wav method.

## Image File Processing Methods

### Glitcher.get_bmp_dims()

**Returns** the dimensions dictated by the header of the image's BMP representation, as a (width, height) tuple.

### Glitcher.rescale_bmp_dims(newdims, filename="")

Rescale a bmp image by changing the width and height values in the
header.

**newdims**: Either the new dimensions that the image should be OR, if the either of them are less than zero, the change to the dimensions.

**filename**: If this is provided, it will write directly to that file name, and the rescaling will not effect the internal image represerntation. This is useful, since if you continue working with the image, PIL might complain that the image is not formatted correctly.


### Glitcher.jpeg_noise(quality)

Saves to a jpeg of **quality** 0 to 100.


### Glitcher.jpeg_bit_flip(num_bits, change_bytes=False, filename="")

Let's flip some bits in our pretty little jpeg!

**num_bits**: Number of bits we will flip.

**change_bytes**: if True, we select a byte, rather than a bit, and randomize its values. From a visual standpoint, changing this doesn't really do much.

**filename**: If this is provided, it will write directly to that file name, and the rescaling will not effect the internal image represerntation. This is useful, since if you continue working with the image, PIL might complain that the image is not formatted correctly.


### Glitcher.shuffle(format=0, random_order=True, even_slices=False, chunks=2, entire_image=True, filename="")

Cuts the bytes like a deck of cards... well sort of.
Take a section from somewhere in the middle of the array and put it
somewhere else.

**format**:

0: Numpy array

1: BMP

2: JPEG

**random_order**: If random_order is true, the ordering of the sliced components will be random. Otherwise, they will be put back in the reverse order to how they started out.

**even_slices**: If even_slices is false, the slices will be placed randomly within the image. Otherwise, they will be evenly spaced.

**slices**: This is the number of slices in the shuffle.

**entire_image**: If true, the entire image will be shuffled. Otherwise, just a randomly chosen segment from the middle of the image will be. Recommended: use entire_image=False for jpeg images.

**filename**: If this is provided, it will write directly to that file name, and the rescaling will not effect the internal image represerntation. This is useful, since if you continue working with the image, PIL might complain that the image is not formatted correctly.

## Miscellaneous Glitch Methods

### Glitcher.smear(distance)

Moving from the top of the image downwards, if the channel value at a particular pixel is less than **distance** away from the channel value at the pixel immediately above it, we replace it with the value of the channel in the pixel above it.
        
### Glitcher.srgb_to_rgb()

Using the algorithm in https://surma.dev/things/ditherpunk/, convert an srgb image to rgb.

### Glitcher.rgb_to_srgb()

Using the algorithm in https://surma.dev/things/ditherpunk/, convert an rgb image to srgb.

### Glitcher.dither(n, initial=-1)

Using a Bayer algorithm, dither an image.

**n**: The number of times we apply the recursive matrix generator rule to the initial matrix

**initial**: The initial matrix. Defaults to \[\[0,2\],\[3,1\]\]

### Glitcher.elementary_automata(rule=154, cutoff=230)

Suggested rule: 154

Does a cellular automata effect to the image. The **rule** should be an integer between 0 and 255. 

**cutoff**: Any pixels brighter than this value will trigger that cell into an on state.

See here for more info: https://en.wikipedia.org/wiki/Elementary_cellular_automaton
        
### Glitcher.cellular_2d_automata(rule, high_cutoff=100):

**rule** can be either a number, or a list that will be converted to a
number.

If it's a list: Rule format should be a list of strings, either 5
characters long made of 1, 0, and \*, or 6 characters long, like above,
with a - at the start. The strings without a minus are the patterns that
should map to 1, with \* as a wildcard. Strings with a minus are the
patterns that should map to 0. Later strings in the list override
earlier ones.

**high_cutoff**: Before starting the automata algorithm, values above the high cutoff are sent to 255, values below it to 0.

### Glitcher.flatten_reshape(newdims, fillwith=0):

**newdims**: a width height tuple.

**fillwith**: if the image is not large enough for the new dimensions, what do we fill the remainder with?

0 – black

1 - white

2 - the image repeated again.
### Glitcher.fly_eye(box_dims, scale, x_backwards=False, y_backwards=False, in_place=False):

A grid, a bit like a fly's eye, or some sort of funky lens.

**box_dims**: The size of each grid box, as a (width, height) tuple

**scale**: How much area should be inside the box. I.e. scale=2 means that an image twice the size of the box is shrunken down to fit in the box.

**x_backwards**: Loop backwards horizontally.

**y_backwards**: Loop backwards vertically.

**in_place**: Modifies the image in place, instead of working with a copy. Note that if in_place=False, x_backwards and y_backwards will have no effect.

### Glitcher.pixel_extreme(cutoff=-1):

Send all pixel channels below the cutoff to 0 and all pixel channels above the cutoff to 255. If cutoff is negative, or not provided, use the mean pixel value as the cutoff.

### Glitcher.edge_detect(self, kernel_dims=(3, 3), kernel=(-1, -1, -1, -1, 8, -1, -1, -1, -1)):

Edge detect... or whatever you want it to be. Note that the maximum kernel
dimensions are (5,5) because of PIL.
source:
https://www.geeksforgeeks.org/python-edge-detection-using-pillow/

## Utility Image Methods

### Glitcher.get_dimensions()

**Returns** (width, height) in pixels.

### Glitcher.make_grayscale()

Convert the image to grayscale.

### Glitcher.invert_colors()

Invert the colors to their opposite.

### Glitcher.rotate(turns)

Rotate clockwise by the given number of **turns**.
### Glitcher.horizontal_flip()

Flip the image horizontally.

### Glitcher.vertical_flip()

Flip the image vertically.

### Glitcher.contrast(value)

Modify contrast by **value** (float, 1 keeps the image as is).

### Glitcher.color(value)

Modify color saturation by **value** (float, 1 keeps the image as is).

### Glitcher.brightness(value)

Modify brightness by **value** (float, 1 keeps the image as is).
### Glitcher.sharpness(value)

Modify sharpness by **value** (float, 1 keeps the image as is).

### Glitcher.enhance( style, value)

Some basic image enhancing, using our trusty friend PIL. Options for **style** are: color, contrast, brightness, and sharpness.
