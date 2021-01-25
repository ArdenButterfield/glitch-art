import numpy as np

WHITE_CLIP = 255 * 3 + 1
BLACK_CLIP = 0 * 3 - 1

LIGHT_MODE = 1
DARK_MODE = 0

ANY = (0,0)
N = (0,-1)
W = (-1,0)
E = (1,0)
S = (0,1)


class Bug:
    def __init__(self, val_array, mode, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.val_array = val_array
        self.height = len(self.val_array)
        self.width = len(self.val_array[0])
        self.mode = mode
        self.infect_current_pos()
        self.dir = ANY

    def make_move(self):
        if self.dir == ANY:
            if self.mode == LIGHT_MODE:
                v, neighbor = self.min_neighbor()
            else:
                v, neighbor = self._neighbor()
            if self.mode == LIGHT_MODE and v = W

    def infect_current_pos(self):
        if self.mode == LIGHT_MODE:
            self.val_array[self.y][self.x] = WHITE_CLIP
        else:
            self.val_array[self.y][self.x] = BLACK_CLIP

    # TODO: you could probably do this much nicer with numpy.
    # Of interest? https://www.xspdf.com/resolution/56383400.html
    # https://codereview.stackexchange.com/questions/178603/compare-neighbors-in-array
    def min_neighbor(self):
        """
        Return neighboring cell, value that is the minimum
        """
        min_v = WHITE_CLIP
        min_neighbor = (self.x, self.y)
        for x in range(self.x-1, self.x+2):
            for y in range(self.y-1, self.y+2):
                if self._in_grid(x,y):
                    val = self.val_array[x,y]
                    if val < min_v:
                        min_v = val
                        min_neighbor = (x,y)
        return min_v, min_neighbor

    def max_neighbor(self):
        """
        Return neighboring cell, value that is the minimum
        """
        max_v = BLACK_CLIP
        max_neighbor = (self.x, self.y)
        for x in range(self.x-1, self.x+2):
            for y in range(self.y-1, self.y+2):
                if self._in_grid(x,y):
                    val = self.val_array[x,y]
                    if val > max_v:
                        max_v = val
                        max_neighbor = (x,y)
        return max_v, max_neighbor

    def _in_grid(self, x, y):
        return (not(x==self.x and y==self.y)) and x >= 0 and y >= 0 and \
               x < self.width and y < self.height

    def pos_is_infected(self, x, y):
        return (self.mode == LIGHT_MODE and
                self.val_array[y][x] == WHITE_CLIP) or \
               (self.mode == DARK_MODE and
                self.val_array[y][x] == BLACK_CLIP)


class Board:
    def __init__(self, im_array, mode, starting_num_bugs=8):
        if mode in [LIGHT_MODE, DARK_MODE]:
            self.mode = mode
        else:
            raise ValueError("Board mode must be 0 or 1.")

        self.height = len(im_array)
        self.width = len(im_array[0])
        self.im_array = im_array
        self.val_array = np.sum(im_array, axis=2)
        if self.mode == LIGHT_MODE:
            start_pt = np.argmax(self.val_array)
        else:
            start_pt = np.argmin(self.val_array)
        pos = (start_pt % self.width, start_pt // self.width) # x, y
        self.bugs = [Bug(self.val_array, mode, pos)
                     for i in range(starting_num_bugs)]
