"""
Create matrices for Bayer dithering.
"""

import numpy as np

class Bayer:
    def __init__(self,initial=-1):
        self.matrices = []
        if initial == -1:
            self.initial = np.array([[0,2],[3,1]])
        else:
            self.initial = np.array(initial)
        self.init_height = len(self.initial)
        self.init_width = len(self.initial[0])
        self.init_size = self.init_height * self.init_width
        self.matrices.append(self.initial)
        self.m_len = 1

    def get_matrix(self, n):
        """
        Algorithm described here:
        https://en.wikipedia.org/wiki/Ordered_dithering#Threshold_map
        """
        if self.m_len > n:
            return self.matrices[n]
        else:
            m_prev_scaled = self.get_matrix(n - 1) * self.init_size

            m = np.concatenate(
                [np.concatenate([m_prev_scaled + self.initial[i][j]
                                 for j in range(self.init_width)],axis=1 )
                for i in range(self.init_height)], axis=0)
            self.m_len += 1
            self.matrices.append(m)

            return(m)

    def get_scaled_matrix(self, n, max_val):
        return self.get_matrix(n) * (max_val / (self.init_size ** (n + 1)))
