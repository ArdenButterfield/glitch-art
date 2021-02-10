import numpy as np
class Bayer:
    def __init__(self,initial=-1):
        self.matrices = []
        if initial == -1:
            self.initial = np.array([[0,2],[3,1]])
        else:
            self.initial = np.array(initial)
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
            m_prev_scaled = self.get_matrix(n - 1) * 4

            top = np.concatenate(
                (m_prev_scaled + self.initial[0][0],
                 m_prev_scaled + self.initial[0][1]), axis=1)
            bottom = np.concatenate(
                (m_prev_scaled + self.initial[1][0],
                 m_prev_scaled + self.initial[1][1]), axis=1)
            m = np.concatenate((top,bottom),axis=0)

            self.m_len += 1
            self.matrices.append(m)

            return(m)

    def get_scaled_matrix(self, n, max_val):
        return self.get_matrix(n) * (max_val / (4 ** (n + 1)))
