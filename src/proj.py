import numpy as np
import matplotlib.pyplot as plt


class Img:
    def __init__(self, ima):
        self.ima = ima
        self.color = None
        self.grey = None

    def is_greyscale(self):
        return True

    def glatt(self, alpha):
        plt.ion()
        data = plt.imshow(self.ima)
        plt.draw()
        while True:
            laplace = (self.ima[0:-2, 1:-1] +
                       self.ima[2:, 1:-1] +
                       self.ima[1:-1, 0:-2] +
                       self.ima[1:-1, 2:] -
                       4 * self.ima[1:-1, 1:-1])
            self.ima[1:-1, 1:-1] += alpha * laplace
            self.ima[:, 0] = self.ima[:, 1]
            self.ima[:, -1] = self.ima[:, -2]
            self.ima[0, :] = self.ima[1, :]
            self.ima[-1, :] = self.ima[-2, :]
            data.set_array(self.ima)
            plt.draw()
            plt.pause(1e-4)


im = Img(plt.imread('GreyCat.png'))
im.glatt(0.05)
