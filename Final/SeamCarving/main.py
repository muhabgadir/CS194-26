import numpy as np
import skimage.io as io
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from scipy.signal import convolve
import cv2 as cv
import tkinter as tk
from PIL import ImageTk, Image
import argparse

class seam_carver:
    def __init__(self, image_path, energy, spread, spread_decay, k_size, N):
        self.__image__ = io.imread(image_path)
        self.__image__ = np.pad(self.__image__, pad_width=((10, 10), (10, 10), (0, 0)), mode="symmetric")
        self.spread = spread
        self.spread_decay = spread_decay
        self.k_size = k_size
        self.__N__ = N
        
        if energy == "Heuristic":
            func_ls = [self.__hog_gauss_energy__, self.__hog_energy__, self.__grad_gauss_energy__, self.__grad_energy__,
                       self.__my_energy__, self.__scharr_energy__, self.__scharr_gauss_energy__]
            energy_func = self.__pick_energy_heuristic__(func_ls)
            print("Function picked: ", energy_func.__name__)
            self.__energy_map__ = energy_func()

        elif energy == "HoGauss":
            self.__energy_map__ = self.__hog_gauss_energy__()

        elif energy == "HoG":
            self.__energy_map__ = self.__hog_energy__()

        elif energy == "Grad":
            self.__energy_map__ = self.__grad_energy__()

        elif energy == "Gauss":
            self.__energy_map__ = self.__grad_gauss_energy__()

        elif energy == "My":
            self.__energy_map__ = self.__my_energy__()

        elif energy == "Scharr":
            self.__energy_map__ = self.__scharr_energy__()

        elif energy == "ScharrGauss":
            self.__energy_map__ = self.__scharr_gauss_energy__()

        else:
            raise ValueError("Unrecognized Energy Method!")

        self.__energy_map__ = self.__energy_map__[10:-10, 10:-10]
        self.__image__ = io.imread(image_path)
        self.__vertical_seams__ = self.__compute_vertical_seams__()
        self.__horizontal_seams__ = self.__compute_horizontal_seams__()

    def __pick_energy_heuristic__(self, func_ls):
        heur_ls = []
        for func in func_ls:
            self.__energy_map__ = func()
            vert = self.__compute_vertical_seams__()
            hor = self.__compute_horizontal_seams__()
            heur_ls.append([self.__get_adj_minimmum_coef__(vert[-1]) +\
                            self.__get_adj_minimmum_coef__(hor[:, -1]), func])

        return max(heur_ls, key=lambda x: x[0])[1]

    def __get_adj_minimmum_coef__(self, arr_in):
        arr = arr_in.copy()
        prev_idx = np.argmin(arr)
        np.delete(arr, prev_idx)
        overall_val = 0
        max_val = np.max(arr)
        for i in range(self.__N__):
            next_idx = np.argmin(arr)
            overall_val += abs(next_idx - prev_idx)
            prev_idx = next_idx
            arr[prev_idx] = max_val

        return overall_val

    def get_image(self):
        return self.__image__

    def __my_energy__(self):
        D_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        D_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        im = cv.GaussianBlur(im, (self.k_size, self.k_size), 5)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        new = imx + imy
        imx2 = np.abs(convolve2d(new, D_x, mode="same"))
        imy2 = np.abs(convolve2d(new, D_y, mode="same"))
        energy = new + imx2 + imy2
        return energy/np.max(energy)

    def __color_gauss_energy__(self):
        D_x = [[[1, 1, 1], [-1, -1, -1]], [[0, 0, 0], [0, 0, 0]]]
        D_y = [[[1, 1, 1], [0, 0, 0]], [[-1, -1, -1], [0, 0, 0]]]
        im = self.__image__
        im = cv.GaussianBlur(im, (self.k_size, self.k_size), 0)
        imx = np.abs(convolve(im, D_x, mode="same"))
        imy = np.abs(convolve(im, D_y, mode="same"))
        energy = imx + imy
        new_energy = np.sum(energy, axis=2)
        return new_energy/np.max(new_energy)

    def __color_grad_energy__(self):
        D_x = [[[1, 1, 1], [-1, -1, -1]], [[0, 0, 0], [0, 0, 0]]]
        D_y = [[[1, 1, 1], [0, 0, 0]], [[-1, -1, -1], [0, 0, 0]]]
        im = self.__image__
        imx = np.abs(convolve(im, D_x, mode="same"))
        imy = np.abs(convolve(im, D_y, mode="same"))
        energy = imx + imy
        new_energy = np.sum(energy, axis=2)
        return new_energy/np.max(new_energy)

    def __grad_gauss_energy__(self):
        D_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        D_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        im = cv.GaussianBlur(im, (self.k_size, self.k_size), 5)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        energy = imx + imy

        return energy/np.max(energy)

    def __grad_energy__(self):
        D_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        D_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        energy = imx + imy

        return energy/np.max(energy)

    def __scharr_energy__(self):
        D_x = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
        D_y = [[3, 10, 3], [0, 0, 0], [-10, -10, -3]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        energy = imx + imy

        return energy/np.max(energy)

    def __scharr_gauss_energy__(self):
        D_x = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
        D_y = [[3, 10, 3], [0, 0, 0], [-10, -10, -3]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        im = cv.GaussianBlur(im, (self.k_size, self.k_size), 5)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        energy = imx + imy

        return energy/np.max(energy)
    def __hog_energy__(self):
        D_x = [[1, -1], [0, 0]]
        D_y = [[1, 0], [-1, 0]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        energy = imx + imy
        energy = energy/np.max(hog(self.__image__, pixels_per_cell=(11, 11)))
        return energy/np.max(energy)

    def __hog_gauss_energy__(self):
        D_x = [[1, -1], [0, 0]]
        D_y = [[1, 0], [-1, 0]]
        im = cv.cvtColor(self.__image__, cv.COLOR_RGB2GRAY)
        im = cv.GaussianBlur(im, (self.k_size, self.k_size), 5)
        imx = np.abs(convolve2d(im, D_x, mode="same"))
        imy = np.abs(convolve2d(im, D_y, mode="same"))
        energy = (imx + imy)/np.max(hog(self.__image__, pixels_per_cell=(11, 11)))
        return energy/np.max(energy)

    def __compute_vertical_seams__(self):
        seams = self.__energy_map__.copy()
        for i in range(1, seams.shape[0]):
            for j in range(seams.shape[1]):
                to_add_ls = [seams[i-1, j]]
                if j != 0:
                    to_add_ls.append(seams[i-1, j-1])
                if j != seams.shape[1]-1:
                    to_add_ls.append(seams[i-1, j+1])
                seams[i, j] += min(to_add_ls)

        return seams/np.max(seams)

    def __compute_horizontal_seams__(self):
        seams = self.__energy_map__.copy()
        for j in range(1, seams.shape[1]):
            for i in range(seams.shape[0]):
                to_add_ls = [seams[i, j-1]]
                if i != 0:
                    to_add_ls.append(seams[i-1, j-1])
                if i != seams.shape[0]-1:
                    to_add_ls.append(seams[i+1, j-1])

                seams[i, j] += min(to_add_ls)

        return seams/np.max(seams)

    def get_seams(self):
        stack = np.vstack([self.__vertical_seams__ * 255, self.__horizontal_seams__ * 255])
        return stack

    def trim_vertical(self):
        acu_seam_ls = self.__vertical_seams__[-1]
        to_remove = np.zeros_like(self.__vertical_seams__).astype(np.bool)
        min_seam = min(acu_seam_ls)
        min_idx = np.where(acu_seam_ls == min_seam)
        min_idx = min_idx[0][0]
        acu_seam_ls = np.delete(acu_seam_ls, min_idx)

        col = min_idx
        for j in range(self.__vertical_seams__.shape[0] - 1):
            row = self.__vertical_seams__.shape[0] - (j+1)
            to_remove[row, col] = True

            new_col_list = []
            new_col_list.append((self.__vertical_seams__[row-1, col], col))
            if col != 0:
                new_col_list.append((self.__vertical_seams__[row-1, col-1], col-1))
                self.__vertical_seams__[row, col-1] += self.__vertical_seams__[row, col] * self.spread

            if col != self.__vertical_seams__.shape[1] - 1:
                new_col_list.append((self.__vertical_seams__[row-1, col+1], col+1))
                self.__vertical_seams__[row, col+1] += self.__vertical_seams__[row, col] * self.spread

            col = min(new_col_list, key=lambda x: x[0])[1]

        to_remove[0, col] = True

        new_img = np.zeros((self.__image__.shape[0], self.__image__.shape[1]-1, 3))
        new_energy = np.zeros((self.__image__.shape[0], self.__image__.shape[1]-1))
        new_vert_seams = np.zeros((self.__vertical_seams__.shape[0], self.__vertical_seams__.shape[1]-1))
        new_hor_seams = np.zeros((self.__vertical_seams__.shape[0], self.__vertical_seams__.shape[1]-1))
        to_remove = (-1 * to_remove + 1).astype(np.bool)

        for row in range(self.__image__.shape[0]):
            new_vert_seams[row] = self.__vertical_seams__[row, to_remove[row]]
            new_hor_seams[row] = self.__horizontal_seams__[row, to_remove[row]]
            new_img[row] = self.__image__[row, to_remove[row]]
            new_energy[row] = self.__energy_map__[row, to_remove[row]]

        self.__vertical_seams__ = new_vert_seams/np.max(new_vert_seams)
        self.__horizontal_seams__ = new_hor_seams/np.max(new_hor_seams)
        self.__image__ = new_img
        self.__energy_map__ = new_energy/np.max(new_energy)
        self.spread *= self.spread_decay

    def trim_horizontal(self):
        acu_seam_ls = self.__horizontal_seams__[:, -1]
        to_remove = np.zeros_like(self.__horizontal_seams__).astype(np.bool)
        min_seam = min(acu_seam_ls)
        min_idx = np.where(acu_seam_ls == min_seam)
        min_idx = min_idx[0][0]
        acu_seam_ls = np.delete(acu_seam_ls, min_idx)

        row = min_idx
        for i in range(self.__horizontal_seams__.shape[1] - 1):
            col = self.__horizontal_seams__.shape[1] - (i+1)
            to_remove[row, col] = True

            new_row_list = []
            new_row_list.append((self.__horizontal_seams__[row, col-1], row))

            if row != 0:
                new_row_list.append((self.__horizontal_seams__[row-1, col-1], row-1))
                self.__horizontal_seams__[row-1, col] += self.__horizontal_seams__[row, col] * self.spread

            if row != self.__horizontal_seams__.shape[0] - 1:
                new_row_list.append((self.__horizontal_seams__[row+1, col-1], row+1))
                self.__horizontal_seams__[row+1, col] += self.__horizontal_seams__[row, col] * self.spread

            row = min(new_row_list, key=lambda x: x[0])[1]

        to_remove[row, 0] = True

        new_img = np.zeros((self.__image__.shape[0]-1, self.__image__.shape[1], 3))
        new_energy = np.zeros((self.__image__.shape[0]-1, self.__image__.shape[1]))
        new_vert_seams = np.zeros((self.__horizontal_seams__.shape[0]-1, self.__horizontal_seams__.shape[1]))
        new_hor_seams = np.zeros((self.__horizontal_seams__.shape[0]-1, self.__horizontal_seams__.shape[1]))
        to_remove = (-1 * to_remove + 1).astype(np.bool)

        for col in range(self.__image__.shape[1]):
            new_vert_seams[:, col] = self.__vertical_seams__[to_remove[:, col], col]
            new_hor_seams[:, col] = self.__horizontal_seams__[to_remove[:, col], col]
            new_img[:, col] = self.__image__[to_remove[:, col], col]
            new_energy[:, col] = self.__energy_map__[to_remove[:, col], col]

        self.__vertical_seams__ = new_vert_seams/np.max(new_vert_seams)
        self.__horizontal_seams__ = new_hor_seams/np.max(new_hor_seams)
        self.__image__ = new_img
        self.__energy_map__ = new_energy/np.max(new_energy)
        self.spread *= self.spread_decay

    def get_minimum_vertical_seams(self, N):
        new_image = self.__image__.copy()
        vertical_seams = self.__vertical_seams__.copy()
        acu_seam_ls = vertical_seams[-1]

        for i in range(N):
            min_seam = min(acu_seam_ls)
            min_idx = np.where(acu_seam_ls == min_seam)
            min_idx = min_idx[0][0]
            acu_seam_ls = np.delete(acu_seam_ls, min_idx)
            col = min_idx
            for j in range(self.__vertical_seams__.shape[0] - 1):
                row = self.__vertical_seams__.shape[0] - (j+1)

                new_image[row, col] = [0, (N - i)/N * 255, 0]
                vertical_seams[row, col] = 10000000

                new_col_list = []
                new_col_list.append((vertical_seams[row-1, col], col))
                if col != 0:
                    new_col_list.append((vertical_seams[row-1, col-1], col-1))
                    vertical_seams[row, col-1] += vertical_seams[row, col] * self.spread

                if col != vertical_seams.shape[1] - 1:
                    new_col_list.append((vertical_seams[row-1, col+1], col+1))
                    vertical_seams[row, col+1] += vertical_seams[row, col] * self.spread

                col = min(new_col_list, key=lambda x: x[0])[1]

        return new_image

    def get_minimum_horizontal_seams(self, N):
        new_image = self.__image__.copy()
        horizontal_seams = self.__horizontal_seams__.copy()
        acu_seam_ls = horizontal_seams[:, -1]

        for i in range(N):
            min_seam = min(acu_seam_ls)
            min_idx = np.where(acu_seam_ls == min_seam)
            min_idx = min_idx[0][0]
            acu_seam_ls = np.delete(acu_seam_ls, min_idx)

            row = min_idx
            for j in range(self.__horizontal_seams__.shape[1] - 1):
                col = self.__horizontal_seams__.shape[1] - (j+1)
                new_image[row, col] = [(N - i)/N * 255, 0, 0]
                horizontal_seams[row, col] = 10000000

                new_row_list = []
                new_row_list.append((horizontal_seams[row, col-1], row))

                if row != 0:
                    new_row_list.append((horizontal_seams[row-1, col-1], row-1))
                    horizontal_seams[row-1, col] += horizontal_seams[row, col] * self.spread

                if row != horizontal_seams.shape[0] - 1:
                    new_row_list.append((horizontal_seams[row+1, col-1], row+1))
                    horizontal_seams[row+1, col] += horizontal_seams[row, col] * self.spread

                row = min(new_row_list, key=lambda x: x[0])[1]

        return new_image

    def get_energy_map(self):
        return self.__energy_map__ * 255


class window_UI():
    def __init__(self, image_path, energy="Hog", N=10, spread=0.0, spread_decay=1., k_size=3):
        self.sm = seam_carver(image_path, energy, spread, spread_decay, k_size, N)
        self.__image_path__ = image_path
        self.N = N
        self.root = tk.Tk()
        self.root.title("Seam Carver!")
        initial_image = ImageTk.PhotoImage(Image.fromarray(self.sm.get_image()))
        self.window = tk.Label(self.root, width=self.sm.get_image().shape[1],
                               height=self.sm.get_image().shape[0],
                               image=initial_image)
        self.window.image = initial_image
        self.window.place(x=0, y=0)
        self.window.pack()

        self.window.bind('<Configure>', self.__update__)
        self.__get_func__ = self.sm.get_image
        self.__showing__ = "Image"

        self.root.bind('<KeyPress>', self.__keyPress__)

        tk.mainloop()

    def __force_update__(self):
        n_image = ImageTk.PhotoImage(Image.fromarray(np.uint8(self.__get_func__())))
        self.window.configure(image=n_image)
        self.window.image = n_image

    def __update__(self, event):
        if event.width < self.sm.get_image().shape[1]:
            for i in range(self.sm.get_image().shape[1] - event.width):
                self.sm.trim_vertical()

        if event.height < self.sm.get_image().shape[0]:
            for i in range(self.sm.get_image().shape[0] - event.height):
                self.sm.trim_horizontal()

        self.__force_update__()

    def __keyPress__(self, event):
        if event.char == "q":
            self.root.destroy()
        if event.char == "e":
            if self.__showing__ == "Energy":
                self.__get_func__ = self.sm.get_image
                self.__showing__ = "Image"
                self.__force_update__()
            else:
                self.__get_func__ = self.sm.get_energy_map
                self.__showing__ = "Energy"
                self.__force_update__()

        if event.char == "s":
            if self.__showing__ == "EnergyPaths":
                self.__get_func__ = self.sm.get_image
                self.__showing__ = "Image"
                self.root.resizable(True, True)
                self.window.configure(height=self.sm.get_image().shape[0])
                self.root.geometry('{}x{}'.format(self.sm.get_image().shape[1], self.sm.get_image().shape[0]))
                self.__force_update__()
            else:
                self.__get_func__ = self.sm.get_seams
                self.__showing__ = "EnergyPaths"
                self.root.resizable(False, False)
                self.window.configure(height=self.sm.get_image().shape[0]*2)
                self.root.geometry('{}x{}'.format(self.sm.get_image().shape[1], self.sm.get_image().shape[0]*2))
                self.__force_update__()
        if event.char == "v":
            if self.__showing__ == "VertNSeams":
                self.__get_func__ = self.sm.get_image
                self.__showing__ = "Image"
                self.root.resizable(True, True)
                self.window.configure(height=self.sm.get_image().shape[0])
                self.root.geometry('{}x{}'.format(self.sm.get_image().shape[1], self.sm.get_image().shape[0]))
                self.__force_update__()
            else:
                if self.sm.__image__.shape[1] < self.N:
                    self.N = self.sm.__image__.shape[1] - 1
                self.__get_func__ = lambda: self.sm.get_minimum_vertical_seams(self.N)
                self.__showing__ = "VertNSeams"
                self.root.resizable(False, False)
                self.window.configure(height=self.sm.get_image().shape[0])
                self.root.geometry('{}x{}'.format(self.sm.get_image().shape[1], self.sm.get_image().shape[0]))
                self.__force_update__()
        if event.char == "h":
            if self.__showing__ == "HorNSeams":
                self.__get_func__ = self.sm.get_image
                self.__showing__ = "Image"
                self.root.resizable(True, True)
                self.window.configure(height=self.sm.get_image().shape[0])
                self.root.geometry('{}x{}'.format(self.sm.get_image().shape[1], self.sm.get_image().shape[0]))
                self.__force_update__()
            else:
                if self.sm.__image__.shape[0] < self.N:
                    self.N = self.sm.__image__.shape[0] - 1
                self.__get_func__ = lambda: self.sm.get_minimum_horizontal_seams(self.N)
                self.__showing__ = "HorNSeams"
                self.root.resizable(False, False)
                self.window.configure(height=self.sm.get_image().shape[0])
                self.root.geometry('{}x{}'.format(self.sm.get_image().shape[1], self.sm.get_image().shape[0]))
                self.__force_update__()
        if event.char == "p":
            to_plot = self.__get_func__()
            plt.imshow(to_plot)
            plt.show()

        if event.char == "o":
            to_plot = plt.imread(self.__image_path__)
            plt.imshow(to_plot)
            plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path for image to mess with seams!")
parser.add_argument("energy", type=str, help="Energy Method for seam formation! Values: Hog, Grad, Gauss, HoGauss, Color, ColorGauss, My or Heuristic")
parser.add_argument("energy_plot_N", type=int, help="Energy Sample number for plotting!")
parser.add_argument("spread", type=float, help="Spread coefficient")
parser.add_argument("spread_decay", type=float, help="Spread coefficient")
parser.add_argument("k_size", type=int, help="Spread coefficient")
args = parser.parse_args()
path = args.path
energy = args.energy
spread = args.spread
spread_decay = args.spread_decay
N = args.energy_plot_N
k = args.k_size
ui = window_UI(path, energy=energy, N=N, spread=spread, spread_decay=spread_decay, k_size=k)
