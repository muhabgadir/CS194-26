import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d
from skimage.color import rgb2gray

def energy(image):
    D_x = [[1, -1], [0, 0]]
    D_y = [[1, 0], [-1, 0]]

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    R_X = np.abs(convolve2d(R, D_x, mode="same"))
    G_X = np.abs(convolve2d(G, D_x, mode="same"))
    B_X = np.abs(convolve2d(B, D_x, mode="same"))
    X_energy = np.dstack([R_X, G_X, B_X])

    R_Y = np.abs(convolve2d(R, D_y, mode="same"))
    G_Y = np.abs(convolve2d(G, D_y, mode="same"))
    B_Y = np.abs(convolve2d(B, D_y, mode="same"))
    Y_energy = np.dstack([R_Y, G_Y, B_Y])

    energy = (X_energy/255 + Y_energy/255)
    plt.imshow(energy)
    return rgb2gray(energy)

def get_vert_seams(energy):
    seams = energy.copy()
    for i in range(1, energy.shape[0]):
        for j in range(energy.shape[1]):
            to_add_ls = [seams[i-1, j]]
            if j != 0:
                to_add_ls.append([seams[i-1, j-1]])
            if j != energy.shape[1]-1:
                to_add_ls.append([seams[i-1, j-1]])

            seams[i, j] += min(to_add_ls)
    return (seams)#/np.max(seams)

def get_hor_seams(energy):
    seams = energy.copy()
    for j in range(1, seams.shape[1]):
        # plt.plot(seams[:, j])
        # plt.plot(seams[:, j-1])
        # plt.show()
        for i in range(seams.shape[0]):
            to_add_ls = [seams[i, j-1]]
            if i != 0:
                to_add_ls.append([seams[i-1, j-1]])
            if i != seams.shape[0]-1:
                to_add_ls.append([seams[i+1, j-1]])

            # if i == 400:
            #     print("400")
            #     print(to_add_ls)
            #     print(seams[i, j])
            #     print(seams[i, j] + min(to_add_ls))
            # if i == 200:
            #     print("200")
            #     print(to_add_ls)
            #     print(seams[i, j])
            #     print(seams[i, j] + min(to_add_ls))

            seams[i, j] += min(to_add_ls)


    return seams

def paint_hor_paths(seams, img):

    test_img = img.copy()
    for i in range(3):
        min_row = min(seams[:, -1])
        min_idx = np.where(seams[:, -1] == min_row)
        row = min_idx[0][0]
        seams[row, -1] = 100000
        for j in range(seams.shape[1]):
            col = seams.shape[1] - (j+1)
            test_img[row, col] = 0
            seams[row, col] = 10000

            new_row_ls = [([seams[row, col-1], row])]

            if row != 0:
                new_row_ls.append(([seams[row-1, col-1], row-1]))
            if row != seams.shape[0]-1:
                new_row_ls.append(([seams[row+1, col-1], row+1]))

            row = min(new_row_ls, key=lambda x: x[0])[1]


    return test_img


im = np.array([[1, 5, 9],
               [4, 2, 6],
               [7, 8, 5]])

seams = get_hor_seams(im)
print(seams)
print(paint_hor_paths(seams, im))






# im = plt.imread("from_paper.png")
# print(im.shape)

# im2 = np.pad(im, pad_width=((10, 10), (10, 10), (0, 0)), mode="symmetric")

# en = energy(im2)
# en = en[10:-10, 10:-10]
# print(en.shape)
# plt.imshow(en)
# plt.show()
# seam = get_vert_seams(en)
# # plt.imshow(seam)
# # plt.show()
# seam = get_hor_seams(en[:, 1:])
# plt.imshow(seam)
# plt.show()
# paint_hor_paths(seam, im[:, 1:])
