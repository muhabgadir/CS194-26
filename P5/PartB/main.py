from harris import *
import skimage.io as io
import matplotlib.pyplot as plt
import re
import pickle
import numpy as np
from scipy.signal import convolve2d
import cv2
from os import path
from rect import image_warpper
import random

# Creates gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, np.transpose(kernel))

def SSD(A, B):
    return np.sum(np.sum((A-B)**2))


class mops:
    def __init__(self, image1, image2, edge_discard, gray=False):
        self.edge_discard = edge_discard
        self.image1 = io.imread(image1)
        self.image2 = io.imread(image2)
        self.image1_name = image1
        self.image2_name = image2
        self.feature_map = None
        self.isgray = gray
        self.points1, self.points2 = self.__get_points__(self.image1_name,
                                                         self.image2_name)

    def __get_points__(self, image1_name, image2_name):
        if not (self.__points_exist__(image1_name) and
                self.__points_exist__(image2_name)):
            points1, points2 = self.__compute_points__()
            self.__save_points__(image1_name, points1)
            self.__save_points__(image2_name, points2)

        points1 = self.__load_points__(image1_name)
        points2 = self.__load_points__(image2_name)
        return [points1, points2]

    def __save_points__(self, image_name, points):
        pickle_name = re.split("\.", image_name)[0] + ".p2"
        pickle.dump(points, open(pickle_name, "wb"))

    def __compute_points__(self):
        to_comp1 = self.image1
        to_comp2 = self.image2
        if not self.isgray:
            to_comp1 = to_comp1[:, :, 0]
            to_comp2 = to_comp2[:, :, 0]
        h1, coords1 = get_harris_corners(to_comp1, self.edge_discard)
        h2, coords2 = get_harris_corners(to_comp2, self.edge_discard)
        coords1 = coords1.T
        coords2 = coords2.T
        points1 = self.__supress_points__(coords1, h1, 350)
        points2 = self.__supress_points__(coords2, h2, 350)
        return [points1, points2]

    def __load_points__(self, image_name):
        pickle_name = re.split("\.", image_name)[0] + ".p2"
        points = pickle.load(open(pickle_name, "rb"))
        return points

    def __points_exist__(self, image_name):
        return path.exists(re.split("\.", image_name)[0]+".p2")

    def __get_area_points__(self, point, radius):
        x, y = np.mgrid[point[0]-radius:point[0]+radius+1,
                        point[1]-radius:point[1]+radius+1]
        return np.vstack([x.flatten(), y.flatten()]).T

    def __check_maximal_neigh__(self, h_map, point, radius):
        area = h_map[point[0]-radius:point[0]+radius+1,
                     point[1]-radius:point[1]+radius+1]

        if area.size == 0:
            return [point], point

        max_val = np.max(area)
        max_point = point + np.array(np.where(area == max_val)).flatten() - radius
        to_eliminate = self.__get_area_points__(point, radius)
        return [to_eliminate, max_point]

    def __supress_points__(self, points, h_map, goal_num):
        radius = 10
        points_buf = points.copy()
        new_batch = []

        print("Goal: ", goal_num, " Start: ", len(points_buf))
        while len(points_buf) > goal_num:
            while len(points_buf) > 0:
                to_elim, max_point = self.__check_maximal_neigh__(h_map,
                                                                  points_buf[0],
                                                                  radius)
                new_batch.append(max_point)
                points_buf_idx = (points_buf[:, None] == to_elim).all(-1).any(-1)
                points_buf_idx = np.invert(points_buf_idx)
                points_buf = points_buf[points_buf_idx]
            print("New size: ", len(new_batch))
            radius += 5
            points_buf = np.array(new_batch.copy())
            new_batch = []

        return points_buf
        
    def __get_patches__(self, image, points):
        patches = []
        for point in points:
            if point[1] - 20 < 0 or point[1] + 20 > image.shape[1] or point[0] - 20 < 0 or point[0] + 20 > image.shape[0]:
                continue
            patch = image[point[0]-20:point[0]+20, point[1]-20:point[1]+20]
            patches.append((point, patch))
        return patches

    def __get_feature_patches__(self, patches):
        gauss_kernel = gaussian_kernel(33, 3)
        blurred_patches = []
        for patch in patches:
            if not self.isgray:
                blurred_patch = convolve2d(patch[1][:, :, 0], gauss_kernel, mode="valid", boundary="wrap")
            else:
                blurred_patch = convolve2d(patch[1], gauss_kernel, mode="valid", boundary="wrap")
            blurred_patch = (blurred_patch - np.mean(blurred_patch))/np.std(blurred_patch)
            blurred_patches.append((patch[0], blurred_patch))
        return blurred_patches

    def __compute_map_features__(self, feat1, feat2):
        feature_map = {}
        for i in range(len(feat1)):
            corner_NNs = []
            for j in range(len(feat2)):
                corner_NNs.append((feat2[j][0], SSD(feat1[i][1], feat2[j][1])))
            corner_NNs.sort(key = lambda x: x[1])
            thresh = corner_NNs[0][1] / corner_NNs[1][1]
            if thresh < 0.2:
                feature_map[tuple(feat1[i][0])] = corner_NNs[0][0]
        return feature_map

    def compute_features(self, plot_feat):
        patches1 = self.__get_patches__(self.image1, self.points1)
        patches2 = self.__get_patches__(self.image2, self.points2)
        feat1 = self.__get_feature_patches__(patches1)
        feat2 = self.__get_feature_patches__(patches2)
        self.feature_map = self.__compute_map_features__(feat1, feat2)
        if plot_feat:
            keys = np.array(list(self.feature_map.keys()))
            plt.imshow(self.image1)
            plt.scatter(keys[:, 1], keys[:, 0],
                        marker="x", color="red", s=200)
            plt.show()
            for key in self.feature_map:
                plt.imshow(self.image2)
                plt.scatter(self.feature_map[key][1], self.feature_map[key][0],
                            marker="x", color="red", s=200)
                plt.show()
        assert(len(self.feature_map) >= 4)

    def __RANSAC_step__(self):
        random_points = random.sample(list(self.feature_map.keys()), 4)
        assert(len(random_points) >= 4)
        im1_points = []
        im2_points = []
        for key in random_points:
            im1_points.append(key)
            im2_points.append(self.feature_map[key])

        self.__save_new_simple_points__(self.image1_name, im1_points)
        self.__save_new_simple_points__(self.image2_name, im2_points)

    def __save_new_simple_points__(self, image_name, points):
        points = np.array(points)
        new_pts = points.copy()
        new_pts[:, 0] =  points[:, 1]
        new_pts[:, 1] =  points[:, 0]
        pickle_name = re.split("\.", image_name)[0] + ".p"
        pickle.dump(new_pts, open(pickle_name, "wb"))

    def __RANSAC__(self):
        largest_set = ([], [])
        best_imwarp = None
        for i in range(1000):
            self.__RANSAC_step__()
            try:
                imwarp = image_warpper(self.image1_name, self.image2_name, gray=self.isgray)
            except:
                continue
            this_set = []
            for key in self.feature_map:
                p = np.array([[key[1]], [key[0]], [1]])
                new_p = np.array([[self.feature_map[key][1]],
                                 [self.feature_map[key][0]],
                                 [1]])
                new_p = imwarp.H * new_p
                new_p = new_p / new_p[2]

                p = p[:2].flatten().reshape(1, 2)
                new_p = np.array(new_p[:2].flatten()).reshape(1, 2).astype(np.int)

                distance = dist2(p, new_p)
                if distance < 4:
                    this_set.append([key, self.feature_map[key]])

            if len(largest_set[0]) < len(this_set):
                largest_set = (np.array(this_set), imwarp.H)
                best_imwarp = imwarp

        try:
            best_imwarp.compute_morph(2)
            best_imwarp.show_result()
        except:
            print('''There was an error computing the approximate optimal stitch.
            \n Let's try running with all the inliners''')

        self.__save_new_simple_points__(self.image1_name, largest_set[0][:, 0])
        self.__save_new_simple_points__(self.image2_name, largest_set[0][:, 1])
        return len(largest_set[0])

    def blend_images(self, save_name=None):
        size = self.__RANSAC__()
        print("Found optimal points")
        imwarp = image_warpper(self.image1_name, self.image2_name, N=size, gray=self.isgray)
        print("Ready to compute")
        imwarp.compute_morph(2)
        imwarp.show_result()
        if not save_name is None and type(save_name) == str:
            imwarp.save_result(save_name)

    def __show_feat__(self):
        for key in self.feature_map:
            plt.imshow(self.image1)
            plt.scatter(key[1], key[0],
                        marker="x", color="red", s=200)
            plt.show()
            plt.imshow(self.image2)
            plt.scatter(self.feature_map[key][1], self.feature_map[key][0],
                        marker="x", color="red", s=200)
            plt.show()

