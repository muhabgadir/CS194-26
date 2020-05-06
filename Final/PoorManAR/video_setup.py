import skvideo.io as vid_io
import skimage.io as img_io
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
import re
from os import path
from harris import *
from numpy.linalg import lstsq

def ginput_abstraction(image, N):
    plt.imshow(image)
    pts = plt.ginput(N, timeout=0)
    plt.close()
    return pts

def draw(img, imgpts):
    imgpts = imgpts[:, :2].astype(np.int)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0), -3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


class video_setup:
    def __init__(self, video_path, N, bbox_size=8):
        self.__pickle_name__ = re.split("\.", video_path)[0]
        self.__video__ = vid_io.vread(video_path)
        self.__first_frame_edges__ = self.__get_first_points__(N)
        self.__3d_coords__ = self.__get_3D_coords__()
        self.__tracker_list__ = self.__setup_tracker__(bbox_size)

    def __get_points__(self, extension, comp_func):
        if not (self.__points_exist__(extension)):
            points = comp_func()
            self.__save_points__(points, extension)

        points = self.__load_points__(extension)
        return points

    def __save_points__(self, points, extension):
        pickle.dump(points, open(self.__pickle_name__ + extension, "wb"))

    def __compute_points__(self, N):
        first_frame = self.__video__[0]
        plt.imshow(first_frame)
        points = plt.ginput(N, timeout=0)
        plt.close()
        return points

    def __load_points__(self, extension):
        points = pickle.load(open(self.__pickle_name__ + extension, "rb"))
        return np.array(points)

    def __points_exist__(self, extension):
        return path.exists(self.__pickle_name__ + extension)

    def __get_first_points__(self, N):
        get_first_points = lambda: ginput_abstraction(self.__video__[0], N)
        return self.__get_points__(".edges", get_first_points)

    def __request_user_based_3D_coords__(self):
        point3d_ls = []
        for point in self.__first_frame_edges__:
            plt.imshow(self.__video__[0])
            plt.scatter(point[0], point[1], marker="x", c="red")
            plt.show(block=False)
            x = int(input("X: "))
            y = int(input("Y: "))
            z = int(input("Z: "))
            point3d_ls.append(np.array([x, y, z]))
            plt.close()

        return point3d_ls

    def __get_3D_coords__(self):
        get_coords = lambda: self.__request_user_based_3D_coords__()
        return self.__get_points__(".3Dcoords", get_coords)

    def __init_bboxs__(self, bbox_size):
        bbox_list = []
        for point in self.__first_frame_edges__:
            bbox_list.append(np.array([[int(point[0]-(bbox_size/2)), int(point[1]-(bbox_size/2))],
                                       [bbox_size, bbox_size]]))
        return bbox_list

    def __setup_tracker__(self, bbox_size):
        bboxs = self.__init_bboxs__(bbox_size)
        first_frame = self.__video__[0]
        tracker_list = []
        for bbox in bboxs:
            new_tracker = cv.TrackerCSRT_create()
            bbox = tuple(bbox.flatten())
            ok = new_tracker.init(first_frame, bbox)
            if not ok:
                print("There was an error while setting up one of the trackers!")
                return None

            tracker_list.append(new_tracker)
        return tracker_list

    def __compute_corresp__(self, show_vid):
        computed_points = []
        for i in range(self.__video__.shape[0]):
            frame = self.__video__[i]
            errorcnt = 0
            computed_points.append([])
            for tracker in self.__tracker_list__:
                ok, bbox = tracker.update(frame)
                if ok:
                    computed_points[i].append(np.array([int(bbox[0] + bbox[2]/2),
                                                        int(bbox[1] + bbox[3]/2)]))
                    if show_vid:
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv.circle(frame, (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)), 3, (0, 255, 0), -1)
                        cv.rectangle(frame, p1, p2, (0, 0, 255))
                else:
                    errorcnt += 1

            print("ERRORS: ", errorcnt)
            if show_vid:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                cv.imshow("Tracking", frame)
                cv.imwrite("tracking_"+str(i)+".jpg", frame)
                k = cv.waitKey(1) & 0xff
                if k == 27:
                    return
        return computed_points

    def __compute_frames_prespective_mtx__(self, frame_id, points):
        frame_points = points[frame_id]
        frame_points = np.hstack([frame_points, np.ones((frame_points.shape[0], 1))])
        hom_3d = np.hstack([self.__3d_coords__, np.ones((self.__3d_coords__.shape[0], 1))])
        M = (np.array(lstsq(hom_3d, frame_points)))[0]

        return M.T

    def __project_cube__(self, image, points, project_M):
        n_points = np.dot(project_M, points)
        n_points /= n_points[2]
        n_points = n_points.T
        return draw(image, n_points)

    def run(self, axis, show_vid=True):
        compute_corresp_func = lambda: self.__compute_corresp__(show_vid)
        computed_points = self.__get_points__(".corresp", compute_corresp_func)
        for frame_id in range(self.__video__.shape[0]):
            project_M = self.__compute_frames_prespective_mtx__(frame_id, computed_points)
            frame = self.__project_cube__(self.__video__[frame_id], axis, project_M)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            if show_vid:
                cv.imshow("geom", frame)
            cv.imwrite("new_geom"+str(frame_id)+".jpg", frame)
            k = cv.waitKey(0) & 0xFF


    # Deprecated for now
    # def __detect_first_corners__(self, corner_threshold):
    #     first_frame = self.__video__[0]
    #     gray_first_frame = first_frame[:, :, 0]
    #     corner_map, corners = get_harris_corners(gray_first_frame, edge_discard=40)
    #     filtered_idx = np.where(corner_map[corners[0], corners[1]] > corner_threshold)
    #     corner_map = corner_map[corners[0, filtered_idx], corners[1, filtered_idx]]

    #     # plt.imshow(first_frame)
    #     # plt.scatter(corners[1, filtered_idx], corners[0, filtered_idx], marker="x", c="red")
    #     # plt.show()
    #     corners = np.vstack([corners[0, filtered_idx],
    #                          corners[1, filtered_idx]])
    #     aux = corners[1].copy()
    #     corners[1] = corners[0]
    #     corners[0] = aux
    #     print(corners.T)
    #     return corners.T

test = video_setup("box.mp4", 24, bbox_size=10)
axis = np.float32([[1,2,0,1], [1,2,1,1], [2,2,1,1], [2,2,0,1],
                   [1,3,0,1],[1,3,1,1],[2,3,1,1],[2,3,0,1]]).T
test.run(axis, show_vid=True)
