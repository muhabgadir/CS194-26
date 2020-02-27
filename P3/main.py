import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from skimage import img_as_uint
from skimage import img_as_float
import skimage.io as io
import skimage.draw as draw
import numpy as np
import numpy.linalg as lin
from scipy.spatial import Delaunay
import pickle
import re
import argparse
import os.path as path
from os import system
import time

N = 40

def define_points(image_name):
    image = io.imread(image_name)
    plt.imshow(image)
    points = plt.ginput(N, 0)
    plt.close()
    pickle_name = re.split("\.", image_name)[0] + ".p"
    pickle.dump(points, open(pickle_name, "wb"))

def load_points(image_name):
    pickle_name = re.split("\.", image_name)[0] + ".p"
    return np.array(pickle.load(open(pickle_name, "rb")))

def points_exist(image_name):
    return path.exists(re.split("\.", image_name)[0]+".p")

def mount_and_show_points(imageA_name, imageB_name):
    pointsA = load_points(imageA_name)
    pointsB = load_points(imageB_name)
    pointImageA_name = re.split("\.", imageA_name)[0]+"_points.jpg"
    pointImageB_name = re.split("\.", imageB_name)[0]+"_points.jpg"

    imgA = io.imread(imageA_name)
    plt.imshow(imgA)
    for point in pointsA:
        plt.scatter(point[0], point[1], color="red")
    plt.savefig(pointImageA_name)
    plt.close()

    imgB = io.imread(args.imgB)
    plt.imshow(imgB)
    for point in pointsB:
        plt.scatter(point[0], point[1], color="red")
    plt.savefig(pointImageB_name)
    plt.close()

    pointImageA = io.imread(pointImageA_name)
    pointImageB = io.imread(pointImageB_name)
    io.imshow(np.concatenate([pointImageA, pointImageB]))
    io.show()

def avg_points(pointsA, pointsB, alpha):
    final_points = []
    for i in range(0, len(pointsA)):
        final_points.append((alpha * pointsA[i] + (1 - alpha) * pointsB[i]))
    return np.array(final_points)

def affine(triangle, target):
    A = np.matrix([triangle[:,0], triangle[:, 1], [1, 1, 1]])
    B = np.matrix([target[:,0], target[:, 1], [1, 1, 1]])
    return B * lin.inv(A)

def apply_affine(A, point):
    point = np.array([[point[0], point[1], 1]])
    final = A * np.transpose(point)
    return np.transpose(np.array(final[:2]))

def triangle_bool_matrix(triangle, image_shape):
    tri_buf = triangle
    shape = (image_shape[1], image_shape[0], image_shape[2])
    points = draw.polygon(tri_buf[:,0], tri_buf[:,1], shape=shape)
    return np.vstack([points, np.ones(len(points[0]))])

def apply_masked_affine(mask, image, src_tri, target_tri):
    A = affine(src_tri, target_tri)

    # Invert the mask so it's
    # [x]            [y]
    # [y] instead of [x]
    # [1]            [1]
    final_mask = mask.copy()
    final_mask[0] = mask[0]
    final_mask[1] = mask[1]

    affined = (lin.inv(A) * final_mask).astype(np.int)
    cc, rr, stub = affined

    final_mask = final_mask.astype(np.int)
    canvas = np.zeros_like(image)
    canvas[final_mask[1], final_mask[0]] = image[rr, cc]
    return canvas

def paint_triangles(tri_list, imageA, imageB, pointsA, pointsB, avg_points, alpha):
    assert(imageA.shape == imageB.shape)
    shape = imageA.shape
    final = np.zeros_like(imageA).astype(np.float)
    for tri in tri_list:
        avg_tri = avg_points[tri]
        tri_mask = triangle_bool_matrix(avg_tri, shape)
        triA = pointsA[tri]
        colorA = apply_masked_affine(tri_mask, imageA, triA, avg_tri)
        tri_mask = triangle_bool_matrix(avg_tri, shape)
        triB = pointsB[tri]
        colorB = apply_masked_affine(tri_mask, imageB, triB, avg_tri)
        toAdd = alpha * colorA/255 + float(1 - alpha) * colorB/255
        final += toAdd
    # This shows the triangles
    # plt.triplot(avg_points[:,0], avg_points[:,1], tri_list)
    # io.imshow(final)
    # io.show()
    return final

def update(i):
    return 



intro = "Project 3 for CS 194-26: Face Morphing\n"
parser = argparse.ArgumentParser(intro)

parser.add_argument('imgA',
                    metavar='Image A',
                    type=str,
                    help="The first image.")

parser.add_argument('imgB',
                    metavar='Image B',
                    type=str,
                    help="The second image.")

parser.add_argument('--reset',
                    dest="reset",
                    action='store_const',
                    const=True, default=False,
                    help='Asks for point settings again')

args = parser.parse_args()

if args.reset:
    input("If you don't want to remove the current config, Interrupt process (C-c)")
    system("rm *.p")

if not points_exist(args.imgA):
    define_points(args.imgA)
if not points_exist(args.imgB):
    define_points(args.imgB)


pointsA = load_points(args.imgA)
pointsB = load_points(args.imgB)
delu = Delaunay(pointsA)
triangles = delu.simplices
imgA = io.imread(args.imgA)
imgB = io.imread(args.imgB)
mov = []
fig, ax = plt.subplots()
strt = time.time()
for i in range(0, 11):
    paint = paint_triangles(triangles, imgA, imgB, pointsA, pointsB,
                    avg_points(pointsA, pointsB, float((10 - i)/10)), float((10 - i)/10))
    im = plt.imshow(paint, animated=True)
    mov.append([im])
fin = time.time() - strt
print(fin)
mov2 = copy.copy(mov)
mov2.reverse()
mov += mov2

ani = animation.ArtistAnimation(fig, mov, interval=1000, blit=True,
                                repeat_delay=0)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# plt.show()
ani.save("animation.mp4", writer=writer)
