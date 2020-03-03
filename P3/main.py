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
import os
import time
import asf_reader

# Number of points requested
N = 58

## Parameter Svaing/Loading
def define_points(image_name):
    image = io.imread(image_name)
    plt.imshow(image)
    points = plt.ginput(N, 0)
    points.append((0, 0))
    points.append((0, image.shape[1]))
    points.append((image.shape[0], 0))
    points.append((image.shape[0], image.shape[1]))
    plt.close()
    pickle_name = re.split("\.", image_name)[0] + ".p"
    pickle.dump(points, open(pickle_name, "wb"))

def load_points(image_name):
    img = io.imread(image_name)
    shape = img.shape
    pickle_name = re.split("\.", image_name)[0] + ".p"
    points = pickle.load(open(pickle_name, "rb"))
    points.append((0, 0))
    points.append((0, shape[0]))
    points.append((shape[1], 0))
    points.append((shape[1], shape[0]))
    return np.array(points)


def points_exist(image_name):
    return path.exists(re.split("\.", image_name)[0]+".p")


## Display Points
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


## Get interpolated points
def avg_pair_points(pointsA, pointsB, alpha):
    final_points = []
    for i in range(0, len(pointsA)):
        final_points.append((alpha * pointsA[i] + (1 - alpha) * pointsB[i]))
    return np.array(final_points)


## Get average population points
def avg_population_points(point_sets):
    avg = point_sets[0]
    for i in range(1, len(point_sets)):
        avg = np.add(avg, point_sets[i])

    return avg/len(point_sets)


## Compute Affine matrix
def affine(triangle, target):
    A = np.matrix([triangle[:,0], triangle[:, 1], [1, 1, 1]])
    B = np.matrix([target[:,0], target[:, 1], [1, 1, 1]])
    return B * lin.inv(A)


## Testing point Affine application function
def apply_affine(A, point):
    point = np.array([[point[0], point[1], 1]])
    final = A * np.transpose(point)
    return np.transpose(np.array(final[:2]))


## Get triangle coordinates in image
def triangle_bool_matrix(triangle, image_shape):
    tri_buf = triangle
    shape = (image_shape[1], image_shape[0], image_shape[2])
    points = draw.polygon(tri_buf[:,0], tri_buf[:,1], shape=shape)
    return np.vstack([points, np.ones(len(points[0]))])


## Apply inverse transformation based on mask
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


## Blend two images
def create_pair_blend(tri_list, images, points, alpha):
    avg = avg_pair_points(points[0], points[1], alpha)
    assert(images[0].shape == images[1].shape)
    final = np.zeros_like(images[0]).astype(np.float)
    for tri in tri_list:
        target_tri = avg[tri]
        target_mask = triangle_bool_matrix(target_tri, images[0].shape)
        src_tri = points[0][tri]
        final_tri = alpha * apply_masked_affine(target_mask, images[0],
                                                src_tri, target_tri) / 255
        src_tri = points[1][tri]
        final_tri += (1-alpha) * apply_masked_affine(target_mask, images[1],
                                                     src_tri, target_tri) / 255
        final += final_tri
    final[final>1] = 1
    return final, avg


## Save middle face blennded image
def compute_middle_face(imageA_name, imageB_name, dest_img, alpha):
    pointsA = load_points(imageA_name)
    pointsB = load_points(imageB_name)
    imgA = io.imread(imageA_name)
    imgB = io.imread(imageB_name)
    delu = Delaunay(pointsA)
    triangles = delu.simplices
    final, _ = create_pair_blend(triangles, [imgA, imgB],
                              [pointsA, pointsB], alpha)
    io.imsave(dest_img, final)


## Save movie based on two images
def compute_morph_video(imageA_name, imageB_name, dest_video, depth):
    pointsA = load_points(imageA_name)
    pointsB = load_points(imageB_name)
    imgA = io.imread(imageA_name)
    imgB = io.imread(imageB_name)
    delu = Delaunay(pointsA)
    triangles = delu.simplices
    mov = []
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i in range(0, depth+1):
        alpha = float((depth - i)/depth)
        strt = time.time()
        paint, _ = create_pair_blend(triangles, [imgA, imgB], [pointsA, pointsB], alpha)
        print("Frame morph time:", time.time() - strt)
        im = plt.imshow(paint)
        mov.append([im])
    mov2 = copy.copy(mov)
    mov2.reverse()
    mov += mov2
    ratio = imgA.shape[1] / imgA.shape[0]
    fig.set_size_inches(int(imgA.shape[1]/50), int(imgA.shape[0]/50), 10)
    ani = animation.ArtistAnimation(fig, mov, interval=1000, blit=True, repeat_delay=0)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(dest_video, writer=writer)


## Population file reading/interpretation functions
def get_points_asf(file_name):
    lines = asf_reader.read_asf(file_name)
    points = []
    for line in lines:
        data = line.split(" \t")
        points.append((float(data[2]), float(data[3])))
    points.append((0., 0.))
    points.append((1., 0.))
    points.append((0., 1.))
    points.append((1., 1.))
    return np.array(points)

def read_db(path, regex_img):
    file_ls = []
    points_ls = []
    for file in os.listdir(path):
       if re.match(regex_img, file):
           img = io.imread(os.path.join(path, file))
           file_ls.append(img)
           asf_file = re.sub("jpg", "asf", os.path.join(path, file))
           points_ls.append(get_points_asf(asf_file))

    shape = file_ls[0].shape
    for point_set in points_ls:
        for i in range(0, len(point_set)):
            point_set[i] = (point_set[i][0]*shape[1], point_set[i][1]*shape[0])

    return file_ls, points_ls


## Blend population
def create_population_blend(tri_list, images, point_sets):
    avg = avg_population_points(point_sets)
    final = np.zeros_like(images[0]).astype(np.float)
    for tri in tri_list:
        target_tri = avg[tri]
        target_mask = triangle_bool_matrix(target_tri, images[0].shape)
        for i in range(0, len(images)):
            src_tri = np.array(point_sets[i])[tri]
            final_tri = (1/len(images)) * apply_masked_affine(target_mask, images[i],
                                                              src_tri, target_tri) / 255
            final += final_tri
    final[final > 1] = 1
    return final, avg


## Save population mean face
def compute_pop_mean_face(path, regex, dest_img):
    images, points = read_db(path, regex)
    delu = Delaunay(points[0])
    triangles = delu.simplices
    final, avg_points = create_population_blend(triangles, images, points)
    io.imsave(dest_img, final)
    pickle_name = re.split("\.", dest_img)[0] + ".p"
    pickle.dump(list(avg_points), open(pickle_name, "wb"))


## Compute facial space video
def compute_space_morph(imageA_name, imageB_name, imageC_name, depth, alpha, dest_video):
    pointsA = load_points(imageA_name)
    pointsB = load_points(imageB_name)
    pointsC = load_points(imageC_name)
    imgA = io.imread(imageA_name)
    imgB = io.imread(imageB_name)
    imgC = io.imread(imageC_name)
    delu = Delaunay(pointsA)
    triangles = delu.simplices
    mov = []
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    for i in range(0, depth+1):
        space_alpha = float((depth - i)/depth)
        mid, avg = create_pair_blend(triangles, [imgB, imgC],
                                     [pointsB, pointsC], space_alpha)
        final,_ = create_pair_blend(triangles, [imgA, mid*255],
                                    [pointsA, avg], alpha)
        im = plt.imshow(final)
        mov.append([im])
    mov2 = copy.copy(mov)
    mov2.reverse()
    mov += mov2
    ratio = imgA.shape[1] / imgA.shape[0]
    fig.set_size_inches(int(imgA.shape[1]/50), int(imgA.shape[0]/50), 10)
    ani = animation.ArtistAnimation(fig, mov, interval=1000, blit=True, repeat_delay=0)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(dest_video, writer=writer)


intro = "Project 3 for CS 194-26: Face Morphing\n"
parser = argparse.ArgumentParser(intro)

parser.add_argument('imgA',
                    metavar='Image',
                    type=str,
                    help="The first image or a Population directory path")

parser.add_argument('imgB',
                    metavar='ImageB',
                    type=str,
                    help="The second image for morphing or the Regex for population images.")

parser.add_argument('-C',
                    '--imageC',
                    dest="imgC",
                    type=str,
                    help="The third image for facial space interpolation.")

parser.add_argument('method',
                    metavar='Method',
                    type=str,
                    help="Method to use (Middle, Video, Population, Space).")

parser.add_argument('out',
                    metavar='Output',
                    type=str,
                    help="Path in which to save the result.")

parser.add_argument('-d',
                    '--depth',
                    dest="depth",
                    type=int,
                    default=10,
                    help='Sets the number of interpolation layers for the movie.')

parser.add_argument('-a',
                    '--alpha',
                    dest="alpha",
                    type=float,
                    default=0.5,
                    help='Sets the number of interpolation alpha.')

parser.add_argument('--reset',
                    dest="reset",
                    action='store_const',
                    const=True, default=False,
                    help='Asks for point settings again')

args = parser.parse_args()

if not (args.method == "Middle" or args.method == "Video" or args.method == "Population" or args.method == "Space") :
    print("Invalid method!")
    exit()

if args.method == "Population":
    compute_pop_mean_face(args.imgA, args.imgB, args.out)
    exit()

if args.reset:
    input("If you don't want to remove the current config, Interrupt process (C-c)")
    system("rm *.p")

if not points_exist(args.imgA):
    define_points(args.imgA)
if not points_exist(args.imgB):
    define_points(args.imgB)

if args.method == "Middle":
    compute_middle_face(args.imgA, args.imgB, args.out, args.alpha)
    exit()

if args.method == "Video":
    compute_morph_video(args.imgA, args.imgB, args.out, args.depth)
    exit()

if not points_exist(args.imgC):
    define_points(args.imgC)

if args.method == "Space":
    compute_space_morph(args.imgA, args.imgB, args.imgC, args.depth, args.alpha, args.out)
