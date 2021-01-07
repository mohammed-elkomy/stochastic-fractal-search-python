import time
from functools import partial

import imutils
import numpy as np
import cv2

from my_sfs import StochasticFractalSearch
from PIL import Image, ImageDraw
import cv2
import glfw
import numpy as np
from OpenGL.GL import *

TARGET_INTERNAL_SHAPE = 75

COLOR_MAX = 255

RGBA = 4  # color space used
DIMENSIONS = 2  # working in 2d images
NUM_OF_POLY = 1000  # number of polygons, for example if POINTS_PER_POLYGON = 3, we will have 30 triangles
POINTS_PER_POLYGON = 3  # a triangle

COLORS_TENSOR_SHAPE = (NUM_OF_POLY, RGBA)
POLYGONS_TENSOR_SHAPE = (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS)

# load the image
REFERENCE_IMG = cv2.imread("imgs/liza.jpg")
REFERENCE_IMG = REFERENCE_IMG[80:80+300, 180:+180+300]
INTERNAL_IMG = imutils.resize(REFERENCE_IMG, TARGET_INTERNAL_SHAPE, TARGET_INTERNAL_SHAPE).astype(np.float)
REF_SHAPE = REFERENCE_IMG.shape
REF_HEIGHT, REF_WIDTH = REF_SHAPE[0], REF_SHAPE[1]

INTER_SHAPE = INTERNAL_IMG.shape
INTER_HEIGHT, INTER_WIDTH = INTER_SHAPE[0], INTER_SHAPE[1]


def draw_image_from_polygons(polygons, colors, show=False, shape=INTER_SHAPE):
    """
    :param shape: image shape to draw
    :param show: to show the images
    :param polygons: numpy (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS,), the polygons to draw
    :param colors: the color of every polygon (overlaid using Alpha channel)
    :return: 2d BGR image drawn
    """

    image = np.zeros(shape, dtype=np.uint8)
    if shape != INTER_SHAPE:
        polygons[..., 0] = polygons[..., 0] / INTER_SHAPE[0] * shape[0]
        polygons[..., 1] = polygons[..., 1] / INTER_SHAPE[1] * shape[1]
    for polygon, color in zip(polygons, colors):
        overlay = image.copy()
        color = color.tolist()
        RGB, alpha = color[:-1], color[-1] / COLOR_MAX

        cv2.drawContours(overlay, [polygon], 0, RGB, -1)
        # apply the overlay
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    img = np.array(image)
    if show:
        cv2.imshow("LIZA", REFERENCE_IMG)
        # cv2.imshow("Output", image)
        cv2.imshow("img", img)
        cv2.waitKey(1)

    return img  # , b - a


def generation_callback(best_point, generation):
    polygons, colors = decode(best_point.state)
    drawer = partial(draw_image_from_polygons, show=True, shape=REF_SHAPE)
    drawer(polygons, colors)


def encode(polygons, colors):
    """
    to encode polygons and colors into SFS state vector (squashing into a big vector)
    :param polygons: numpy (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS,), the polygons to draw
    :param colors: the color of every polygon (overlaid using Alpha channel)
    :return: state vector for SFS
    """
    squashed_polygons = polygons.reshape(-1)
    squashed_colors = colors.reshape(-1)
    return np.concatenate([squashed_polygons, squashed_colors])


def decode(state_vector):
    polygons_length = np.prod(POLYGONS_TENSOR_SHAPE)
    colors_length = np.prod(COLORS_TENSOR_SHAPE)
    polygons = state_vector[:polygons_length]
    polygons = polygons.reshape(POLYGONS_TENSOR_SHAPE).astype(np.int)
    colors = state_vector[-colors_length:]
    colors = colors.reshape(COLORS_TENSOR_SHAPE).astype(np.int)
    return polygons, colors


def custom_fitness(state_vector):
    polygons, colors = decode(state_vector)
    drawn = draw_image_from_polygons(polygons, colors)
    diff = INTERNAL_IMG - drawn.astype(np.float32)
    return np.mean(np.multiply(diff, diff)) / 255 / 255


if __name__ == "__main__":
    lower_polygons = np.zeros(shape=POLYGONS_TENSOR_SHAPE, )
    lower_colors = np.zeros(shape=COLORS_TENSOR_SHAPE)
    lower_colors[:, -1] = 20
    lower = encode(lower_polygons, lower_colors)

    upper_polygons = np.ones(shape=POLYGONS_TENSOR_SHAPE, )
    upper_polygons[..., 0] = INTER_WIDTH  # x coordinate, the upper bound for the search algorithm(to be within the canvas)
    upper_polygons[..., 1] = INTER_HEIGHT  # y coordinate, the upper bound for the search algorithm(to be within the canvas)
    upper_colors = np.ones(shape=COLORS_TENSOR_SHAPE) * COLOR_MAX
    upper = encode(upper_polygons, upper_colors)

    sfs = StochasticFractalSearch(lower, upper, 4, 50, 0, custom_fitness, 5000, iter_callback=generation_callback)
    sfs.optimize()
    cv2.waitKey()
