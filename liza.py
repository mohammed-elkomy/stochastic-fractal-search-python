import time
from functools import partial

import numpy as np
import cv2

from parallel_sfs import StochasticFractalSearch
from PIL import Image, ImageDraw
import cv2
import glfw
import numpy as np
from OpenGL.GL import *

COLOR_MAX = 255

IMG_DIM = 500

RGBA = 4  # color space used
DIMENSIONS = 2  # working in 2d images
NUM_OF_POLY = 10  # number of polygons, for example if POINTS_PER_POLYGON = 3, we will have 30 triangles
POINTS_PER_POLYGON = 3  # a triangle

COLORS_TENSOR_SHAPE = (NUM_OF_POLY, RGBA)
POLYGONS_TENSOR_SHAPE = (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS)

# load the image
LIZA = cv2.imread("simple.jpg")
LIZA_HSV = cv2.cvtColor(LIZA, cv2.COLOR_RGB2HSV)
SHAPE = LIZA.shape
HEIGHT, WIDTH = SHAPE[0], SHAPE[1]


def draw_image_from_polygons(polygons, colors, show=False):
    """
    :param show: to show the images
    :param polygons: numpy (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS,), the polygons to draw
    :param colors: the color of every polygon (overlaid using Alpha channel)
    :return: 2d BGR image drawn
    """

    # image = np.zeros(SHAPE, dtype=np.uint8)
    # for polygon, color in zip(polygons, colors):
    #     overlay = image.copy()
    #     color = color.tolist()
    #     RGB, alpha = color[:-1], color[-1] / COLOR_MAX
    #
    #     cv2.drawContours(overlay, [polygon], 0, RGB, -1)
    #     # apply the overlay
    #     cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # a = time.time()
    img = Image.new('RGB', SHAPE[1::-1])
    drw = ImageDraw.Draw(img, 'RGBA')
    for polygon, color in zip(polygons, colors):
        drw.polygon(polygon.flatten().tolist(), tuple(color))

    img = np.array(img)
    if show:
        cv2.imshow("LIZA", LIZA)
        # cv2.imshow("Output", image)
        cv2.imshow("img", img)
        cv2.waitKey(1)
    # b = time.time()
    return img  # , b - a


# #########
# # Initialize the library
# glfw.init()
# # Set window hint NOT visible
# glfw.window_hint(glfw.VISIBLE, False)
# # Create a windowed mode window and its OpenGL context
# window = glfw.create_window(WIDTH, HEIGHT, "hidden window", None, None)
# if not window:
#     glfw.terminate()
#
# # Make the window's context current
# glfw.make_context_current(window)
#
# glMatrixMode(GL_PROJECTION)
# glLoadIdentity()
# glOrtho(0.0, WIDTH, 0.0, HEIGHT, 0.0, 1.0)
# glMatrixMode(GL_MODELVIEW)
# glEnable(GL_BLEND)
# glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#
#
# def draw_image_from_polygons2(polygons, colors, show=False):
#     """
#     :param show: to show the images
#     :param polygons: numpy (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS,), the polygons to draw
#     :param colors: the color of every polygon (overlaid using Alpha channel)
#     :return: 2d BGR image drawn
#     """
#     #a = time.time()
#     glClear(GL_COLOR_BUFFER_BIT)
#     glBegin(GL_TRIANGLES)
#     for polygon, color in zip(polygons, colors):
#         glColor4f(*(color / 255))
#
#         for point in polygon:
#             glVertex2f(*point)
#     glEnd()
#
#     screen = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT)
#     screen = np.copy(np.frombuffer(screen, np.float32)).reshape((HEIGHT, WIDTH) + (3,))  # Read buffer and flip Y
#     if show:
#         cv2.imshow("LIZA2", LIZA)
#         # cv2.imshow("Output", image)
#         cv2.imshow("img2", screen)
#         cv2.waitKey(1)
#     #b = time.time()
#     return screen#, b - a
#

def generation_callback(best_point, best_fit, generation):
    polygons, colors = decode(best_point)
    drawer = partial(draw_image_from_polygons, show=True)
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
    drawn = cv2.cvtColor(drawn, cv2.COLOR_RGB2HSV)
    diff = LIZA_HSV - drawn
    # black_points = np.sum(np.all(drawn[:, :] == (0, 0, 0), axis=-1))
    error = np.sum(np.multiply(diff, diff))  # + black_points
    return error


if __name__ == "__main__":
    # testing drawer
    # NUM_OF_POLY = 100
    # t1, t2 = [], []
    # for i in range(1000):
    #     pol = np.random.randint(0, IMG_DIM + 1, size=(NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS), )
    #     clrs = np.random.randint(0, 255 + 1, size=(NUM_OF_POLY, RGBA))
    #     _, b = draw_image_from_polygons(pol, clrs, show=True)
    #     _, b2 = draw_image_from_polygons2(pol, clrs, show=True)
    #     t1.append(b)
    #     t2.append(b2)
    # print(sum(t1) / sum(t2))
    # exit()

    lower_polygons = np.zeros(shape=POLYGONS_TENSOR_SHAPE, )
    lower_colors = np.zeros(shape=COLORS_TENSOR_SHAPE)
    lower_colors[:, -1] = 38
    lower = encode(lower_polygons, lower_colors)

    upper_polygons = np.zeros(shape=POLYGONS_TENSOR_SHAPE, )
    upper_polygons[..., 0] = SHAPE[1]  # x coordinate, the upper bound for the search algorithm(to be within the canvas)
    upper_polygons[..., 1] = SHAPE[0]  # y coordinate, the upper bound for the search algorithm(to be within the canvas)
    upper_colors = np.ones(shape=COLORS_TENSOR_SHAPE) * COLOR_MAX
    upper_colors[:, -1] = 38
    upper = encode(upper_polygons, upper_colors)

    sfs = StochasticFractalSearch(lower, upper, 5, 50, 1, custom_fitness, 500, iter_callback=generation_callback)
    sfs.optimize()
