import time
from functools import partial

import numpy as np
import cv2

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

    import numpy as np
    from geneticalgorithm import geneticalgorithm as ga


    def f(X):
        return np.sum(X)


    varbound = np.array([[0, 10]] * 3)

    # model = ga(function=custom_fitness, dimension=upper.shape[0], variable_type='real', variable_boundaries=np.stack([lower, upper]).T,
    #            algorithm_parameters={'max_num_iteration': 20000,
    #                                  'population_size': 100,
    #                                  'mutation_probability': 0.1,
    #                                  'elit_ratio': 0.01,
    #                                  'crossover_probability': 0.5,
    #                                  'parents_portion': 0.3,
    #                                  'crossover_type': 'uniform',
    #                                  'max_iteration_without_improv': None}, )
    #
    # model.run()

    best = [5.29965046e+02, 5.86301880e+02, 1.69390561e+02, 6.31951693e+02
        , 4.54205034e+01, 6.65553866e+02, 6.20700103e+02, 8.14359652e+02
        , 1.76928143e+02, 5.40065234e+02, 1.04620327e+02, 4.96596070e+02
        , 5.89539483e+02, 8.84443352e+02, 1.66033423e+02, 6.31146937e+02
        , 4.91978038e+02, 8.23598330e+02, 3.40831536e+02, 4.59713028e+02
        , 2.22887136e+02, 5.17588522e+02, 3.97351367e+02, 4.47686943e+02
        , 3.62093418e+02, 6.41308354e+02, 4.06261693e+02, 5.44958795e+02
        , 3.48016703e+02, 8.05629837e+02, 4.85108200e+02, 5.94686138e+02
        , 3.88088478e+02, 5.97487556e+02, 2.62048276e+02, 6.21610994e+02
        , 3.88518182e+02, 3.20497412e+02, 2.88630211e+02, 2.16847388e+02
        , 6.86615023e+02, 8.19034057e+02, 1.63394502e+02, 1.37907223e+02
        , 6.33001373e+02, 9.87579178e+02, 6.65782673e+02, 6.53351878e+00
        , 4.67004255e+02, 9.55872036e+02, 4.91364964e+02, 9.67688837e+02
        , 5.00114709e+02, 8.36339888e+02, 4.25104220e+00, 1.02293774e+03
        , 6.43403606e+02, 9.99756340e+02, 2.78376620e+01, 1.23642273e+01
        , 1.97194275e+02, 1.06491951e+02, 2.26034572e+02, 3.80000000e+01
        , 1.27569852e+02, 1.35039438e+02, 2.49084136e+02, 3.80000000e+01
        , 8.58026213e+01, 4.96871561e+01, 1.60591678e+02, 3.80000000e+01
        , 1.15347148e+02, 2.10447902e+02, 1.23781750e+02, 3.80000000e+01
        , 2.06225973e+02, 1.21554925e+01, 8.49148525e+01, 3.80000000e+01
        , 2.22374599e+02, 7.68347801e+01, 1.77042622e+02, 3.80000000e+01
        , 1.57525048e+02, 1.85232662e+02, 2.52013894e+02, 3.80000000e+01
        , 8.63468019e-01, 3.00344190e+00, 9.18629509e+01, 3.80000000e+01
        , 1.21457484e+02, 7.67276531e-01, 3.79575981e+00, 3.80000000e+01
        , 2.03195376e+02, 9.80404575e+01, 1.44525766e+02, 3.80000000e+01]

    generation_callback(np.array(best), 45343856, 0)
    cv2.waitKey()
    # sfs = StochasticFractalSearch(lower, upper, 5, 50, 1, custom_fitness, 500, iter_callback=generation_callback)
    # sfs.optimize()
