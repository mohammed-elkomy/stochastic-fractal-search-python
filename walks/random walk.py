import numpy as np
import random
import cv2
from scipy.stats import norm

draw = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
max_energy = 500


def fractal(point, total_energy, depth=4):
    if total_energy > 5:

        # branches = np.random.randint(2, depth)
        props = np.random.rand(depth)
        energies = total_energy * props / np.sum(props)
        sampled_points = []
        for energy in energies:
            normal = norm(loc=point, scale=(energy, energy))
            sampled_point = tuple([int(compon) for compon in normal.rvs()])
            cv2.line(draw, point, sampled_point, random_rgb(), 2)

            cv2.circle(draw, sampled_point, int(np.log(total_energy * total_energy)), gray_scalar(total_energy), -1)
            cv2.imshow("draw", draw)
            cv2.waitKey(40)

            sampled_points.append(sampled_point)
        for sampled_point, energy in zip(sampled_points, energies):
            fractal(sampled_point, energy, max(depth - 1, 2))


def random_rgb():
    return tuple([(random.randint(0, 255)) for _ in range(3)])


def gray_scalar(total_energy):
    return tuple([int((1 - total_energy / max_energy) * 180) for _ in range(3)])


fractal((500, 500), max_energy)
cv2.waitKey()
