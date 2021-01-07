import os

import numpy as np
import random
import cv2
from scipy.stats import norm

preview = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
preview_mask = np.zeros((1000, 1000), dtype=np.uint8)
max_energy = 10000
cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 900, 900)


def inbound(pt):
    return 0 < pt[0] < 1000 and 0 < pt[1] < 1000


def try_draw_line(pt1, pt2, color, thickness=None):
    if inbound(pt1) and inbound(pt2):
        mask = np.zeros_like(preview_mask)
        cv2.line(mask, pt1, pt2, (255, 255, 255), thickness)  # simulate drawing on the mask

        intersection = np.bitwise_and(preview_mask, mask)

        if np.sum(intersection) < 4000:  # no intersection
            cv2.line(preview_mask, pt1, pt2, (255, 255, 255), thickness)  # simulate drawing on the mask
            cv2.line(preview, pt1, pt2, color, thickness)  # simulate drawing on the mask

            cv2.imshow("window", preview)
            #cv2.imwrite("imgs/{}.png".format(len(os.listdir("imgs"))), preview)
            cv2.waitKey(40)
            return True

    return False


def fractal(point, total_energy, depth=5):
    if total_energy > 10:

        # branches = np.random.randint(2, depth)
        props = np.random.rand(depth)
        energies = total_energy * props / np.sum(props)
        sampled_points = []
        for energy in energies:
            draw_success, sampled_point = False, (0, 0)
            while not draw_success:
                normal = norm(loc=point, scale=(energy, energy))
                sampled_point = tuple([int(compon) for compon in normal.rvs()])
                draw_success = try_draw_line(point, sampled_point, random_rgb(), 2)

            cv2.circle(preview, sampled_point, int(np.log(total_energy * total_energy)), gray_scalar(total_energy), -1)
            sampled_points.append(sampled_point)
        for sampled_point, energy in zip(sampled_points, energies):
            fractal(sampled_point, energy, depth + 1)


def random_rgb():
    return tuple([(random.randint(0, 255)) for _ in range(3)])


def gray_scalar(total_energy):
    return tuple([int((1 - total_energy / max_energy) * 60) for _ in range(3)])


for img in os.listdir("imgs"):
    os.remove(os.path.join("imgs", img))
fractal((500, 500), max_energy)
