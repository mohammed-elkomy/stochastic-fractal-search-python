import cv2
import glfw
import numpy as np
from OpenGL.GL import *


def square(i):
    glBegin(GL_TRIANGLES)
    glVertex2f(100 + i, 100 + i)
    glVertex2f(200 + i, 100 + i)
    glVertex2f(200 + i, 200 + i)
    glVertex2f(+ i,  + i)
    glVertex2f(0 + i, 100 + i)
    glVertex2f(100 + i, 100 + i)
    glEnd()


w, h = 500, 600

# Initialize the library
glfw.init()
# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(w, h, "hidden window", None, None)
if not window:
    glfw.terminate()

# Make the window's context current
glfw.make_context_current(window)

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0.0, w, 0.0, h, 0.0, 1.0)
glMatrixMode(GL_MODELVIEW)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def main():
    for i in range(500):
        glClear(GL_COLOR_BUFFER_BIT)
        glColor4f(0.0, 0.0, (50+i) / 255, .15)
        square(i)

        screen = glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT)
        screen = np.copy(np.frombuffer(screen, np.float32)).reshape((h, w) + (3,))[::-1, :]  # Read buffer and flip Y

        cv2.imshow("a", screen[::-1, :])
        cv2.imwrite("/Core/Workspaces/jetBrains/pycharm/stochastic-fractal-search/asd.png", screen)
        cv2.waitKey(1)

    # image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    # image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)
    #

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
