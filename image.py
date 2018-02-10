import cv2
import numpy as np

def prepare(image):
    for i in range(len(image)):
        prepared = prepare_for_ann(image[i])
        image[i] = cv2.resize(prepared, (28, 28), interpolation=cv2.INTER_NEAREST)
    return image


def prepare_for_ann(image):
    ret, thresh = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cont = thresh[y:y + h, x:x + w]
    return cont


def dilate_image(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def load_rgb_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def in_range(image, lower, upper):
    image = load_rgb_image(image)
    mask = cv2.inRange(image, np.array(lower), np.array(upper))
    return cv2.bitwise_and(image, image, mask=mask)


